from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    def check(self) -> "CommandResult":
        if self.returncode:
            raise RuntimeError(self.stderr.strip() or self.stdout.strip() or "command failed")
        return self


class LocalExecutor:
    is_remote = False

    def run(self, command: Sequence[str], timeout: int = 30) -> CommandResult:
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return CommandResult(127, "", f"command not found: {command[0]}")
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", f"command timed out after {timeout}s")
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def run_script(self, script: str, timeout: int = 30) -> CommandResult:
        if os.name == "nt":
            return self.run(["powershell", "-NoProfile", "-Command", script], timeout)
        return self.run(["bash", "-lc", script], timeout)


class SSHExecutor:
    is_remote = True

    def __init__(self, alias: str, shell_init: str = ""):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", alias) or alias.startswith("-"):
            raise ValueError("SSH alias contains unsupported characters")
        if "\n" in shell_init or "\r" in shell_init:
            raise ValueError("shell initialization must be one line")
        self.alias = alias
        self.shell_init = shell_init.strip()

    def run(self, command: Sequence[str], timeout: int = 30) -> CommandResult:
        return self.run_script(shlex.join(command), timeout)

    def run_script(self, script: str, timeout: int = 30) -> CommandResult:
        if self.shell_init:
            script = f"{self.shell_init}; {script}"
        remote_command = f"bash -lc {shlex.quote(script)}"
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.alias, remote_command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return CommandResult(result.returncode, result.stdout, result.stderr)

    def upload(self, local: str | Path, remote: str, timeout: int = 300) -> None:
        target = f"{self.alias}:{shlex.quote(remote)}"
        result = subprocess.run(
            ["scp", "-q", str(local), target],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "SCP upload failed")

    def download(self, remote: str, local: str | Path, timeout: int = 300) -> None:
        source = f"{self.alias}:{shlex.quote(remote)}"
        result = subprocess.run(
            ["scp", "-q", source, str(local)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "SCP download failed")


Executor = LocalExecutor | SSHExecutor


def executor_for(server: dict | None) -> Executor:
    if server is None:
        return LocalExecutor()
    return SSHExecutor(server["ssh_alias"], server.get("shell_init", ""))


def test_connection(executor: Executor) -> dict:
    if executor.is_remote:
        result = executor.run_script("printf 'ok\\n'; uname -srm", timeout=15).check()
        lines = result.stdout.strip().splitlines()
        return {"ok": bool(lines and lines[0] == "ok"), "system": " ".join(lines[1:])}
    return {"ok": True, "system": platform.platform()}


def _parse_csv_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def _number(value: str, default: float | None = 0.0) -> float | None:
    if value.strip().lower() in {"", "n/a", "[n/a]", "na", "none", "unknown"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_nvidia_smi(gpus: str, processes: str = "", users: dict[int, str] | None = None) -> list[dict]:
    users = users or {}
    devices: list[dict] = []
    by_uuid: dict[str, dict] = {}
    for line in gpus.splitlines():
        if not line.strip():
            continue
        parts = _parse_csv_line(line)
        if len(parts) < 7:
            continue
        index, uuid, name, total, used, utilization, temperature = parts[:7]
        try:
            device_id = int(index)
        except ValueError:
            continue
        device = {
            "backend": "cuda",
            "id": device_id,
            "uuid": uuid,
            "name": name,
            "memory_total_mb": int(_number(total) or 0),
            "memory_used_mb": int(_number(used) or 0),
            "utilization_percent": _number(utilization) or 0.0,
            "temperature_c": _number(temperature, None),
            "processes": [],
        }
        devices.append(device)
        by_uuid[uuid] = device
    for line in processes.splitlines():
        if not line.strip():
            continue
        parts = _parse_csv_line(line)
        if len(parts) < 4 or parts[0] not in by_uuid:
            continue
        uuid, pid, name, memory = parts[:4]
        try:
            pid_value = int(pid)
        except ValueError:
            continue
        by_uuid[uuid]["processes"].append(
            {
                "pid": pid_value,
                "user": users.get(pid_value),
                "name": name,
                "memory_mb": int(_number(memory) or 0),
            }
        )
    for device in devices:
        device["busy"] = bool(
            device["processes"]
            or device["memory_used_mb"] > 512
            or device["utilization_percent"] > 5
        )
    return devices


def parse_npu_smi(output: str) -> list[dict]:
    """Best-effort parser; raw output is retained because formats vary by Atlas generation."""
    devices: dict[int, dict] = {}
    process_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if "process id" in lowered or "process name" in lowered:
            process_section = True
            continue
        if not line.startswith("|") or set(line) <= {"|", "+", "-", "="}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if process_section:
            numbers = re.findall(r"\b\d+\b", cells[0] if cells else "")
            joined = " ".join(cells)
            pid_match = re.search(r"\b(\d{2,})\b", joined)
            if numbers and pid_match:
                device_id = int(numbers[0])
                if device_id in devices:
                    devices[device_id]["processes"].append(
                        {"pid": int(pid_match.group(1)), "user": None, "name": joined}
                    )
            continue
        first = cells[0] if cells else ""
        match = re.match(r"(\d+)\s+([A-Za-z0-9_.-]+)", first)
        if not match:
            continue
        device_id = int(match.group(1))
        name = match.group(2)
        memory_match = re.search(r"(\d+)\s*/\s*(\d+)", raw_line)
        percent_values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", raw_line)]
        device = devices.setdefault(
            device_id,
            {
                "backend": "npu",
                "id": device_id,
                "name": name,
                "memory_total_mb": None,
                "memory_used_mb": None,
                "utilization_percent": None,
                "temperature_c": None,
                "processes": [],
            },
        )
        if memory_match:
            device["memory_used_mb"] = int(memory_match.group(1))
            device["memory_total_mb"] = int(memory_match.group(2))
        if percent_values:
            device["utilization_percent"] = percent_values[0]
    for device in devices.values():
        used = device["memory_used_mb"] or 0
        utilization = device["utilization_percent"] or 0
        device["busy"] = bool(device["processes"] or used > 512 or utilization > 5)
    return list(devices.values())


def _process_users(executor: Executor, pids: list[int]) -> dict[int, str]:
    if not pids or (not executor.is_remote and os.name == "nt"):
        return {}
    result = executor.run(["ps", "-o", "pid=,user=", "-p", ",".join(map(str, pids))])
    users = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                users[int(parts[0])] = parts[1].strip()
    return users


def hardware_snapshot(executor: Executor | None = None) -> dict:
    executor = executor or LocalExecutor()
    if executor.is_remote:
        system_result = executor.run_script(
            "uname -srm; printf '\\n'; getconf _NPROCESSORS_ONLN; "
            "awk '/MemTotal/ {printf \"%d\\n\", $2/1024}' /proc/meminfo"
        )
        lines = system_result.stdout.splitlines()
        system = {
            "platform": lines[0] if lines else "unknown",
            "cpu_count": int(lines[2]) if len(lines) > 2 and lines[2].isdigit() else None,
            "memory_total_mb": int(lines[3]) if len(lines) > 3 and lines[3].isdigit() else None,
        }
    else:
        system = {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "memory_total_mb": None,
        }

    gpu_result = executor.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_result = executor.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    raw_processes = process_result.stdout if process_result.returncode == 0 else ""
    pids = []
    for line in raw_processes.splitlines():
        parts = _parse_csv_line(line)
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(int(parts[1]))
    users = _process_users(executor, pids)
    devices = (
        parse_nvidia_smi(gpu_result.stdout, raw_processes, users)
        if gpu_result.returncode == 0
        else []
    )

    npu_result = executor.run(["npu-smi", "info"])
    npu_devices = parse_npu_smi(npu_result.stdout) if npu_result.returncode == 0 else []
    devices.extend(npu_devices)
    return {
        "system": system,
        "devices": devices,
        "tools": {
            "nvidia_smi": gpu_result.returncode == 0,
            "npu_smi": npu_result.returncode == 0,
        },
        "warnings": [
            warning
            for warning in [
                None
                if devices
                else "No NVIDIA or Ascend accelerator was detected.",
                "Ascend status parsing is best-effort; inspect raw npu-smi output if a field is unknown."
                if npu_result.returncode == 0 and not npu_devices
                else None,
            ]
            if warning
        ],
    }


PYTHON_PROBE = r"""
import importlib.util, json, platform, sys
result = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": None,
    "cuda_available": False,
    "cuda_devices": 0,
    "npu_available": False,
    "npu_devices": 0,
}
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_devices"] = int(torch.cuda.device_count())
    if importlib.util.find_spec("torch_npu"):
        import torch_npu
        result["torch_npu"] = getattr(torch_npu, "__version__", "installed")
        result["npu_available"] = bool(torch.npu.is_available())
        result["npu_devices"] = int(torch.npu.device_count())
except Exception as exc:
    result["probe_error"] = f"{type(exc).__name__}: {exc}"
print("RESEARCHLAB_PROBE=" + json.dumps(result))
""".strip()


def probe_python(executor: Executor, python: str) -> dict:
    result = executor.run([python, "-c", PYTHON_PROBE], timeout=30)
    marker = "RESEARCHLAB_PROBE="
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(marker):
            data = json.loads(line[len(marker) :])
            data["ok"] = result.returncode == 0
            return data
    return {
        "executable": python,
        "ok": False,
        "probe_error": result.stderr.strip() or "Python probe produced no result",
    }


def discover_environments(executor: Executor | None = None) -> list[dict]:
    executor = executor or LocalExecutor()
    candidates: list[tuple[str, str]] = []
    conda_result = executor.run(["conda", "env", "list", "--json"], timeout=30)
    if conda_result.returncode == 0:
        try:
            data = json.loads(conda_result.stdout[conda_result.stdout.index("{") :])
            for prefix in data.get("envs", []):
                python = (
                    str(Path(prefix) / "python.exe")
                    if not executor.is_remote and os.name == "nt"
                    else f"{prefix.rstrip('/')}/bin/python"
                )
                candidates.append((Path(prefix).name or "base", python))
        except (ValueError, json.JSONDecodeError):
            pass
    if executor.is_remote:
        current = executor.run_script("command -v python3 || command -v python")
        if current.returncode == 0 and current.stdout.strip():
            candidates.append(("system", current.stdout.strip().splitlines()[-1]))
    else:
        candidates.append(("current", sys.executable))
    seen = set()
    environments = []
    for name, python in candidates:
        if python in seen:
            continue
        seen.add(python)
        environments.append({"name": name, **probe_python(executor, python)})
    return environments


def require_local_tools() -> dict[str, bool]:
    return {name: shutil.which(name) is not None for name in ("ssh", "scp", "git", "tar")}
