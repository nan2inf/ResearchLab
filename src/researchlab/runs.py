from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .config import ProjectManifest, build_command, check_environment, load_manifest, slugify
from .store import Database, app_home, utc_now
from .system import (
    SSHExecutor,
    discover_environments,
    executor_for,
    hardware_snapshot,
    test_connection,
)


DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "logs",
    "runs",
    "artifacts",
    "checkpoints",
    "*.pyc",
]


def timestamp_name(name: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = slugify(name, "") if name.strip() else ""
    return f"{timestamp}_{suffix}" if suffix else timestamp


def _gitignore_patterns(root: Path) -> list[str]:
    path = root / ".gitignore"
    if not path.is_file():
        return []
    return [
        line.strip().rstrip("/")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]


def _excluded(relative: Path, patterns: list[str]) -> bool:
    text = relative.as_posix()
    for pattern in patterns:
        pattern = pattern.lstrip("/")
        if (
            fnmatch.fnmatch(text, pattern)
            or fnmatch.fnmatch(relative.name, pattern)
            or any(fnmatch.fnmatch(part, pattern) for part in relative.parts)
        ):
            return True
    return False


def create_source_archive(root: Path, target: Path, excludes: list[str]) -> dict:
    root = root.resolve()
    patterns = [*DEFAULT_EXCLUDES, *_gitignore_patterns(root), *excludes]
    file_count = 0
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if _excluded(relative, patterns) or path.is_symlink() or not path.is_file():
                continue
            archive.add(path, arcname=relative.as_posix(), recursive=False)
            file_count += 1
        runtime_root = Path(__file__).parent
        archive.add(
            runtime_root / "__init__.py",
            arcname=".researchlab_runtime/researchlab/__init__.py",
            recursive=False,
        )
        archive.add(
            runtime_root / "sdk.py",
            arcname=".researchlab_runtime/researchlab/sdk.py",
            recursive=False,
        )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": target.stat().st_size, "files": file_count}


def extract_source_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError(f"unsafe source archive member: {member.name}")
        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)


def source_files(root: Path, excludes: list[str] | None = None, limit: int = 5000) -> list[dict]:
    root = root.resolve()
    patterns = [*DEFAULT_EXCLUDES, *_gitignore_patterns(root), *(excludes or [])]
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if _excluded(relative, patterns) or path.is_symlink() or not path.is_file():
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
            }
        )
        if len(files) >= limit:
            break
    return files


def read_source_file(root: Path, relative: str, max_bytes: int = 1024 * 1024) -> dict:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("invalid source path")
    root = root.resolve()
    target = (root / Path(*relative_path.parts)).resolve()
    if not target.is_relative_to(root) or target.is_symlink() or not target.is_file():
        raise FileNotFoundError(relative)
    size = target.stat().st_size
    if size > max_bytes:
        raise ValueError(f"source file is larger than {max_bytes} bytes")
    content = target.read_bytes()
    if b"\0" in content:
        raise ValueError("binary source files cannot be previewed")
    return {
        "path": relative_path.as_posix(),
        "size": size,
        "content": content.decode("utf-8", errors="replace"),
    }


class LabService:
    def __init__(self, database: Database | None = None):
        self.db = database or Database()
        self._processes: dict[str, subprocess.Popen] = {}
        self._process_lock = threading.Lock()
        self._remote_monitors: set[str] = set()
        self._recover_runs()
        self._recover_operations()

    def _recover_runs(self) -> None:
        for run in self.db.list("runs"):
            if run["status"] in {"queued", "running"}:
                self.db.update("runs", run["id"], status="unknown")

    def _recover_operations(self) -> None:
        for operation in self.db.list("operations"):
            if operation["status"] in {"queued", "running"}:
                self.db.update(
                    "operations",
                    operation["id"],
                    status="failed",
                    message="Interrupted when the local controller stopped.",
                    error="controller_restarted",
                    finished_at=utc_now(),
                )

    def _start_operation(self, kind: str, target_id: str | None, action) -> dict:
        operation = self.db.insert(
            "operations",
            {
                "kind": kind,
                "target_id": target_id,
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
            },
        )

        def work() -> None:
            self.db.update(
                "operations",
                operation["id"],
                status="running",
                progress=0.1,
                message="Running",
                started_at=utc_now(),
            )
            try:
                result = action()
                self.db.update(
                    "operations",
                    operation["id"],
                    status="completed",
                    progress=1.0,
                    message="Completed",
                    result=result,
                    finished_at=utc_now(),
                )
            except Exception as exc:
                self.db.update(
                    "operations",
                    operation["id"],
                    status="failed",
                    message="Failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=utc_now(),
                )

        threading.Thread(target=work, daemon=True).start()
        return self.db.get("operations", operation["id"])

    def create_version_operation(
        self, project_id: str, *, name: str = "", server_id: str | None = None
    ) -> dict:
        self.db.get("projects", project_id)
        if server_id:
            self.db.get("servers", server_id)
        return self._start_operation(
            "create_version",
            project_id,
            lambda: self.create_version(project_id, name=name, server_id=server_id),
        )

    def create_environment_operation(
        self,
        server_id: str | None,
        *,
        name: str,
        python_version: str = "3.11",
        pip_packages: list[str] | None = None,
        install_command: str = "",
    ) -> dict:
        if server_id:
            self.db.get("servers", server_id)
        return self._start_operation(
            "create_environment",
            server_id,
            lambda: self.create_conda_environment(
                server_id,
                name=name,
                python_version=python_version,
                pip_packages=pip_packages,
                install_command=install_command,
            ),
        )

    def add_server(
        self,
        *,
        name: str,
        ssh_alias: str,
        remote_root: str,
        shell_init: str = "",
    ) -> dict:
        values = self._validated_server(
            name=name,
            ssh_alias=ssh_alias,
            remote_root=remote_root,
            shell_init=shell_init,
        )
        return self.db.insert("servers", values)

    def update_server(self, server_id: str, **changes: str) -> dict:
        current = self.db.get("servers", server_id)
        values = self._validated_server(
            name=changes.get("name", current["name"]),
            ssh_alias=changes.get("ssh_alias", current["ssh_alias"]),
            remote_root=changes.get("remote_root", current["remote_root"]),
            shell_init=changes.get("shell_init", current["shell_init"]),
        )
        return self.db.update("servers", server_id, **values)

    @staticmethod
    def _validated_server(
        *, name: str, ssh_alias: str, remote_root: str, shell_init: str
    ) -> dict[str, str]:
        path = PurePosixPath(remote_root)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("remote_root must be an absolute Linux path")
        if not name.strip():
            raise ValueError("server name cannot be blank")
        executor = SSHExecutor(ssh_alias, shell_init)
        connection = test_connection(executor)
        if not connection["ok"]:
            raise RuntimeError("SSH connection test failed")
        return {
            "name": name.strip(),
            "ssh_alias": ssh_alias,
            "remote_root": str(path),
            "shell_init": shell_init.strip(),
        }

    def server_diagnostics(self, server_id: str | None = None) -> dict:
        server = self.db.get("servers", server_id) if server_id else None
        executor = executor_for(server)
        return {
            "connection": test_connection(executor),
            "hardware": hardware_snapshot(executor),
            "environments": discover_environments(executor),
        }

    def probe_environment(self, server_id: str | None, python: str) -> dict:
        server = self.db.get("servers", server_id) if server_id else None
        python = python.strip()
        if not python or "\n" in python or "\r" in python:
            raise ValueError("python executable must be a non-empty single-line path")
        return probe_environment(executor_for(server), python)

    def version_environments(self, version_id: str, backend: str = "cpu") -> list[dict]:
        version = self.db.get("versions", version_id)
        manifest = ProjectManifest.model_validate(version["manifest"])
        server = self.db.get("servers", version["server_id"]) if version["server_id"] else None
        environments = discover_environments(executor_for(server))
        return [
            {
                **environment,
                **check_environment(manifest.environment, environment, backend),
            }
            for environment in environments
        ]

    def add_project(self, source_path: str) -> dict:
        manifest, root = load_manifest(source_path)
        return self.db.insert(
            "projects",
            {
                "name": manifest.name,
                "source_path": str(root),
                "manifest": manifest.model_dump(mode="json"),
            },
        )

    def inspect_project(self, source_path: str) -> dict:
        manifest, root = load_manifest(source_path)
        files = source_files(root, manifest.exclude)
        return {
            "root": str(root),
            "manifest": manifest.model_dump(mode="json"),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(file["size"] for file in files),
            "truncated": len(files) >= 5000,
        }

    def project_files(self, project_id: str) -> list[dict]:
        project = self.db.get("projects", project_id)
        manifest, root = load_manifest(project["source_path"])
        return source_files(root, manifest.exclude)

    def version_files(self, version_id: str) -> list[dict]:
        version = self.db.get("versions", version_id)
        return source_files(Path(version["source_path"]))

    def read_version_source(self, version_id: str, relative: str) -> dict:
        version = self.db.get("versions", version_id)
        return read_source_file(Path(version["source_path"]), relative)

    def create_version(
        self,
        project_id: str,
        *,
        name: str = "",
        server_id: str | None = None,
    ) -> dict:
        project = self.db.get("projects", project_id)
        manifest, root = load_manifest(project["source_path"])
        version_name = timestamp_name(name)
        local_version = (
            app_home()
            / "projects"
            / slugify(project["name"])
            / "versions"
            / version_name
        )
        if local_version.exists():
            local_version = local_version.with_name(f"{version_name}_{Database.new_id()[:6]}")
            version_name = local_version.name
        local_version.mkdir(parents=True, exist_ok=False)
        archive_path = local_version / "source.tar.gz"
        metadata = create_source_archive(root, archive_path, manifest.exclude)
        local_source = local_version / "source"
        extract_source_archive(archive_path, local_source)

        remote_path = None
        server = self.db.get("servers", server_id) if server_id else None
        if server:
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            remote_parent = (
                PurePosixPath(server["remote_root"])
                / "projects"
                / slugify(project["name"])
                / "versions"
            )
            remote_version = remote_parent / version_name
            remote_source = remote_version / "source"
            remote_archive = remote_version / "source.tar.gz"
            executor.run_script(
                f"mkdir -p {shlex.quote(str(remote_parent))} && "
                f"mkdir {shlex.quote(str(remote_version))}"
            ).check()
            try:
                executor.upload(archive_path, str(remote_archive))
                executor.run_script(
                    f"mkdir {shlex.quote(str(remote_source))} && "
                    f"tar -xzf {shlex.quote(str(remote_archive))} "
                    f"-C {shlex.quote(str(remote_source))} && "
                    f"rm -f {shlex.quote(str(remote_archive))}"
                ).check()
            except Exception:
                # Keep the local immutable copy for diagnosis; never overwrite a remote version.
                raise
            remote_path = str(remote_version)
        archive_path.unlink()

        return self.db.insert(
            "versions",
            {
                "project_id": project_id,
                "name": version_name,
                "source_path": str(local_source),
                "server_id": server_id,
                "remote_path": remote_path,
                "manifest": manifest.model_dump(mode="json"),
                "metadata": metadata,
            },
        )

    def start_run(
        self,
        version_id: str,
        *,
        task_name: str,
        params: dict[str, Any],
        devices: list[int] | None = None,
        backend: str = "cpu",
        python: str | None = None,
        name: str = "",
    ) -> dict:
        resolved = self._resolve_run(
            version_id,
            task_name=task_name,
            params=params,
            devices=devices,
            backend=backend,
            python=python,
        )
        version = resolved["version"]
        server = resolved["server"]
        devices = resolved["devices"]
        command = resolved["command"]
        resolved_params = resolved["params"]
        run_name = timestamp_name(name)
        local_run_path = Path(version["source_path"]).parent / "runs" / run_name
        local_run_path.mkdir(parents=True, exist_ok=False)
        remote_run_path = (
            str(PurePosixPath(version["remote_path"]) / "runs" / run_name)
            if version["remote_path"]
            else None
        )
        run = self.db.insert(
            "runs",
            {
                "project_id": version["project_id"],
                "version_id": version_id,
                "server_id": version["server_id"],
                "name": run_name,
                "task": task_name,
                "status": "queued",
                "params": resolved_params,
                "devices": devices,
                "backend": backend,
                "run_path": str(local_run_path),
                "remote_path": remote_run_path,
                "pid": None,
                "command": command,
                "exit_code": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
            },
        )
        try:
            return (
                self._start_remote(run, version, server, command)
                if server
                else self._start_local(run, version, command)
            )
        except Exception as exc:
            self.db.update(
                "runs",
                run["id"],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=utc_now(),
            )
            raise

    def _resolve_run(
        self,
        version_id: str,
        *,
        task_name: str,
        params: dict[str, Any],
        devices: list[int] | None,
        backend: str,
        python: str | None,
    ) -> dict:
        version = self.db.get("versions", version_id)
        manifest = ProjectManifest.model_validate(version["manifest"])
        if task_name not in manifest.tasks:
            raise ValueError(f"unknown task: {task_name}")
        if backend not in {"cpu", "cuda", "npu"}:
            raise ValueError("backend must be cpu, cuda, or npu")
        selected_devices = sorted(set(devices or []))
        if backend != "cpu" and not selected_devices:
            raise ValueError("select at least one accelerator")
        server = self.db.get("servers", version["server_id"]) if version["server_id"] else None
        selected_python = python or ("python3" if server else sys.executable)
        command, resolved_params = build_command(
            manifest.tasks[task_name],
            params,
            python=selected_python,
            devices=selected_devices,
        )
        return {
            "version": version,
            "manifest": manifest,
            "server": server,
            "python": selected_python,
            "devices": selected_devices,
            "command": command,
            "params": resolved_params,
        }

    def validate_run(
        self,
        version_id: str,
        *,
        task_name: str,
        params: dict[str, Any],
        devices: list[int] | None = None,
        backend: str = "cpu",
        python: str | None = None,
    ) -> dict:
        resolved = self._resolve_run(
            version_id,
            task_name=task_name,
            params=params,
            devices=devices,
            backend=backend,
            python=python,
        )
        executor = executor_for(resolved["server"])
        environment = probe_environment(executor, resolved["python"])
        compatibility = check_environment(
            resolved["manifest"].environment, environment, backend
        )
        hardware = hardware_snapshot(executor)
        warnings: list[str] = []
        if backend != "cpu":
            available = {
                device["id"]: device
                for device in hardware["devices"]
                if device["backend"] == backend
            }
            missing = [device for device in resolved["devices"] if device not in available]
            if missing:
                raise ValueError(f"selected {backend} devices were not detected: {missing}")
            busy = [device for device in resolved["devices"] if available[device]["busy"]]
            if busy:
                warnings.append(f"selected {backend} devices are currently busy: {busy}")
        warnings.extend(hardware.get("warnings", []))
        return {
            "valid": compatibility["compatible"],
            "command": resolved["command"],
            "params": resolved["params"],
            "environment": {**environment, **compatibility},
            "hardware": hardware,
            "warnings": warnings,
        }

    def rerun(self, run_id: str, name: str = "") -> dict:
        previous = self.db.get("runs", run_id)
        python = previous["command"][0] if previous["command"] else None
        return self.start_run(
            previous["version_id"],
            task_name=previous["task"],
            params=previous["params"],
            devices=previous["devices"],
            backend=previous["backend"],
            python=python,
            name=name or f"rerun-{previous['name']}",
        )

    def refresh_run(self, run_id: str) -> dict:
        run = self.db.get("runs", run_id)
        if run["status"] not in {"queued", "running", "unknown"}:
            return run
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            exit_path = str(PurePosixPath(run["remote_path"]) / "exit_code")
            result = executor.run_script(
                f"if [ -f {shlex.quote(exit_path)} ]; then cat {shlex.quote(exit_path)}; "
                f"elif kill -0 {run['pid']} 2>/dev/null; then echo RUNNING; "
                "else echo LOST; fi"
            ).check()
            state = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "LOST"
        else:
            with self._process_lock:
                process = self._processes.get(run_id)
            if process is not None:
                code = process.poll()
                state = "RUNNING" if code is None else str(code)
            elif run["pid"]:
                try:
                    os.kill(run["pid"], 0)
                    state = "RUNNING"
                except (OSError, PermissionError):
                    state = "LOST"
            else:
                state = "LOST"
        if state == "RUNNING":
            if run["server_id"] and run["pid"]:
                self._ensure_remote_monitor(
                    run_id, executor, run["remote_path"], run["pid"]
                )
            return self.db.update("runs", run_id, status="running")
        if re.fullmatch(r"-?\d+", state):
            exit_code = int(state)
            return self.db.update(
                "runs",
                run_id,
                status="completed" if exit_code == 0 else "failed",
                exit_code=exit_code,
                finished_at=utc_now(),
            )
        return self.db.update("runs", run_id, status="unknown")

    def list_artifacts(self, run_id: str) -> list[dict]:
        run = self.db.get("runs", run_id)
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            artifact_root = str(PurePosixPath(run["remote_path"]) / "artifacts")
            result = executor.run_script(
                f"test ! -d {shlex.quote(artifact_root)} || "
                f"find {shlex.quote(artifact_root)} -type f -printf '%P\\t%s\\n'"
            ).check()
            records = []
            for line in result.stdout.splitlines():
                path, separator, size = line.rpartition("\t")
                if separator and size.isdigit():
                    records.append({"path": f"artifacts/{path}", "size": int(size)})
            return records
        artifact_root = Path(run["run_path"]) / "artifacts"
        if not artifact_root.is_dir():
            return []
        return [
            {
                "path": f"artifacts/{path.relative_to(artifact_root).as_posix()}",
                "size": path.stat().st_size,
            }
            for path in sorted(artifact_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]

    def _runtime_environment(self, run: dict, source: str) -> dict[str, str]:
        environment = {
            "RESEARCHLAB_RUN_ID": run["id"],
            "RESEARCHLAB_RUN_DIR": run["remote_path"] or run["run_path"],
            "PYTHONUNBUFFERED": "1",
        }
        runtime = (
            f"{source}/.researchlab_runtime"
            if run["remote_path"]
            else str(Path(source) / ".researchlab_runtime")
        )
        environment["PYTHONPATH"] = runtime
        if run["backend"] == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, run["devices"]))
        elif run["backend"] == "npu":
            environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, run["devices"]))
        return environment

    def _start_local(self, run: dict, version: dict, command: list[str]) -> dict:
        source = version["source_path"]
        environment = os.environ.copy()
        runtime_environment = self._runtime_environment(run, source)
        existing_pythonpath = environment.get("PYTHONPATH")
        if existing_pythonpath:
            runtime_environment["PYTHONPATH"] += os.pathsep + existing_pythonpath
        environment.update(runtime_environment)
        log_path = Path(run["run_path"]) / "train.log"
        log_handle = log_path.open("ab")
        kwargs: dict[str, Any] = {"start_new_session": os.name != "nt"}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        finally:
            log_handle.close()
        with self._process_lock:
            self._processes[run["id"]] = process
        updated = self.db.update(
            "runs",
            run["id"],
            status="running",
            pid=process.pid,
            started_at=utc_now(),
        )
        threading.Thread(
            target=self._monitor_local,
            args=(run["id"], process),
            daemon=True,
        ).start()
        return updated

    def _start_remote(
        self,
        run: dict,
        version: dict,
        server: dict,
        command: list[str],
    ) -> dict:
        executor = executor_for(server)
        assert isinstance(executor, SSHExecutor)
        remote_source = str(PurePosixPath(version["remote_path"]) / "source")
        remote_run = run["remote_path"]
        environment = self._runtime_environment(run, remote_source)
        exports = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in environment.items()
        )
        exit_code_path = str(PurePosixPath(remote_run) / "exit_code")
        log_path = str(PurePosixPath(remote_run) / "train.log")
        inner = (
            f"cd {shlex.quote(remote_source)} && "
            f"env {exports} {shlex.join(command)}; "
            "code=$?; "
            f"printf '%s\\n' \"$code\" > {shlex.quote(exit_code_path)}; "
            "exit \"$code\""
        )
        script = (
            f"mkdir -p {shlex.quote(remote_run)} || exit 1; "
            f"nohup setsid bash -lc {shlex.quote(inner)} "
            f"> {shlex.quote(log_path)} 2>&1 < /dev/null & "
            "pid=$!; printf '%s\\n' \"$pid\""
        )
        result = executor.run_script(script).check()
        pid_lines = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
        if not pid_lines:
            raise RuntimeError(f"remote launcher returned no PID: {result.stdout}")
        pid = int(pid_lines[-1])
        updated = self.db.update(
            "runs",
            run["id"],
            status="running",
            pid=pid,
            started_at=utc_now(),
        )
        self._ensure_remote_monitor(run["id"], executor, remote_run, pid)
        return updated

    def _ensure_remote_monitor(
        self, run_id: str, executor: SSHExecutor, remote_run: str, pid: int
    ) -> None:
        with self._process_lock:
            if run_id in self._remote_monitors:
                return
            self._remote_monitors.add(run_id)

        def monitor() -> None:
            try:
                self._monitor_remote(run_id, executor, remote_run, pid)
            finally:
                with self._process_lock:
                    self._remote_monitors.discard(run_id)

        threading.Thread(target=monitor, daemon=True).start()

    def _monitor_local(self, run_id: str, process: subprocess.Popen) -> None:
        exit_code = process.wait()
        current = self.db.get("runs", run_id)
        status = "cancelled" if current["status"] == "cancelled" else (
            "completed" if exit_code == 0 else "failed"
        )
        self.db.update(
            "runs",
            run_id,
            status=status,
            exit_code=exit_code,
            finished_at=utc_now(),
        )
        with self._process_lock:
            self._processes.pop(run_id, None)

    def _monitor_remote(
        self,
        run_id: str,
        executor: SSHExecutor,
        remote_run: str,
        pid: int,
    ) -> None:
        exit_path = str(PurePosixPath(remote_run) / "exit_code")
        while True:
            result = executor.run_script(
                f"if [ -f {shlex.quote(exit_path)} ]; then "
                f"cat {shlex.quote(exit_path)}; "
                f"elif kill -0 {pid} 2>/dev/null; then echo RUNNING; "
                "else echo LOST; fi",
                timeout=20,
            )
            state = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "LOST"
            if state == "RUNNING":
                time.sleep(5)
                continue
            current = self.db.get("runs", run_id)
            if state.isdigit():
                exit_code = int(state)
                status = "cancelled" if current["status"] == "cancelled" else (
                    "completed" if exit_code == 0 else "failed"
                )
            else:
                exit_code = None
                status = "cancelled" if current["status"] == "cancelled" else "unknown"
            self.db.update(
                "runs",
                run_id,
                status=status,
                exit_code=exit_code,
                finished_at=utc_now(),
            )
            return

    def cancel_run(self, run_id: str) -> dict:
        run = self.db.get("runs", run_id)
        if run["status"] not in {"queued", "running", "unknown"}:
            return run
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            if run["pid"]:
                executor.run_script(
                    f"kill -TERM -- -{run['pid']} 2>/dev/null || "
                    f"kill -TERM {run['pid']} 2>/dev/null || true"
                )
        elif run["pid"]:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(run["pid"]), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                try:
                    os.killpg(run["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return self.db.update("runs", run_id, status="cancelled", finished_at=utc_now())

    def read_events(self, run_id: str, after: int = 0, limit: int = 1000) -> dict:
        run = self.db.get("runs", run_id)
        after = max(0, after)
        limit = min(max(1, limit), 5000)
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            path = str(PurePosixPath(run["remote_path"]) / "events.jsonl")
            result = executor.run_script(
                f"test -f {shlex.quote(path)} && "
                f"tail -n +{after + 1} {shlex.quote(path)} | head -n {limit} || true"
            )
            lines = result.stdout.splitlines()
        else:
            path = Path(run["run_path"]) / "events.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()[after : after + limit] if path.is_file() else []
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"events": events, "next": after + len(lines)}

    def read_log(self, run_id: str, lines: int = 200) -> str:
        run = self.db.get("runs", run_id)
        lines = min(max(1, lines), 2000)
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            path = str(PurePosixPath(run["remote_path"]) / "train.log")
            return executor.run_script(
                f"test -f {shlex.quote(path)} && tail -n {lines} {shlex.quote(path)} || true"
            ).stdout
        path = Path(run["run_path"]) / "train.log"
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def artifact_path(self, run_id: str, relative: str) -> Path:
        run = self.db.get("runs", run_id)
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("invalid artifact path")
        if run["server_id"]:
            server = self.db.get("servers", run["server_id"])
            executor = executor_for(server)
            assert isinstance(executor, SSHExecutor)
            cache = app_home() / "cache" / run_id / Path(*relative_path.parts)
            cache.parent.mkdir(parents=True, exist_ok=True)
            remote = str(PurePosixPath(run["remote_path"]) / relative_path)
            executor.download(remote, cache)
            return cache
        target = (Path(run["run_path"]) / Path(*relative_path.parts)).resolve()
        root = Path(run["run_path"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError(relative)
        return target

    def create_conda_environment(
        self,
        server_id: str | None,
        *,
        name: str,
        python_version: str = "3.11",
        pip_packages: list[str] | None = None,
        install_command: str = "",
    ) -> dict:
        server = self.db.get("servers", server_id) if server_id else None
        executor = executor_for(server)
        if not re.fullmatch(r"3\.\d{1,2}", python_version):
            raise ValueError("unsupported Python version")
        pip_packages = pip_packages or []
        if any(not package.strip() or "\n" in package or "\r" in package for package in pip_packages):
            raise ValueError("pip package specifications must be non-empty single-line values")
        if "\n" in install_command or "\r" in install_command:
            raise ValueError("install command must be one line")

        if server:
            assert isinstance(executor, SSHExecutor)
            env_path: Path | PurePosixPath = (
                PurePosixPath(server["remote_root"]) / ".envs" / slugify(name)
            )
            script = (
                f"test ! -e {shlex.quote(str(env_path))} || "
                f"{{ printf 'environment already exists\\n' >&2; exit 2; }}; "
                f"mkdir -p {shlex.quote(str(env_path.parent))} && "
                f"conda create --prefix {shlex.quote(str(env_path))} "
                f"python={shlex.quote(python_version)} -y"
            )
            executor.run_script(script, timeout=1800).check()
            python_path = str(env_path / "bin" / "python")
            if pip_packages:
                executor.run(
                    [python_path, "-m", "pip", "install", *pip_packages], timeout=1800
                ).check()
        else:
            env_path = app_home() / ".envs" / slugify(name)
            if env_path.exists():
                raise ValueError(f"environment already exists: {env_path}")
            env_path.parent.mkdir(parents=True, exist_ok=True)
            executor.run(
                ["conda", "create", "--prefix", str(env_path), f"python={python_version}", "-y"],
                timeout=1800,
            ).check()
            python_path = str(
                env_path / ("python.exe" if os.name == "nt" else "bin/python")
            )
            if pip_packages:
                executor.run(
                    [python_path, "-m", "pip", "install", *pip_packages], timeout=1800
                ).check()

        if install_command.strip():
            quoted_python = (
                shlex.quote(python_path)
                if server or os.name != "nt"
                else subprocess.list2cmdline([python_path])
            )
            command = install_command.replace("{python}", quoted_python)
            executor.run_script(command, timeout=1800).check()
        return {"name": slugify(name), "prefix": str(env_path), **probe_environment(executor, python_path)}


def probe_environment(executor, python: str) -> dict:
    from .system import probe_python

    return probe_python(executor, python)
