from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, model_validator


ParameterType = Literal[
    "int", "float", "str", "bool", "path", "choice", "int_list", "float_list", "str_list"
]


class ParameterSpec(BaseModel):
    type: ParameterType = "str"
    default: Any = None
    required: bool = False
    label: str | None = None
    help: str | None = None
    choices: list[Any] = Field(default_factory=list)
    min: float | None = None
    max: float | None = None
    flag: str | None = None
    style: Literal["value", "flag"] = "value"

    @model_validator(mode="after")
    def validate_choices(self) -> "ParameterSpec":
        if self.type == "choice" and not self.choices:
            raise ValueError("choice parameters require choices")
        return self


class TaskSpec(BaseModel):
    entrypoint: str
    launcher: Literal["python", "torchrun", "auto"] = "auto"
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel):
    python: str | None = None
    torch: str | None = None
    torch_npu: str | None = None


class ProjectManifest(BaseModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    tasks: dict[str, TaskSpec]
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tasks(self) -> "ProjectManifest":
        if not self.tasks:
            raise ValueError("at least one task is required")
        return self


def slugify(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return value[:80] or fallback


def load_manifest(source: str | Path) -> tuple[ProjectManifest, Path]:
    source_path = Path(source).expanduser().resolve()
    manifest_path = source_path / "project.yaml" if source_path.is_dir() else source_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.yaml not found: {manifest_path}")
    root = manifest_path.parent.resolve()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = ProjectManifest.model_validate(data)
    for task_name, task in manifest.tasks.items():
        entrypoint = (root / task.entrypoint).resolve()
        if not entrypoint.is_relative_to(root):
            raise ValueError(f"task {task_name!r} entrypoint leaves the project directory")
        if not entrypoint.is_file():
            raise FileNotFoundError(f"task {task_name!r} entrypoint not found: {entrypoint}")
    return manifest, root


def _coerce_value(name: str, spec: ParameterSpec, value: Any) -> Any:
    if value is None:
        if spec.required and spec.default is None:
            raise ValueError(f"missing required parameter: {name}")
        value = spec.default
    if value is None:
        return None
    converters = {"int": int, "float": float, "str": str, "path": str, "choice": str}
    if spec.type == "bool":
        if isinstance(value, str):
            value = value.lower() in {"1", "true", "yes", "on"}
        else:
            value = bool(value)
    elif spec.type.endswith("_list"):
        item_type = spec.type.removesuffix("_list")
        converter = converters[item_type]
        values = value if isinstance(value, list) else str(value).split(",")
        value = [converter(item) for item in values]
    else:
        value = converters[spec.type](value)
    if spec.choices and value not in spec.choices:
        raise ValueError(f"{name} must be one of {spec.choices}")
    if isinstance(value, (int, float)):
        if spec.min is not None and value < spec.min:
            raise ValueError(f"{name} must be >= {spec.min}")
        if spec.max is not None and value > spec.max:
            raise ValueError(f"{name} must be <= {spec.max}")
    return value


def build_command(
    task: TaskSpec,
    values: dict[str, Any],
    *,
    python: str,
    devices: list[int] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    unknown = set(values) - set(task.parameters)
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
    resolved = {
        name: _coerce_value(name, spec, values.get(name))
        for name, spec in task.parameters.items()
    }
    device_count = len(devices or [])
    launcher = "torchrun" if task.launcher == "auto" and device_count > 1 else task.launcher
    if launcher == "auto":
        launcher = "python"
    if launcher == "torchrun":
        if device_count < 1:
            raise ValueError("torchrun requires at least one selected device")
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={device_count}",
            task.entrypoint,
        ]
    else:
        command = [python, task.entrypoint]
    for name, spec in task.parameters.items():
        value = resolved[name]
        if value is None:
            continue
        flag = spec.flag or f"--{name.replace('_', '-')}"
        if spec.style == "flag":
            if value:
                command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    return command, resolved


def check_environment(
    requirements: EnvironmentSpec,
    probe: dict[str, Any],
    backend: str = "cpu",
) -> dict[str, Any]:
    issues: list[str] = []

    def require_version(label: str, actual: Any, specifier: str | None) -> None:
        if not specifier:
            return
        if not actual:
            issues.append(f"{label} is not installed")
            return
        try:
            normalized = str(actual).split("+", 1)[0].split(".post", 1)[0]
            if Version(normalized) not in SpecifierSet(specifier):
                issues.append(f"{label} {actual} does not satisfy {specifier}")
        except (InvalidVersion, InvalidSpecifier):
            issues.append(f"cannot compare {label} {actual!r} with {specifier!r}")

    if not probe.get("ok"):
        issues.append(probe.get("probe_error") or "environment probe failed")
    require_version("Python", probe.get("python"), requirements.python)
    require_version("PyTorch", probe.get("torch"), requirements.torch)
    if backend == "cuda" and not probe.get("cuda_available"):
        issues.append("CUDA is unavailable in this environment")
    if backend == "npu":
        require_version("torch_npu", probe.get("torch_npu"), requirements.torch_npu or ">=0")
        if not probe.get("npu_available"):
            issues.append("Ascend NPU is unavailable in this environment")
    return {"compatible": not issues, "issues": issues}
