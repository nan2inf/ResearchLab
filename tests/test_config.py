from pathlib import Path

from researchlab.config import (
    EnvironmentSpec,
    ParameterSpec,
    TaskSpec,
    build_command,
    check_environment,
    load_manifest,
)


def test_manifest_builds_python_and_torchrun_commands(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        """
schema_version: 1
name: demo
tasks:
  train:
    entrypoint: train.py
    launcher: auto
    parameters:
      epochs: {type: int, default: 3, min: 1}
      augment: {type: bool, default: false, style: flag}
""",
        encoding="utf-8",
    )
    manifest, _ = load_manifest(tmp_path)
    task = manifest.tasks["train"]

    command, values = build_command(task, {"augment": True}, python="python", devices=[1, 3])

    assert command[:5] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
    ]
    assert command[-4:] == ["train.py", "--epochs", "3", "--augment"]
    assert values == {"epochs": 3, "augment": True}


def test_environment_compatibility_checks_backend() -> None:
    result = check_environment(
        EnvironmentSpec(python=">=3.10", torch=">=2.0"),
        {"ok": True, "python": "3.11.8", "torch": "2.4.0+cu124", "cuda_available": False},
        "cuda",
    )

    assert result == {
        "compatible": False,
        "issues": ["CUDA is unavailable in this environment"],
    }


def test_optional_empty_path_is_not_added_to_command() -> None:
    task = TaskSpec(
        entrypoint="train.py",
        parameters={"resume": ParameterSpec(type="path")},
    )

    command, resolved = build_command(task, {"resume": ""}, python="python")

    assert command == ["python", "train.py"]
    assert resolved == {"resume": None}
