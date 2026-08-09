import time
from pathlib import Path

import researchlab.runs as runs_module
from researchlab.runs import LabService, create_source_archive, extract_source_archive
from researchlab.store import Database
from researchlab.system import CommandResult


def wait_for_run(service: LabService, run: dict) -> dict:
    deadline = time.time() + 60
    while time.time() < deadline:
        run = service.db.get("runs", run["id"])
        if run["status"] not in {"queued", "running"}:
            return run
        time.sleep(0.2)
    return run


def test_source_versions_keep_model_weights_unless_excluded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"weights")
    archive = tmp_path / "source.tar.gz"
    create_source_archive(source, archive, [])
    extracted = tmp_path / "extracted"
    extract_source_archive(archive, extracted)

    assert (extracted / "model.pt").read_bytes() == b"weights"


def test_local_conda_environment_uses_isolated_prefix(tmp_path: Path, monkeypatch) -> None:
    class FakeExecutor:
        is_remote = False

        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def run(self, command, timeout=30):
            self.commands.append(list(command))
            if "-c" in command:
                return CommandResult(
                    0,
                    'RESEARCHLAB_PROBE={"python":"3.11.9","executable":"fake",'
                    '"platform":"test","torch":null,"cuda_available":false,'
                    '"cuda_devices":0,"npu_available":false,"npu_devices":0}\n',
                    "",
                )
            return CommandResult(0, "", "")

        def run_script(self, script, timeout=30):
            raise AssertionError("no install command was requested")

    fake = FakeExecutor()
    monkeypatch.setattr(runs_module, "executor_for", lambda server: fake)
    monkeypatch.setattr(runs_module, "app_home", lambda: tmp_path / "state")
    service = LabService(Database(tmp_path / "app.db"))

    environment = service.create_conda_environment(
        None,
        name="Vision Lab",
        python_version="3.11",
        pip_packages=["torch>=2.7"],
    )

    expected_prefix = tmp_path / "state" / ".envs" / "vision-lab"
    assert fake.commands[0] == [
        "conda",
        "create",
        "--prefix",
        str(expected_prefix),
        "python=3.11",
        "-y",
    ]
    assert fake.commands[1][-2:] == ["install", "torch>=2.7"]
    assert environment["prefix"] == str(expected_prefix)
    assert environment["ok"] is True


def test_local_project_version_and_run_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCHLAB_HOME", str(tmp_path / "state"))
    service = LabService(Database())
    example = Path(__file__).parents[1] / "examples" / "synthetic-classification"
    project = service.add_project(str(example))
    version = service.create_version(project["id"], name="test")
    run = service.start_run(
        version["id"],
        task_name="train",
        params={
            "epochs": 2,
            "eval_every": 1,
            "batch_size": 64,
            "learning_rate": 0.01,
            "samples": 256,
            "noise_std": 0,
            "preload_device": False,
            "seed": 1,
        },
    )

    run = wait_for_run(service, run)

    assert run["status"] == "completed", service.read_log(run["id"])
    event_types = {event["type"] for event in service.read_events(run["id"])["events"]}
    assert {"metrics", "matrix", "table", "artifact"} <= event_types
    checkpoint = service.artifact_path(run["id"], "artifacts/latest.pt")
    assert checkpoint.is_file()

    offline = service.start_run(
        version["id"],
        task_name="offline",
        params={
            "checkpoint": str(checkpoint),
            "mode": "evaluate",
            "batch_size": 64,
            "samples": 256,
            "seed": 1,
        },
    )
    offline = wait_for_run(service, offline)

    assert offline["status"] == "completed", service.read_log(offline["id"])
    offline_types = {
        event["type"] for event in service.read_events(offline["id"])["events"]
    }
    assert {"metrics", "matrix"} <= offline_types

    inference = service.start_run(
        version["id"],
        task_name="offline",
        params={
            "checkpoint": str(checkpoint),
            "mode": "infer",
            "batch_size": 64,
            "samples": 256,
            "seed": 1,
        },
    )
    inference = wait_for_run(service, inference)

    assert inference["status"] == "completed", service.read_log(inference["id"])
    inference_types = {
        event["type"] for event in service.read_events(inference["id"])["events"]
    }
    assert {"table", "artifact"} <= inference_types
    assert service.artifact_path(
        inference["id"], "artifacts/predictions.json"
    ).is_file()
