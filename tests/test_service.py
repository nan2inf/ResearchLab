import time
from pathlib import Path

from researchlab.runs import LabService
from researchlab.store import Database


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

    deadline = time.time() + 60
    while time.time() < deadline:
        run = service.db.get("runs", run["id"])
        if run["status"] not in {"queued", "running"}:
            break
        time.sleep(0.2)

    assert run["status"] == "completed", service.read_log(run["id"])
    event_types = {event["type"] for event in service.read_events(run["id"])["events"]}
    assert {"metrics", "matrix", "table", "artifact"} <= event_types
    assert service.artifact_path(run["id"], "artifacts/latest.pt").is_file()

