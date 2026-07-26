import json
from pathlib import Path

from researchlab.sdk import Run, Trainer


def test_run_and_trainer_write_portable_events(tmp_path: Path) -> None:
    run = Run(tmp_path, "test-run")
    Trainer(run, epochs=3, eval_every=2).fit(
        lambda epoch: {"loss": 1 / epoch},
        lambda epoch: {"acc": epoch / 3},
    )
    run.log_matrix("confusion", [[4, 1], [0, 5]], labels=["a", "b"])
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [event["split"] for event in events if event["type"] == "metrics"] == [
        "train",
        "train",
        "eval",
        "train",
        "eval",
    ]
    assert events[-1]["type"] == "matrix"
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"

