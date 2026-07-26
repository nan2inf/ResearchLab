from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


Metrics = dict[str, int | float]


class Run:
    """Dependency-free event writer used inside training processes."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.dir = Path(run_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        self.id = run_id or os.environ.get("RESEARCHLAB_RUN_ID") or uuid.uuid4().hex
        self.rank = int(os.environ.get("RANK", "0"))
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "Run":
        run_dir = os.environ.get("RESEARCHLAB_RUN_DIR")
        if not run_dir:
            raise RuntimeError("RESEARCHLAB_RUN_DIR is not set")
        return cls(run_dir)

    def log_metrics(
        self,
        metrics: Metrics,
        *,
        split: str = "train",
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        clean = {}
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                try:
                    value = value.item()
                except (AttributeError, ValueError, TypeError) as exc:
                    raise TypeError(f"metric {name!r} is not numeric") from exc
            clean[str(name)] = float(value)
        self._event("metrics", split=split, epoch=epoch, step=step, metrics=clean)

    def log_image(self, path: str | Path, *, name: str | None = None) -> None:
        relative = self._copy_artifact(path, name)
        self._event("artifact", artifact_type="image", name=name or Path(path).name, path=relative)

    def log_file(self, path: str | Path, *, name: str | None = None) -> None:
        relative = self._copy_artifact(path, name)
        self._event("artifact", artifact_type="file", name=name or Path(path).name, path=relative)

    def log_matrix(
        self,
        name: str,
        values: list[list[int | float]],
        *,
        labels: list[str] | None = None,
    ) -> None:
        self._event("matrix", name=name, values=values, labels=labels or [])

    def log_table(
        self,
        name: str,
        columns: list[str],
        rows: list[list[Any]],
    ) -> None:
        self._event("table", name=name, columns=columns, rows=rows)

    def set_status(self, status: str, **details: Any) -> None:
        if self.rank != 0:
            return
        payload = {
            "run_id": self.id,
            "status": status,
            "updated_at": time.time(),
            **details,
        }
        target = self.dir / "status.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        self._event("status", status=status, **details)

    def _copy_artifact(self, path: str | Path, name: str | None) -> str:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        safe_name = Path(name or source.name).name
        target = self.dir / "artifacts" / safe_name
        if source != target:
            shutil.copy2(source, target)
        return target.relative_to(self.dir).as_posix()

    def _event(self, event_type: str, **payload: Any) -> None:
        if self.rank != 0:
            return
        event = {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "time": time.time(),
            **payload,
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, (self.dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


class Trainer:
    """Optional scheduler around user-owned epoch functions."""

    def __init__(self, run: Run, *, epochs: int, eval_every: int = 1):
        if epochs < 1 or eval_every < 1:
            raise ValueError("epochs and eval_every must be positive")
        self.run = run
        self.epochs = epochs
        self.eval_every = eval_every

    def fit(
        self,
        train_epoch: Callable[[int], Metrics],
        evaluate: Callable[[int], Metrics] | None = None,
        checkpoint: Callable[[int, Metrics], None] | None = None,
    ) -> None:
        self.run.set_status("running", epoch=0, epochs=self.epochs)
        started = time.monotonic()
        try:
            for epoch in range(1, self.epochs + 1):
                train_metrics = train_epoch(epoch)
                self.run.log_metrics(train_metrics, split="train", epoch=epoch)
                latest = train_metrics
                if evaluate and (epoch % self.eval_every == 0 or epoch == self.epochs):
                    latest = evaluate(epoch)
                    self.run.log_metrics(latest, split="eval", epoch=epoch)
                if checkpoint:
                    checkpoint(epoch, latest)
                self.run.set_status(
                    "running",
                    epoch=epoch,
                    epochs=self.epochs,
                    progress=epoch / self.epochs,
                )
        except BaseException as exc:
            self.run.set_status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        self.run.set_status("completed", elapsed_seconds=time.monotonic() - started)

