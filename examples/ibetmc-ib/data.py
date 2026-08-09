from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class MultiViewMatDataset(Dataset):
    """Load the pre-split MATLAB format used by the original experiment."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        preload_device: torch.device | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        values = sio.loadmat(source)
        try:
            raw_views = values[f"X_{split}"][0]
            raw_labels = values[f"Y_{split}"].reshape(-1)
        except KeyError as exc:
            raise ValueError(
                f"{source} must contain X_train, X_test, Y_train, and Y_test"
            ) from exc

        self.views = [
            torch.as_tensor(np.asarray(view), dtype=torch.float32)
            for view in raw_views
        ]
        self.labels = torch.as_tensor(raw_labels, dtype=torch.long)
        if not self.views or any(len(view) != len(self.labels) for view in self.views):
            raise ValueError(f"inconsistent view or label lengths in {source}")
        if preload_device is not None:
            self.views = [view.to(preload_device) for view in self.views]
            self.labels = self.labels.to(preload_device)

        classes = sorted(set(self.labels.cpu().tolist()))
        if classes != list(range(len(classes))):
            raise ValueError("labels must be contiguous and zero-based")
        self.dims = [[int(view.shape[1])] for view in self.views]
        self.view_number = len(self.views)
        self.num_classes = len(classes)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [view[index] for view in self.views], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)
