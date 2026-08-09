# IBETMC-IB migration

This is a clean ResearchLab-compatible copy of the representative experiment
from `Classification/exp3/IBETMC-IB`. The original directory is unchanged.

Included:

- the active pre-split MATLAB loader;
- the IBETMC model path used by `train.py`;
- training, validation, checkpoint resume, final evaluation, confusion matrix,
  and prediction export;
- CPU, one selected CUDA device, or one selected Ascend NPU device.

Not copied:

- datasets, previous logs, checkpoints, CSV results, caches, papers, archives;
- `train2.py` and utilities that were byte-identical across experiment copies;
- the large commented-out legacy loader and unused mutual-information helpers.

The dataset path must point to a `.mat` file containing `X_train`, `X_test`,
`Y_train`, and `Y_test`. The former hard-coded dataset path and source-relative
output directories were removed. ResearchLab now owns the run directory.

This migration intentionally keeps single-process execution. Multi-device DDP
would change the original training behavior and should be validated separately.

## Migration verification

A read-only CPU smoke run used the original pre-split ProteinFold data with one
training and one evaluation batch. ResearchLab successfully imported the
project, created an immutable version, and recorded:

- 12 input views with 27 features per view and 27 classes, matching the legacy
  dataset preset;
- `train/loss`, `train/acc`, `eval/loss`, `eval/acc`, `test/loss`, and
  `test/acc`;
- status, matrix, table, and artifact events;
- checkpoint state for the model, optimizer, scheduler, epoch, global step,
  best metric, and early-stopping counter.

The smoke-only `max_batches` parameter defaults to `0`, so ordinary runs still
process the full dataset.
