# ResearchLab project contract

## Contents

1. Required files
2. Manifest schema
3. Reporting API
4. Compatibility checklist

## Required files

A migrated project needs one `project.yaml` and at least one Python entrypoint.
ResearchLab snapshots this directory and injects its small reporting runtime.

## Manifest schema

```yaml
schema_version: 1
name: my-research-project
description: Short human-readable summary
environment:
  python: ">=3.10"
  torch: ">=2.1"
  torch_npu: ">=2.1"
exclude:
  - old-results
  - "*.mat"
tasks:
  train:
    entrypoint: train.py
    launcher: auto
    parameters:
      epochs:
        type: int
        default: 100
        min: 1
        help: Number of training epochs
      data_path:
        type: path
        required: true
      optimizer:
        type: choice
        default: adam
        choices: [adam, sgd]
      augment:
        type: bool
        default: true
      hidden:
        type: int_list
        default: [256, 128]
```

Task `launcher` values:

- `python`: always launch one process.
- `torchrun`: require selected devices and launch one process per device.
- `auto`: use `torchrun` when multiple devices are selected, otherwise Python.

Parameter `type` values are `int`, `float`, `str`, `bool`, `path`, `choice`,
`int_list`, `float_list`, and `str_list`.

Use `flag` to map a manifest name to an existing CLI flag. Use `style: flag`
only for `store_true`-style boolean arguments; the default style passes a value.

## Reporting API

```python
from researchlab import Run

run = Run.from_env()
run.log_metrics(
    {"loss": float(loss), "acc": float(accuracy)},
    split="train",
    epoch=epoch,
    step=global_step,
)
run.log_matrix("confusion", matrix.tolist(), labels=class_names)
run.log_table("per-class", ["class", "precision"], rows)
run.log_image("attention.png")
run.log_file("model_best.pt")
```

The API is rank-zero-only by default. Keep metric reductions outside it so all
distributed workers participate.

## Compatibility checklist

- The project starts from its snapshot directory, not the original directory.
- All local and remote data paths are parameters.
- CPU execution does not call `.cuda()` unconditionally.
- Selected CUDA/NPU devices come from the launch environment.
- `torchrun` code reads `LOCAL_RANK`, initializes the correct backend, uses a
  distributed sampler, and destroys the process group.
- Evaluation frequency and total epochs are user parameters.
- Runs write only inside `RESEARCHLAB_RUN_DIR`.
- Metrics are numeric and use stable names across epochs.
- Checkpoints contain enough state to resume when the original project supports it.
