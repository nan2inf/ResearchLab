# ResearchLab

Local-first PyTorch research control panel for Windows/Linux, with direct SSH
execution on Linux GPU/NPU servers.

ResearchLab keeps the training loop in the researcher's hands. It standardizes
only four boundaries: project parameters, launch commands, run directories,
and result events.

## Current vertical slice

- Local Windows/Linux execution and Linux execution over the system OpenSSH
  client.
- Immutable timestamped source versions and isolated experiment directories.
- `python` and single-node `torchrun` launch modes.
- NVIDIA CUDA and Huawei Ascend device discovery, selection, and busy warnings.
- Conda/Python environment discovery without modifying existing environments.
- Live logs, custom metrics, images, matrices, tables, checkpoints, and files.
- A lightweight Python reporting API plus an optional callback-based trainer.
- Local SQLite state; SSH keys and passwords are never copied into the project.

## Quick start

```bash
python -m pip install -e .
researchlab serve
```

Open <http://127.0.0.1:8765>. Import `examples/synthetic-classification`,
create a source version, and start a local run.

The example also contains an `offline` task. After training, copy the displayed
`latest.pt` path into that task and choose `evaluate` or `infer`. The result page
shows evaluation metrics, a confusion matrix, a prediction table, artifacts,
the actual run directory, and the launch command.

`examples/ibetmc-ib` is a migrated real-world multi-view classification
experiment. It demonstrates how a legacy project can keep its custom model and
training loop while exposing paths and parameters to ResearchLab. Its dataset
remains outside the repository and is selected in the web UI.

Application state defaults to `~/.researchlab`. Override it for testing or
portable use:

```bash
set RESEARCHLAB_HOME=D:\researchlab-state
```

```bash
export RESEARCHLAB_HOME=/data/researchlab-state
```

## Project contract

Every project contains a `project.yaml`:

```yaml
schema_version: 1
name: synthetic-classification
tasks:
  train:
    entrypoint: train.py
    launcher: auto
    parameters:
      epochs:
        type: int
        default: 10
        min: 1
```

`launcher: auto` uses normal Python for zero or one selected accelerator and
`torchrun` for multiple selected devices. Existing projects can keep their own
loop and add only:

```python
from researchlab import Run

run = Run.from_env()
run.log_metrics({"loss": loss, "acc": acc}, split="train", epoch=epoch)
```

## Security boundary

The web server binds to loopback by default. Remote connections reuse the
user's OpenSSH configuration and SSH Agent. ResearchLab stores an SSH alias,
never private-key contents or passwords. Source versions are immutable and
remote commands run as the connected Linux user.

Model weights are not silently removed from a source version. Put large or
machine-specific files in `.gitignore` or the manifest's `exclude` list when
they should stay outside code synchronization.

## Why not wrap an existing platform?

MLflow and Aim are strong experiment trackers. ClearML and Determined add
agents, services, and cluster scheduling. Ray and Lightning provide more
opinionated training runtimes. ResearchLab targets a smaller gap: a
single-user local control panel that uploads ordinary PyTorch projects through
SSH and works without installing a permanent service on the training server.

## License

ResearchLab is released under the [Apache License 2.0](LICENSE).
