# Open-source landscape review

No reviewed project directly covers the intended combination: local
single-user UI, agentless SSH upload/execution, arbitrary PyTorch loops,
per-device selection, and first-class NVIDIA plus Huawei Ascend support.

| Project | Strong fit | Why it is not the base |
| --- | --- | --- |
| [MLflow](https://mlflow.org/docs/latest/tracking/) | Local/remote experiment tracking, metrics, artifacts, comparison UI | Does not provide the intended SSH code upload, environment discovery, or accelerator selection workflow |
| [Aim](https://github.com/aimhubio/aim) | Lightweight self-hosted experiment UI and Python logging API | Primarily tracking; not a remote execution controller |
| [ClearML](https://github.com/clearml/clearml) | Tracking, remote execution, scheduling, agents, web UI | Requires a broader server/agent stack than the local single-user security boundary |
| [Determined](https://docs.determined.ai/) | GPU scheduling, distributed training, WebUI | Cluster master/agent architecture and a more opinionated training platform |
| [Ray Train](https://docs.ray.io/en/latest/train/overview.html) | Distributed PyTorch execution and reporting | Requires Ray runtime and cluster concepts; does not match direct SSH execution |
| [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) | Mature loops, callbacks, DDP, checkpointing | Useful as an optional project integration, but too restrictive as the universal project contract |
| [DVC Experiments](https://dvc.org/doc/start/experiments) | Git-related experiment and data versioning | Data/version workflow is broader than the requested server-path model and does not launch SSH jobs |
| [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch) | Official `torch_npu` adapter and migration guidance | Hardware runtime dependency to support, not an experiment controller |

## Decision

Build a thin orchestration layer rather than fork a platform:

- keep the portable run/metric/artifact concepts proven by MLflow and Aim;
- launch native Python or `torchrun`, as recommended by PyTorch and Ascend;
- use the system OpenSSH client instead of a persistent remote agent;
- leave Lightning, Ray, MLflow, and other libraries usable inside individual
  projects rather than making one of them mandatory.

