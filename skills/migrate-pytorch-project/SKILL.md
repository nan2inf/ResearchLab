---
name: migrate-pytorch-project
description: Safely migrate an existing PyTorch research repository or a tangled experiment directory into a new ResearchLab-compatible project. Use when Codex needs to inspect legacy train/eval scripts, recover argparse parameters, separate code from datasets and outputs, replace hard-coded paths or device selection, add ResearchLab metric reporting, create project.yaml, deduplicate copied experiment variants, or verify that the migrated project preserves the original behavior.
---

# Migrate PyTorch Project

Create a clean compatible copy. Never edit, move, or delete the source project.

## Workflow

1. Resolve the source directory and a distinct empty destination directory.
2. Run the deterministic inventory:

   ```bash
   python scripts/scan_project.py <source> --output <temporary-report.json>
   ```

3. Read the highest-scoring entrypoints and their directly imported local modules. Treat scanner findings as leads, not proof.
4. Identify one runnable experiment boundary: entrypoint, data path, parameters, environment, devices, metrics, checkpoints, and final artifacts.
5. Copy only files required by that boundary. Exclude datasets, logs, checkpoints, caches, paper drafts, archives, and copied variants unless the user explicitly includes them.
6. Create `project.yaml` using [the ResearchLab project contract](references/project-schema.md).
7. Make the minimum compatibility changes:
   - expose hard-coded choices as parameters;
   - accept data and output paths from configuration;
   - replace unconditional `.cuda()` calls with the selected device;
   - keep custom loaders, in-memory datasets, online augmentation, losses, optimizers, and loops intact;
   - add `Run.from_env()` and report existing metrics at their current calculation points.
8. Do not force the optional `Trainer` onto an existing custom loop. Use it only when the original loop is already a conventional epoch/evaluation loop and the change is smaller.
9. Run a tiny smoke configuration that cannot overwrite original results. Compare the original and migrated parameter resolution, output shapes, metric names, and checkpoint keys.
10. Return the destination, runnable command, changes made, excluded material, verification result, and any behavior that still needs manual review.

## Migration rules

- Preserve algorithm behavior before cleaning style.
- Prefer one representative of byte-identical copies. Keep genuinely different algorithms as separate projects or tasks.
- Do not turn dataset-specific constants into a large abstraction. Put stable presets in ordinary Python or YAML data.
- Do not copy SSH keys, credentials, absolute private paths, datasets, or prior run outputs.
- Do not claim CUDA or Ascend compatibility until the code path avoids unconditional backend-specific calls.
- For DDP, log and write shared artifacts only on rank zero. All ranks must still participate in collective operations.
- Keep the source directory read-only throughout the migration.

