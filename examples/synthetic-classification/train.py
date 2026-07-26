from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from researchlab import Run, Trainer


def boolean(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--preload-device", type=boolean, default=False)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def setup_device() -> tuple[torch.device, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        import torch_npu  # type: ignore

        torch_npu.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
        backend = "hccl"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed:
        dist.init_process_group(backend)
    return device, distributed


def reduce_totals(values: list[float], device: torch.device, distributed: bool) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def main() -> None:
    args = arguments()
    run = Run.from_env()
    device, distributed = setup_device()
    rank = int(os.environ.get("RANK", "0"))
    torch.manual_seed(args.seed)

    generator = torch.Generator().manual_seed(args.seed)
    features = torch.randn(args.samples, 12, generator=generator)
    weights = torch.randn(12, 4, generator=generator)
    labels = (features @ weights + 0.15 * torch.randn(args.samples, 4, generator=generator)).argmax(1)
    split = int(args.samples * 0.8)
    train_tensors = (features[:split], labels[:split])
    eval_tensors = (features[split:], labels[split:])
    if args.preload_device:
        train_tensors = tuple(tensor.to(device) for tensor in train_tensors)
        eval_tensors = tuple(tensor.to(device) for tensor in eval_tensors)

    train_data = TensorDataset(*train_tensors)
    eval_data = TensorDataset(*eval_tensors)
    train_sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    eval_sampler = DistributedSampler(eval_data, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
    )
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, sampler=eval_sampler, num_workers=0)

    model = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 4)).to(device)
    if distributed:
        model = (
            DistributedDataParallel(model)
            if device.type == "cpu"
            else DistributedDataParallel(model, device_ids=[device.index])
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(epoch: int) -> dict[str, float]:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = correct = count = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            if args.noise_std:
                batch_features = batch_features + torch.randn_like(batch_features) * args.noise_std
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_labels.numel()
            correct += (logits.argmax(1) == batch_labels).sum().item()
            count += batch_labels.numel()
        loss_sum, correct, count = reduce_totals([loss_sum, correct, count], device, distributed)
        return {"loss": loss_sum / count, "acc": correct / count}

    def evaluate(_: int) -> dict[str, float]:
        model.eval()
        loss_sum = correct = count = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in eval_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_features)
                loss_sum += criterion(logits, batch_labels).item() * batch_labels.numel()
                correct += (logits.argmax(1) == batch_labels).sum().item()
                count += batch_labels.numel()
        loss_sum, correct, count = reduce_totals([loss_sum, correct, count], device, distributed)
        return {"loss": loss_sum / count, "acc": correct / count}

    def checkpoint(epoch: int, metrics: dict[str, float]) -> None:
        if rank != 0:
            return
        model_state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        path = run.dir / "artifacts" / "latest.pt"
        torch.save({"epoch": epoch, "model": model_state, "metrics": metrics}, path)

    try:
        Trainer(run, epochs=args.epochs, eval_every=args.eval_every).fit(
            train_epoch,
            evaluate,
            checkpoint,
        )
        confusion = torch.zeros(4, 4, dtype=torch.int64, device=device)
        model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in eval_loader:
                predictions = model(batch_features.to(device)).argmax(1)
                for actual, predicted in zip(batch_labels.to(device), predictions):
                    confusion[actual.long(), predicted.long()] += 1
        if distributed:
            dist.all_reduce(confusion)
        if rank == 0:
            run.log_matrix(
                "测试集混淆矩阵",
                confusion.cpu().tolist(),
                labels=["class 0", "class 1", "class 2", "class 3"],
            )
            run.log_file(run.dir / "artifacts" / "latest.pt", name="latest.pt")
            run.log_table(
                "运行配置",
                ["参数", "值"],
                [[name, value] for name, value in vars(args).items()],
            )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
