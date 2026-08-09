from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MultiViewMatDataset
from model import IBETMC
from researchlab import Run


def boolean(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", "--batch_sz", type=int, default=64)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--annealing-epoch", type=int, default=10)
    parser.add_argument("--attention-embed-dim", type=int, default=512)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ib-bottleneck-dim", type=int, default=128)
    parser.add_argument("--preload-device", type=boolean, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def selected_device() -> torch.device:
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        import torch_npu  # type: ignore

        torch_npu.npu.set_device(0)
        return torch.device("npu:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    npu = getattr(torch, "npu", None)
    if npu is not None:
        npu.manual_seed_all(seed)


def batches(loader: DataLoader, limit: int):
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        yield batch


def main() -> None:
    args = arguments()
    if args.eval_every < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("eval_every and gradient_accumulation_steps must be positive")
    if args.max_batches < 0:
        raise ValueError("max_batches cannot be negative")
    if args.attention_embed_dim % args.attention_heads:
        raise ValueError("attention_embed_dim must be divisible by attention_heads")
    if args.preload_device and args.num_workers:
        raise ValueError("num_workers must be 0 when preload_device is enabled")

    run = Run.from_env()
    run.set_status("running", epoch=0, epochs=args.epochs)
    device = selected_device()
    seed_everything(args.seed)
    preload = device if args.preload_device else None
    train_data = MultiViewMatDataset(args.data_path, "train", preload)
    test_data = MultiViewMatDataset(args.data_path, "test", preload)
    if train_data.dims != test_data.dims or train_data.num_classes != test_data.num_classes:
        raise ValueError("train and test metadata do not match")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = IBETMC(
        train_data.dims,
        train_data.num_classes,
        args.annealing_epoch,
        args.ib_bottleneck_dim,
        args.attention_embed_dim,
        args.attention_heads,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=args.lr_factor,
    )
    artifacts = run.dir / "artifacts"
    checkpoint_path = artifacts / "checkpoint.pt"
    best_path = artifacts / "model_best.pt"
    start_epoch = 1
    global_step = 0
    best_accuracy = -float("inf")
    no_improve = 0
    if args.resume:
        checkpoint = torch.load(args.resume.expanduser(), map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_accuracy = float(checkpoint.get("best_metric", best_accuracy))
        no_improve = int(checkpoint.get("n_no_improve", 0))

    def move(batch):
        views, target = batch
        return [view.to(device) for view in views], target.long().to(device)

    def train_epoch(epoch: int) -> tuple[float, float]:
        nonlocal global_step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        correct = count = 0
        for views, target in batches(train_loader, args.max_batches):
            views, target = move((views, target))
            _, combined, loss = model(views, target, epoch)
            predicted = combined.argmax(dim=1)
            correct += (predicted == target).sum().item()
            count += target.numel()
            losses.append(float(loss.item()))
            (loss / args.gradient_accumulation_steps).backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if global_step % args.gradient_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return sum(losses) / len(losses), correct / count

    def evaluate(epoch: int) -> tuple[float, float, list[int], list[int]]:
        model.eval()
        losses: list[float] = []
        actual: list[int] = []
        predicted: list[int] = []
        with torch.no_grad():
            for views, target in batches(test_loader, args.max_batches):
                views, target = move((views, target))
                _, combined, loss = model(views, target, epoch)
                losses.append(float(loss.item()))
                actual.extend(target.cpu().tolist())
                predicted.extend(combined.argmax(dim=1).cpu().tolist())
        correct = sum(left == right for left, right in zip(actual, predicted))
        return sum(losses) / len(losses), correct / len(actual), actual, predicted

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        train_loss, train_accuracy = train_epoch(epoch)
        scheduler.step(train_accuracy)
        run.log_metrics(
            {"loss": train_loss, "acc": train_accuracy},
            split="train",
            epoch=epoch,
            step=global_step,
        )
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        eval_loss, eval_accuracy, _, _ = evaluate(epoch)
        run.log_metrics(
            {"loss": eval_loss, "acc": eval_accuracy},
            split="eval",
            epoch=epoch,
            step=global_step,
        )
        improved = eval_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, eval_accuracy)
        no_improve = 0 if improved else no_improve + 1
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "n_no_improve": no_improve,
            "best_metric": best_accuracy,
            "global_step": global_step,
        }
        torch.save(state, checkpoint_path)
        if improved:
            torch.save(state, best_path)
        run.set_status("running", epoch=epoch, epochs=args.epochs, progress=epoch / args.epochs)
        if no_improve >= args.patience:
            break

    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test_loss, test_accuracy, actual, predicted = evaluate(last_epoch)
    confusion = torch.zeros(train_data.num_classes, train_data.num_classes, dtype=torch.int64)
    for label, prediction in zip(actual, predicted):
        confusion[label, prediction] += 1
    predictions_path = artifacts / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "predicted", "actual"])
        writer.writerows(
            (index, prediction, label)
            for index, (prediction, label) in enumerate(zip(predicted, actual))
        )
    config_path = artifacts / "config.json"
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "data_path": str(args.data_path),
                "resume": str(args.resume or ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run.log_metrics({"loss": test_loss, "acc": test_accuracy}, split="test", epoch=last_epoch)
    run.log_matrix(
        "测试集混淆矩阵",
        confusion.tolist(),
        labels=[str(index) for index in range(train_data.num_classes)],
    )
    run.log_table(
        "预测结果（前 100 条）",
        ["样本", "预测类别", "真实类别"],
        [
            [index, prediction, label]
            for index, (prediction, label) in enumerate(
                zip(predicted[:100], actual[:100])
            )
        ],
    )
    for path in (checkpoint_path, best_path, predictions_path, config_path):
        if path.is_file():
            run.log_file(path)
    run.set_status("completed", epoch=last_epoch, best_accuracy=best_accuracy)


if __name__ == "__main__":
    main()
