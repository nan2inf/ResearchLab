from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from researchlab import Run


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["evaluate", "infer"], default="evaluate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def device() -> torch.device:
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        import torch_npu  # type: ignore

        torch_npu.npu.set_device(0)
        return torch.device("npu:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluation_data(samples: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(samples, 12, generator=generator)
    weights = torch.randn(12, 4, generator=generator)
    labels = (features @ weights + 0.15 * torch.randn(samples, 4, generator=generator)).argmax(1)
    split = int(samples * 0.8)
    return TensorDataset(features[split:], labels[split:])


def main() -> None:
    args = arguments()
    run = Run.from_env()
    run.set_status("running", mode=args.mode)
    target = device()
    model = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 4))
    checkpoint = torch.load(args.checkpoint.expanduser(), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(target).eval()
    loader = DataLoader(evaluation_data(args.samples, args.seed), batch_size=args.batch_size)

    loss_sum = correct = count = 0.0
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    predictions: list[dict[str, int]] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            logits = model(batch_features.to(target))
            predicted = logits.argmax(1).cpu()
            loss_sum += criterion(logits, batch_labels.to(target)).item() * batch_labels.numel()
            correct += (predicted == batch_labels).sum().item()
            count += batch_labels.numel()
            for actual, prediction in zip(batch_labels, predicted):
                confusion[actual.long(), prediction.long()] += 1
                predictions.append({"actual": int(actual), "predicted": int(prediction)})

    if args.mode == "evaluate":
        run.log_metrics({"loss": loss_sum / count, "acc": correct / count}, split="eval", step=1)
        run.log_matrix(
            "离线评测混淆矩阵",
            confusion.tolist(),
            labels=[f"class {index}" for index in range(4)],
        )
    else:
        output = run.dir / "predictions.json"
        output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
        run.log_file(output)
        run.log_table(
            "推理结果（前 50 条）",
            ["样本", "预测类别", "真实类别"],
            [[index, item["predicted"], item["actual"]] for index, item in enumerate(predictions[:50])],
        )
    run.set_status("completed", samples=int(count), mode=args.mode)


if __name__ == "__main__":
    main()
