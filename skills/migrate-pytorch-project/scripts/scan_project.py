#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dataset",
    "datasets",
    "logs",
    "runs",
    "artifacts",
    "checkpoints",
}
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
POSIX_DATA_PATH = re.compile(r"^/(?:data|home|mnt|workspace|root)/")


def literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{literal(node.value)}.{node.attr}"
        return None


def source_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        and path.stat().st_size <= 2_000_000
    ]


def inspect_file(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    report: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "entrypoint": False,
        "score": 0,
        "parameters": [],
        "hardcoded_paths": [],
        "device_calls": [],
        "outputs": [],
        "imports": [],
        "parse_error": None,
    }
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        report["parse_error"] = f"{exc.msg} at line {exc.lineno}"
        return report

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            rendered = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if "__name__" in rendered and "__main__" in rendered:
                report["entrypoint"] = True
                report["score"] += 5
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            report["imports"].extend(names)
            if "torch" in names:
                report["score"] += 2
        elif isinstance(node, ast.For):
            rendered = ast.unparse(node.iter) if hasattr(ast, "unparse") else ""
            if "epoch" in rendered.lower() or "range" in rendered:
                report["score"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if WINDOWS_PATH.match(node.value) or POSIX_DATA_PATH.match(node.value):
                report["hardcoded_paths"].append({"line": node.lineno, "value": node.value})
        elif isinstance(node, ast.Call):
            name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if name.endswith(".parse_args"):
                report["entrypoint"] = True
                report["score"] += 3
            if name.endswith(".add_argument") and node.args:
                flags = [literal(argument) for argument in node.args]
                keywords = {keyword.arg: literal(keyword.value) for keyword in node.keywords}
                parameter = {
                    "flags": [flag for flag in flags if isinstance(flag, str)],
                    "type": keywords.get("type"),
                    "default": keywords.get("default"),
                    "required": bool(keywords.get("required", False)),
                    "choices": keywords.get("choices"),
                    "action": keywords.get("action"),
                    "help": keywords.get("help"),
                    "line": node.lineno,
                }
                report["parameters"].append(parameter)
                report["score"] += 1
            if name.endswith(".cuda") or name in {"torch.cuda.set_device", "torch.cuda.device"}:
                report["device_calls"].append({"line": node.lineno, "call": name})
            if name in {"open", "torch.save", "json.dump", "csv.writer", "numpy.save", "np.save"}:
                report["outputs"].append({"line": node.lineno, "call": name})
    report["hardcoded_paths"] = list(
        {json.dumps(item, ensure_ascii=False): item for item in report["hardcoded_paths"]}.values()
    )
    report["imports"] = sorted(set(report["imports"]))
    return report


def scan(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = source_files(root)
    reports = [inspect_file(root, path) for path in files]
    digests: dict[str, list[str]] = defaultdict(list)
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[digest].append(path.relative_to(root).as_posix())
    duplicates = [
        {"sha256": digest, "files": paths}
        for digest, paths in digests.items()
        if len(paths) > 1
    ]
    entrypoints = sorted(
        (report for report in reports if report["entrypoint"]),
        key=lambda report: (-report["score"], report["path"]),
    )
    return {
        "source": str(root),
        "python_files": len(files),
        "entrypoints": entrypoints,
        "duplicate_groups": sorted(duplicates, key=lambda group: (-len(group["files"]), group["files"])),
        "parse_errors": [
            {"path": report["path"], "error": report["parse_error"]}
            for report in reports
            if report["parse_error"]
        ],
        "summary": {
            "entrypoint_count": len(entrypoints),
            "duplicate_group_count": len(duplicates),
            "hardcoded_path_count": sum(len(report["hardcoded_paths"]) for report in reports),
            "backend_specific_call_count": sum(len(report["device_calls"]) for report in reports),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only PyTorch migration inventory")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan(args.source)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

