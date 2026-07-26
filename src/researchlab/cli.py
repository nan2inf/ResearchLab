from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from .system import discover_environments, hardware_snapshot, require_local_tools


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="researchlab")
    commands = root.add_subparsers(dest="command")
    serve = commands.add_parser("serve", help="start the local web control panel")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    commands.add_parser("doctor", help="inspect local tools, hardware, and Python environments")
    return root


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ["serve"]
    args = parser().parse_args(argv)
    command = args.command
    if command == "doctor":
        print(
            json.dumps(
                {
                    "tools": require_local_tools(),
                    "hardware": hardware_snapshot(),
                    "environments": discover_environments(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("ResearchLab is local-only; bind to a loopback address")
    uvicorn.run("researchlab.web:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
