from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated and committed OpenAPI.")
    parser.add_argument("generated", type=Path)
    parser.add_argument(
        "--committed", type=Path, default=Path("docs/openapi-v1.json")
    )
    args = parser.parse_args()
    generated = json.loads(args.generated.read_text(encoding="utf-8"))
    committed = json.loads(args.committed.read_text(encoding="utf-8"))
    if generated != committed:
        raise SystemExit("docs/openapi-v1.json is out of date")


if __name__ == "__main__":
    main()
