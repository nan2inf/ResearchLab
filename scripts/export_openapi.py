from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from researchlab.api_v1 import v1_openapi
from researchlab.runs import LabService
from researchlab.store import Database
from researchlab.web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the stable ResearchLab API v1 schema.")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("docs/openapi-v1.json"),
    )
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="researchlab-openapi-") as temp_dir:
        service = LabService(Database(Path(temp_dir) / "openapi.db"))
        schema = v1_openapi(create_app(service))
        output.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
