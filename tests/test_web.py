from pathlib import Path

from fastapi.testclient import TestClient

from researchlab.runs import LabService
from researchlab.store import Database
from researchlab.web import create_app


def test_local_web_app_serves_ui_and_api(tmp_path: Path) -> None:
    service = LabService(Database(tmp_path / "state" / "app.db"))
    client = TestClient(create_app(service))

    assert client.get("/api/health").json()["ok"] is True
    assert client.get("/api/summary").json()["projects"] == []
    page = client.get("/")
    assert page.status_code == 200
    assert "ResearchLab" in page.text

