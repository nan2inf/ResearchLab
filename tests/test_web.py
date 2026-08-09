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


def test_api_v1_contract_and_errors(tmp_path: Path) -> None:
    service = LabService(Database(tmp_path / "state" / "app.db"))
    client = TestClient(create_app(service))

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["data"]["api_version"] == "v1"

    projects = client.get("/api/v1/projects?limit=10&offset=0")
    assert projects.json() == {
        "data": [],
        "meta": {"total": 0, "limit": 10, "offset": 0},
    }

    missing = client.get("/api/v1/projects/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"

    invalid = client.post("/api/v1/projects", json={"source_path": ""})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "request_validation_failed"

    schema = client.get("/api/v1/openapi.json").json()
    assert schema["info"]["version"] == "1.0.0"
    assert schema["paths"]
    assert all(path.startswith("/api/v1") for path in schema["paths"])
    assert "researchlab__web__RunCreate" not in schema["components"]["schemas"]
