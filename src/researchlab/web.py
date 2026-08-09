from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api_v1 import create_router as create_v1_router
from .api_v1 import register_error_handlers
from .runs import LabService
from .system import discover_environments, executor_for, hardware_snapshot, require_local_tools


STATIC_DIR = Path(__file__).parent / "static"


class ServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    ssh_alias: str = Field(min_length=1, max_length=120)
    remote_root: str = Field(min_length=1, max_length=500)
    shell_init: str = Field(default="", max_length=1000)


class ProjectCreate(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)


class VersionCreate(BaseModel):
    name: str = Field(default="", max_length=80)
    server_id: str | None = None


class RunCreate(BaseModel):
    version_id: str
    task: str
    params: dict[str, Any] = Field(default_factory=dict)
    devices: list[int] = Field(default_factory=list)
    backend: str = "cpu"
    python: str | None = None
    name: str = Field(default="", max_length=80)


class CondaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    python_version: str = "3.11"
    install_command: str = Field(default="", max_length=4000)


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc).strip("'"))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def create_app(service: LabService | None = None) -> FastAPI:
    app = FastAPI(
        title="ResearchLab",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )
    app.state.lab = service or LabService()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def lab() -> LabService:
        return app.state.lab

    register_error_handlers(app)
    app.include_router(create_v1_router(lab))

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "tools": require_local_tools()}

    @app.get("/api/summary")
    def summary() -> dict:
        service = lab()
        return {
            "servers": service.db.list("servers"),
            "projects": service.db.list("projects"),
            "versions": service.db.list("versions"),
            "runs": service.db.list("runs"),
        }

    @app.post("/api/servers", status_code=201)
    def create_server(request: ServerCreate) -> dict:
        try:
            return lab().add_server(**request.model_dump())
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/hardware")
    def hardware(server_id: str | None = None) -> dict:
        try:
            server = lab().db.get("servers", server_id) if server_id else None
            return hardware_snapshot(executor_for(server))
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/environments")
    def environments(server_id: str | None = None) -> list[dict]:
        try:
            server = lab().db.get("servers", server_id) if server_id else None
            return discover_environments(executor_for(server))
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/api/servers/{server_id}/environments", status_code=201)
    def create_environment(server_id: str, request: CondaCreate) -> dict:
        try:
            return lab().create_conda_environment(server_id, **request.model_dump())
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/api/projects", status_code=201)
    def create_project(request: ProjectCreate) -> dict:
        try:
            return lab().add_project(request.source_path)
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/api/projects/{project_id}/versions", status_code=201)
    def create_version(project_id: str, request: VersionCreate) -> dict:
        try:
            return lab().create_version(project_id, **request.model_dump())
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/versions/{version_id}/environments")
    def version_environments(version_id: str, backend: str = "cpu") -> list[dict]:
        try:
            return lab().version_environments(version_id, backend)
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/api/runs", status_code=201)
    def create_run(request: RunCreate) -> dict:
        try:
            data = request.model_dump()
            task = data.pop("task")
            return lab().start_run(data.pop("version_id"), task_name=task, **data)
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        try:
            return lab().cancel_run(run_id)
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/runs/{run_id}/events")
    def events(
        run_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=5000),
    ) -> dict:
        try:
            return lab().read_events(run_id, after, limit)
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(run_id: str, lines: int = Query(default=200, ge=1, le=2000)) -> str:
        try:
            return lab().read_log(run_id, lines)
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            return FileResponse(lab().artifact_path(run_id, artifact_path))
        except Exception as exc:
            raise _error(exc) from exc

    return app
