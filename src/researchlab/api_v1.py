from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import sqlite3
from typing import Any, Callable, Generic, Literal, TypeVar

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .runs import LabService
from .system import discover_environments, executor_for, hardware_snapshot, require_local_tools


T = TypeVar("T")


class ItemResponse(BaseModel, Generic[T]):
    data: T


class PageMeta(BaseModel):
    total: int
    limit: int
    offset: int


class ListResponse(BaseModel, Generic[T]):
    data: list[T]
    meta: PageMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Counts(BaseModel):
    servers: int
    projects: int
    versions: int
    runs: int
    running: int


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    api_version: Literal["v1"] = "v1"
    tools: dict[str, bool]


class Capabilities(BaseModel):
    api_version: Literal["v1"] = "v1"
    deployment: Literal["local-single-user"] = "local-single-user"
    local_platforms: list[str] = ["windows", "linux"]
    remote_platforms: list[str] = ["linux"]
    backends: list[str] = ["cpu", "cuda", "npu"]
    launchers: list[str] = ["python", "torchrun"]
    features: dict[str, bool]


class ServerResource(BaseModel):
    id: str
    name: str
    ssh_alias: str
    remote_root: str
    shell_init: str = ""
    created_at: datetime


class ProjectResource(BaseModel):
    id: str
    name: str
    source_path: str
    manifest: dict[str, Any]
    created_at: datetime


class VersionResource(BaseModel):
    id: str
    project_id: str
    name: str
    source_path: str
    server_id: str | None = None
    remote_path: str | None = None
    manifest: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


RunStatus = Literal["queued", "running", "completed", "failed", "cancelled", "unknown"]


class RunResource(BaseModel):
    id: str
    project_id: str
    version_id: str
    server_id: str | None = None
    name: str
    task: str
    status: RunStatus
    params: dict[str, Any]
    devices: list[int]
    backend: Literal["cpu", "cuda", "npu"]
    run_path: str
    remote_path: str | None = None
    pid: int | None = None
    command: list[str]
    exit_code: int | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Summary(BaseModel):
    counts: Counts
    recent_runs: list[RunResource]


class ServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    ssh_alias: str = Field(min_length=1, max_length=120)
    remote_root: str = Field(min_length=1, max_length=500)
    shell_init: str = Field(default="", max_length=1000)


class ServerUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    ssh_alias: str | None = Field(default=None, min_length=1, max_length=120)
    remote_root: str | None = Field(default=None, min_length=1, max_length=500)
    shell_init: str | None = Field(default=None, max_length=1000)


class EnvironmentProbe(BaseModel):
    python: str = Field(min_length=1, max_length=1000)


class CondaEnvironmentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    python_version: str = Field(default="3.11", pattern=r"^3\.\d{1,2}$")
    pip_packages: list[str] = Field(default_factory=list, max_length=100)
    install_command: str = Field(default="", max_length=4000)


class ProjectCreate(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)


class ProjectInspect(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)


class SourceFile(BaseModel):
    path: str
    size: int


class ProjectInspection(BaseModel):
    root: str
    manifest: dict[str, Any]
    files: list[SourceFile]
    file_count: int
    total_bytes: int
    truncated: bool


class SourceContent(BaseModel):
    path: str
    size: int
    content: str


OperationStatus = Literal["queued", "running", "completed", "failed"]


class OperationResource(BaseModel):
    id: str
    kind: str
    target_id: str | None = None
    status: OperationStatus
    progress: float
    message: str
    result: Any = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class EnvironmentOperationCreate(CondaEnvironmentCreate):
    server_id: str | None = None


class VersionOperationCreate(BaseModel):
    project_id: str
    name: str = Field(default="", max_length=80)
    server_id: str | None = None


class VersionCreate(BaseModel):
    name: str = Field(default="", max_length=80)
    server_id: str | None = None


class RunCreate(BaseModel):
    version_id: str
    task: str
    params: dict[str, Any] = Field(default_factory=dict)
    devices: list[int] = Field(default_factory=list)
    backend: Literal["cpu", "cuda", "npu"] = "cpu"
    python: str | None = None
    name: str = Field(default="", max_length=80)


class RunValidate(BaseModel):
    version_id: str
    task: str
    params: dict[str, Any] = Field(default_factory=dict)
    devices: list[int] = Field(default_factory=list)
    backend: Literal["cpu", "cuda", "npu"] = "cpu"
    python: str | None = None


class RunValidation(BaseModel):
    valid: bool
    command: list[str]
    params: dict[str, Any]
    environment: dict[str, Any]
    hardware: dict[str, Any]
    warnings: list[str]


class RunRerun(BaseModel):
    name: str = Field(default="", max_length=80)


class ArtifactResource(BaseModel):
    path: str
    size: int


class EventPage(BaseModel):
    events: list[dict[str, Any]]
    next: int


class V1Error(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Any = None,
        retryable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.detail = ErrorDetail(
            code=code,
            message=message,
            details=details,
            retryable=retryable,
        )


def _as_v1_error(exc: Exception) -> V1Error:
    message = str(exc).strip("'") or type(exc).__name__
    if isinstance(exc, KeyError):
        return V1Error(404, "not_found", message)
    if isinstance(exc, FileNotFoundError):
        return V1Error(404, "file_not_found", message)
    if isinstance(exc, TimeoutError):
        return V1Error(504, "timeout", message, retryable=True)
    if isinstance(exc, sqlite3.IntegrityError):
        return V1Error(409, "conflict", message)
    if isinstance(exc, ValueError):
        return V1Error(400, "invalid_request", message)
    if isinstance(exc, RuntimeError):
        return V1Error(502, "operation_failed", message, retryable=True)
    return V1Error(500, "internal_error", "ResearchLab encountered an internal error")


def _call(action: Callable[[], T]) -> T:
    try:
        return action()
    except V1Error:
        raise
    except Exception as exc:
        raise _as_v1_error(exc) from exc


def _page(items: list[T], limit: int, offset: int) -> ListResponse[T]:
    return ListResponse(
        data=items[offset : offset + limit],
        meta=PageMeta(total=len(items), limit=limit, offset=offset),
    )


def v1_openapi(app: FastAPI) -> dict[str, Any]:
    schema = deepcopy(app.openapi())
    schema["info"] = {
        "title": "ResearchLab API",
        "version": "1.0.0",
        "description": "Local single-user research control API.",
    }
    schema["paths"] = {
        path: value for path, value in schema["paths"].items() if path.startswith("/api/v1")
    }
    schemas = schema.get("components", {}).get("schemas", {})
    referenced: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
                referenced.add(reference.rsplit("/", 1)[-1])
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(schema["paths"])
    visited: set[str] = set()
    while referenced - visited:
        name = (referenced - visited).pop()
        visited.add(name)
        if name in schemas:
            collect(schemas[name])
    schema.setdefault("components", {})["schemas"] = {
        name: value for name, value in schemas.items() if name in referenced
    }
    return schema


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(V1Error)
    async def v1_error_handler(_: Request, exc: V1Error) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.detail).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/api/v1"):
            return await request_validation_exception_handler(request, exc)
        errors = [
            {key: value for key, value in error.items() if key in {"type", "loc", "msg"}}
            for error in exc.errors()
        ]
        detail = ErrorDetail(
            code="request_validation_failed",
            message="The request does not match the API contract.",
            details=errors,
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error=detail).model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        if not request.url.path.startswith("/api/v1"):
            return await http_exception_handler(request, exc)
        detail = ErrorDetail(code="http_error", message=str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=detail).model_dump(mode="json"),
        )


def create_router(get_lab: Callable[[], LabService]) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["API v1"])

    @router.get("/openapi.json", include_in_schema=False)
    def openapi(request: Request) -> JSONResponse:
        return JSONResponse(v1_openapi(request.app))

    @router.get("/health", response_model=ItemResponse[Health])
    def health() -> ItemResponse[Health]:
        return ItemResponse(data=Health(tools=require_local_tools()))

    @router.get("/capabilities", response_model=ItemResponse[Capabilities])
    def capabilities() -> ItemResponse[Capabilities]:
        return ItemResponse(
            data=Capabilities(
                features={
                    "ssh": True,
                    "environment_discovery": True,
                    "environment_creation": True,
                    "source_versions": True,
                    "live_events": True,
                    "artifacts": True,
                    "run_comparison": False,
                    "background_operations": True,
                }
            )
        )

    @router.get("/summary", response_model=ItemResponse[Summary])
    def summary() -> ItemResponse[Summary]:
        service = get_lab()
        servers = service.db.list("servers")
        projects = service.db.list("projects")
        versions = service.db.list("versions")
        runs = service.db.list("runs")
        return ItemResponse(
            data=Summary(
                counts=Counts(
                    servers=len(servers),
                    projects=len(projects),
                    versions=len(versions),
                    runs=len(runs),
                    running=sum(run["status"] == "running" for run in runs),
                ),
                recent_runs=[RunResource.model_validate(run) for run in runs[:10]],
            )
        )

    @router.get("/operations", response_model=ListResponse[OperationResource])
    def operations(
        status: OperationStatus | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ListResponse[OperationResource]:
        items = get_lab().db.list(
            "operations", **({"status": status} if status else {})
        )
        records = [OperationResource.model_validate(item) for item in items]
        return _page(records, limit, offset)

    @router.get(
        "/operations/{operation_id}", response_model=ItemResponse[OperationResource]
    )
    def operation(operation_id: str) -> ItemResponse[OperationResource]:
        record = _call(lambda: get_lab().db.get("operations", operation_id))
        return ItemResponse(data=OperationResource.model_validate(record))

    @router.post(
        "/operations/environments",
        response_model=ItemResponse[OperationResource],
        status_code=202,
    )
    def create_environment_operation(
        request: EnvironmentOperationCreate,
    ) -> ItemResponse[OperationResource]:
        data = request.model_dump()
        server_id = data.pop("server_id")
        record = _call(
            lambda: get_lab().create_environment_operation(server_id, **data)
        )
        return ItemResponse(data=OperationResource.model_validate(record))

    @router.post(
        "/operations/versions",
        response_model=ItemResponse[OperationResource],
        status_code=202,
    )
    def create_version_operation(
        request: VersionOperationCreate,
    ) -> ItemResponse[OperationResource]:
        data = request.model_dump()
        project_id = data.pop("project_id")
        record = _call(lambda: get_lab().create_version_operation(project_id, **data))
        return ItemResponse(data=OperationResource.model_validate(record))

    @router.get("/servers", response_model=ListResponse[ServerResource])
    def servers(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ListResponse[ServerResource]:
        records = [ServerResource.model_validate(item) for item in get_lab().db.list("servers")]
        return _page(records, limit, offset)

    @router.post("/servers", response_model=ItemResponse[ServerResource], status_code=201)
    def create_server(request: ServerCreate) -> ItemResponse[ServerResource]:
        record = _call(lambda: get_lab().add_server(**request.model_dump()))
        return ItemResponse(data=ServerResource.model_validate(record))

    @router.get("/servers/{server_id}", response_model=ItemResponse[ServerResource])
    def server(server_id: str) -> ItemResponse[ServerResource]:
        record = _call(lambda: get_lab().db.get("servers", server_id))
        return ItemResponse(data=ServerResource.model_validate(record))

    @router.patch("/servers/{server_id}", response_model=ItemResponse[ServerResource])
    def update_server(
        server_id: str, request: ServerUpdate
    ) -> ItemResponse[ServerResource]:
        changes = request.model_dump(exclude_none=True)
        if not changes:
            raise V1Error(400, "invalid_request", "At least one field must be supplied.")
        record = _call(lambda: get_lab().update_server(server_id, **changes))
        return ItemResponse(data=ServerResource.model_validate(record))

    @router.get("/servers/{server_id}/diagnostics", response_model=ItemResponse[dict[str, Any]])
    def diagnostics(server_id: str) -> ItemResponse[dict[str, Any]]:
        return ItemResponse(data=_call(lambda: get_lab().server_diagnostics(server_id)))

    @router.get("/servers/{server_id}/environments", response_model=ListResponse[dict[str, Any]])
    def server_environments(server_id: str) -> ListResponse[dict[str, Any]]:
        records = _call(
            lambda: discover_environments(
                executor_for(get_lab().db.get("servers", server_id))
            )
        )
        return _page(records, 200, 0)

    @router.post(
        "/servers/{server_id}/environments/probe",
        response_model=ItemResponse[dict[str, Any]],
    )
    def probe_server_environment(
        server_id: str, request: EnvironmentProbe
    ) -> ItemResponse[dict[str, Any]]:
        return ItemResponse(
            data=_call(lambda: get_lab().probe_environment(server_id, request.python))
        )

    @router.post(
        "/servers/{server_id}/environments",
        response_model=ItemResponse[dict[str, Any]],
        status_code=201,
    )
    def create_server_environment(
        server_id: str, request: CondaEnvironmentCreate
    ) -> ItemResponse[dict[str, Any]]:
        return ItemResponse(
            data=_call(
                lambda: get_lab().create_conda_environment(
                    server_id, **request.model_dump()
                )
            )
        )

    @router.get("/local/diagnostics", response_model=ItemResponse[dict[str, Any]])
    def local_diagnostics() -> ItemResponse[dict[str, Any]]:
        return ItemResponse(data=_call(lambda: get_lab().server_diagnostics()))

    @router.get("/local/hardware", response_model=ItemResponse[dict[str, Any]])
    def local_hardware() -> ItemResponse[dict[str, Any]]:
        return ItemResponse(data=_call(lambda: hardware_snapshot(executor_for(None))))

    @router.get("/local/environments", response_model=ListResponse[dict[str, Any]])
    def local_environments() -> ListResponse[dict[str, Any]]:
        records = _call(lambda: discover_environments(executor_for(None)))
        return _page(records, 200, 0)

    @router.post(
        "/local/environments/probe", response_model=ItemResponse[dict[str, Any]]
    )
    def probe_local_environment(request: EnvironmentProbe) -> ItemResponse[dict[str, Any]]:
        return ItemResponse(
            data=_call(lambda: get_lab().probe_environment(None, request.python))
        )

    @router.post(
        "/local/environments",
        response_model=ItemResponse[dict[str, Any]],
        status_code=201,
    )
    def create_local_environment(
        request: CondaEnvironmentCreate,
    ) -> ItemResponse[dict[str, Any]]:
        return ItemResponse(
            data=_call(
                lambda: get_lab().create_conda_environment(None, **request.model_dump())
            )
        )

    @router.get("/projects", response_model=ListResponse[ProjectResource])
    def projects(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ListResponse[ProjectResource]:
        records = [ProjectResource.model_validate(item) for item in get_lab().db.list("projects")]
        return _page(records, limit, offset)

    @router.post("/projects/inspect", response_model=ItemResponse[ProjectInspection])
    def inspect_project(request: ProjectInspect) -> ItemResponse[ProjectInspection]:
        record = _call(lambda: get_lab().inspect_project(request.source_path))
        return ItemResponse(data=ProjectInspection.model_validate(record))

    @router.post("/projects", response_model=ItemResponse[ProjectResource], status_code=201)
    def create_project(request: ProjectCreate) -> ItemResponse[ProjectResource]:
        record = _call(lambda: get_lab().add_project(request.source_path))
        return ItemResponse(data=ProjectResource.model_validate(record))

    @router.get("/projects/{project_id}", response_model=ItemResponse[ProjectResource])
    def project(project_id: str) -> ItemResponse[ProjectResource]:
        record = _call(lambda: get_lab().db.get("projects", project_id))
        return ItemResponse(data=ProjectResource.model_validate(record))

    @router.get("/projects/{project_id}/files", response_model=ListResponse[SourceFile])
    def project_files(project_id: str) -> ListResponse[SourceFile]:
        records = [
            SourceFile.model_validate(item)
            for item in _call(lambda: get_lab().project_files(project_id))
        ]
        return _page(records, 5000, 0)

    @router.get("/versions", response_model=ListResponse[VersionResource])
    def versions(
        project_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ListResponse[VersionResource]:
        items = get_lab().db.list("versions", **({"project_id": project_id} if project_id else {}))
        records = [VersionResource.model_validate(item) for item in items]
        return _page(records, limit, offset)

    @router.post(
        "/projects/{project_id}/versions",
        response_model=ItemResponse[VersionResource],
        status_code=201,
    )
    def create_version(project_id: str, request: VersionCreate) -> ItemResponse[VersionResource]:
        record = _call(lambda: get_lab().create_version(project_id, **request.model_dump()))
        return ItemResponse(data=VersionResource.model_validate(record))

    @router.get("/versions/{version_id}", response_model=ItemResponse[VersionResource])
    def version(version_id: str) -> ItemResponse[VersionResource]:
        record = _call(lambda: get_lab().db.get("versions", version_id))
        return ItemResponse(data=VersionResource.model_validate(record))

    @router.get("/versions/{version_id}/files", response_model=ListResponse[SourceFile])
    def version_files(version_id: str) -> ListResponse[SourceFile]:
        records = [
            SourceFile.model_validate(item)
            for item in _call(lambda: get_lab().version_files(version_id))
        ]
        return _page(records, 5000, 0)

    @router.get(
        "/versions/{version_id}/files/{source_path:path}",
        response_model=ItemResponse[SourceContent],
    )
    def version_source(
        version_id: str, source_path: str
    ) -> ItemResponse[SourceContent]:
        record = _call(lambda: get_lab().read_version_source(version_id, source_path))
        return ItemResponse(data=SourceContent.model_validate(record))

    @router.get("/versions/{version_id}/environments", response_model=ListResponse[dict[str, Any]])
    def version_environments(
        version_id: str,
        backend: Literal["cpu", "cuda", "npu"] = "cpu",
    ) -> ListResponse[dict[str, Any]]:
        records = _call(lambda: get_lab().version_environments(version_id, backend))
        return _page(records, 200, 0)

    @router.get("/runs", response_model=ListResponse[RunResource])
    def runs(
        project_id: str | None = None,
        status: RunStatus | None = None,
        task: str | None = None,
        backend: Literal["cpu", "cuda", "npu"] | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ListResponse[RunResource]:
        records = get_lab().db.list("runs", **({"project_id": project_id} if project_id else {}))
        if status:
            records = [record for record in records if record["status"] == status]
        if task:
            records = [record for record in records if record["task"] == task]
        if backend:
            records = [record for record in records if record["backend"] == backend]
        resources = [RunResource.model_validate(item) for item in records]
        return _page(resources, limit, offset)

    @router.post("/runs", response_model=ItemResponse[RunResource], status_code=201)
    def create_run(request: RunCreate) -> ItemResponse[RunResource]:
        data = request.model_dump()
        task = data.pop("task")
        version_id = data.pop("version_id")
        record = _call(lambda: get_lab().start_run(version_id, task_name=task, **data))
        return ItemResponse(data=RunResource.model_validate(record))

    @router.post("/runs/validate", response_model=ItemResponse[RunValidation])
    def validate_run(request: RunValidate) -> ItemResponse[RunValidation]:
        data = request.model_dump()
        task = data.pop("task")
        version_id = data.pop("version_id")
        record = _call(
            lambda: get_lab().validate_run(version_id, task_name=task, **data)
        )
        return ItemResponse(data=RunValidation.model_validate(record))

    @router.get("/runs/{run_id}", response_model=ItemResponse[RunResource])
    def run(run_id: str) -> ItemResponse[RunResource]:
        record = _call(lambda: get_lab().db.get("runs", run_id))
        return ItemResponse(data=RunResource.model_validate(record))

    @router.post("/runs/{run_id}/cancel", response_model=ItemResponse[RunResource])
    def cancel_run(run_id: str) -> ItemResponse[RunResource]:
        record = _call(lambda: get_lab().cancel_run(run_id))
        return ItemResponse(data=RunResource.model_validate(record))

    @router.post("/runs/{run_id}/rerun", response_model=ItemResponse[RunResource])
    def rerun(run_id: str, request: RunRerun) -> ItemResponse[RunResource]:
        record = _call(lambda: get_lab().rerun(run_id, request.name))
        return ItemResponse(data=RunResource.model_validate(record))

    @router.post("/runs/{run_id}/refresh", response_model=ItemResponse[RunResource])
    def refresh_run(run_id: str) -> ItemResponse[RunResource]:
        record = _call(lambda: get_lab().refresh_run(run_id))
        return ItemResponse(data=RunResource.model_validate(record))

    @router.get("/runs/{run_id}/events", response_model=ItemResponse[EventPage])
    def events(
        run_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=5000),
    ) -> ItemResponse[EventPage]:
        record = _call(lambda: get_lab().read_events(run_id, after, limit))
        return ItemResponse(data=EventPage.model_validate(record))

    @router.get("/runs/{run_id}/log", response_class=PlainTextResponse)
    def log(run_id: str, lines: int = Query(default=200, ge=1, le=2000)) -> str:
        return _call(lambda: get_lab().read_log(run_id, lines))

    @router.get("/runs/{run_id}/artifacts", response_model=ListResponse[ArtifactResource])
    def artifacts(run_id: str) -> ListResponse[ArtifactResource]:
        records = [
            ArtifactResource.model_validate(item)
            for item in _call(lambda: get_lab().list_artifacts(run_id))
        ]
        return _page(records, 5000, 0)

    @router.get("/runs/{run_id}/artifacts/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str) -> FileResponse:
        return FileResponse(_call(lambda: get_lab().artifact_path(run_id, artifact_path)))

    return router
