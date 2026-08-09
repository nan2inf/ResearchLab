# API v1 contract

API v1 is the stable boundary between the ResearchLab controller and its web UI.
The generated specification is committed at [`openapi-v1.json`](openapi-v1.json)
and is also served at `GET /api/v1/openapi.json`.

## Conventions

- Resource responses use `{ "data": ... }`.
- Collection responses use `{ "data": [...], "meta": { "total", "limit", "offset" } }`.
- Errors use `{ "error": { "code", "message", "details", "retryable" } }`.
- Timestamps are ISO 8601 values in UTC.
- Event polling uses an integer cursor: pass the previous `data.next` as `after`.
- The controller is local and single-user. It never returns or accepts SSH private keys.
- Existing `/api/*` routes remain temporarily available to the current UI, but new UI
  work must use `/api/v1/*` only.

## Implemented endpoints

| Area | Method and path | Purpose |
| --- | --- | --- |
| Contract | `GET /api/v1/health` | Controller/tool availability |
| Contract | `GET /api/v1/capabilities` | Feature negotiation |
| Contract | `GET /api/v1/summary` | Dashboard counts and recent runs |
| Local host | `GET /api/v1/local/diagnostics` | Connection, hardware, environments |
| Local host | `GET /api/v1/local/hardware` | CPU, CUDA and Ascend status |
| Local host | `GET /api/v1/local/environments` | Conda/system Python probes |
| Local host | `POST /api/v1/local/environments/probe` | Probe a user-specified Python |
| Local host | `POST /api/v1/local/environments` | Create an isolated Conda environment |
| Servers | `GET, POST /api/v1/servers` | List and register SSH servers |
| Servers | `GET, PATCH /api/v1/servers/{id}` | Read or update server configuration |
| Servers | `GET /api/v1/servers/{id}/diagnostics` | Remote Linux diagnostics |
| Servers | `GET /api/v1/servers/{id}/environments` | Remote Conda/system Python probes |
| Servers | `POST /api/v1/servers/{id}/environments/probe` | Probe a remote Python path |
| Servers | `POST /api/v1/servers/{id}/environments` | Create a remote Conda environment |
| Projects | `GET, POST /api/v1/projects` | List and import projects |
| Projects | `GET /api/v1/projects/{id}` | Project manifest |
| Versions | `GET /api/v1/versions` | List immutable versions |
| Versions | `POST /api/v1/projects/{id}/versions` | Snapshot and optionally upload code |
| Versions | `GET /api/v1/versions/{id}` | Version metadata and paths |
| Versions | `GET /api/v1/versions/{id}/environments` | Compatible runtime candidates |
| Runs | `GET, POST /api/v1/runs` | Filter runs or start a run |
| Runs | `GET /api/v1/runs/{id}` | Run state and launch metadata |
| Runs | `POST /api/v1/runs/{id}/cancel` | Stop a local or remote run |
| Results | `GET /api/v1/runs/{id}/events` | Incremental metric/artifact events |
| Results | `GET /api/v1/runs/{id}/log` | Plain-text log tail |
| Results | `GET /api/v1/runs/{id}/artifacts/{path}` | Safely read a run artifact |

The `capabilities` response is authoritative. A client must hide or disable a
feature when its flag is false. Background operations and run comparison are
reserved for later milestones and are currently reported as unavailable.

## Regenerating the schema

From the repository root:

```bash
python scripts/export_openapi.py
```

The generated file must be reviewed and committed whenever a v1 request or
response model changes.
