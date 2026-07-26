from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


JSON_COLUMNS = {"manifest", "params", "devices", "command", "metadata"}
TABLES = {"servers", "projects", "versions", "runs"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def app_home() -> Path:
    override = os.environ.get("RESEARCHLAB_HOME")
    return Path(override).expanduser().resolve() if override else Path.home() / ".researchlab"


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else app_home() / "researchlab.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._create_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_schema(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    ssh_alias TEXT NOT NULL,
                    remote_root TEXT NOT NULL,
                    shell_init TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    server_id TEXT REFERENCES servers(id) ON DELETE SET NULL,
                    remote_path TEXT,
                    manifest TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    version_id TEXT NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
                    server_id TEXT REFERENCES servers(id) ON DELETE SET NULL,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    devices TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    run_path TEXT NOT NULL,
                    remote_path TEXT,
                    pid INTEGER,
                    command TEXT NOT NULL,
                    exit_code INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);
                CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                """
            )
            server_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(servers)").fetchall()
            }
            if "shell_init" not in server_columns:
                db.execute("ALTER TABLE servers ADD COLUMN shell_init TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def insert(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        if table not in TABLES:
            raise ValueError(f"unsupported table: {table}")
        record = {"id": self.new_id(), "created_at": utc_now(), **data}
        encoded = {
            key: json.dumps(value, ensure_ascii=False) if key in JSON_COLUMNS else value
            for key, value in record.items()
        }
        columns = ", ".join(encoded)
        placeholders = ", ".join("?" for _ in encoded)
        with self._lock, self.connect() as db:
            db.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(encoded.values()),
            )
        return self.get(table, record["id"])

    def get(self, table: str, record_id: str) -> dict[str, Any]:
        if table not in TABLES:
            raise ValueError(f"unsupported table: {table}")
        with self.connect() as db:
            row = db.execute(f"SELECT * FROM {table} WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(f"{table[:-1]} not found: {record_id}")
        return self._decode(dict(row))

    def list(self, table: str, **filters: Any) -> list[dict[str, Any]]:
        if table not in TABLES:
            raise ValueError(f"unsupported table: {table}")
        where = ""
        values: tuple[Any, ...] = ()
        if filters:
            invalid = set(filters) - set(self.columns(table))
            if invalid:
                raise ValueError(f"unsupported filters: {invalid}")
            where = " WHERE " + " AND ".join(f"{key} = ?" for key in filters)
            values = tuple(filters.values())
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM {table}{where} ORDER BY created_at DESC", values
            ).fetchall()
        return [self._decode(dict(row)) for row in rows]

    def update(self, table: str, record_id: str, **changes: Any) -> dict[str, Any]:
        if table not in TABLES or not changes:
            raise ValueError("invalid update")
        invalid = set(changes) - set(self.columns(table))
        if invalid:
            raise ValueError(f"unsupported fields: {invalid}")
        encoded = {
            key: json.dumps(value, ensure_ascii=False) if key in JSON_COLUMNS else value
            for key, value in changes.items()
        }
        assignments = ", ".join(f"{key} = ?" for key in encoded)
        with self._lock, self.connect() as db:
            cursor = db.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                (*encoded.values(), record_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"{table[:-1]} not found: {record_id}")
        return self.get(table, record_id)

    def columns(self, table: str) -> list[str]:
        with self.connect() as db:
            return [row["name"] for row in db.execute(f"PRAGMA table_info({table})")]

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        for key in JSON_COLUMNS & row.keys():
            row[key] = json.loads(row[key])
        return row
