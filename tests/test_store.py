import json
import sqlite3
from pathlib import Path

from researchlab.store import Database


def test_database_migrates_existing_run_metadata_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE servers (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, ssh_alias TEXT NOT NULL,
                remote_root TEXT NOT NULL, shell_init TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
            );
            CREATE TABLE projects (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL,
                manifest TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE versions (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
                source_path TEXT NOT NULL, server_id TEXT, remote_path TEXT,
                manifest TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, version_id TEXT NOT NULL,
                server_id TEXT, name TEXT NOT NULL, task TEXT NOT NULL, status TEXT NOT NULL,
                params TEXT NOT NULL, devices TEXT NOT NULL, backend TEXT NOT NULL,
                run_path TEXT NOT NULL, remote_path TEXT, pid INTEGER, command TEXT NOT NULL,
                exit_code INTEGER, error TEXT, created_at TEXT NOT NULL, started_at TEXT,
                finished_at TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run", "project", "version", None, "legacy", "train", "completed",
                json.dumps({}), json.dumps([]), "cpu", "/tmp/run", None, None,
                json.dumps(["python", "train.py"]), 0, None,
                "2026-01-01T00:00:00+00:00", None, "2026-01-01T00:00:01+00:00",
            ),
        )

    database = Database(path)
    run = database.get("runs", "run")

    assert run["tags"] == []
    assert run["notes"] == ""
    assert database.list("operations") == []
