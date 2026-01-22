"""SQLite storage for playground runs."""

import sqlite3
from dataclasses import dataclass

from backbone.models.playground_run import PlaygroundRun
from server.trace_db import TRACE_DB_PATH


@dataclass
class PlaygroundRunRow:
    run_id: str
    prompt_id: str
    version_id: str
    created_at: str


def _connect() -> sqlite3.Connection:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            create table if not exists playground_runs (
                run_id text primary key,
                run_json text not null,
                prompt_id text not null,
                version_id text not null,
                created_at text not null
            )
            """
        )
        conn.execute(
            """
            create index if not exists idx_playground_runs_prompt_id
            on playground_runs(prompt_id)
            """
        )
        conn.execute(
            """
            create index if not exists idx_playground_runs_created_at
            on playground_runs(created_at)
            """
        )
        conn.commit()


def insert_run(run: PlaygroundRun) -> None:
    with _connect() as conn:
        conn.execute(
            """
            insert into playground_runs (
                run_id,
                run_json,
                prompt_id,
                version_id,
                created_at
            )
            values (?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.model_dump_json(),
                run.prompt_id,
                run.version_id,
                run.created_at,
            ),
        )
        conn.commit()


def list_runs(limit: int = 50, prompt_id: str | None = None) -> list[PlaygroundRun]:
    with _connect() as conn:
        if prompt_id:
            rows = conn.execute(
                """
                select run_json
                from playground_runs
                where prompt_id = ?
                order by created_at desc
                limit ?
                """,
                (prompt_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select run_json
                from playground_runs
                order by created_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
    return [PlaygroundRun.model_validate_json(row["run_json"]) for row in rows]


def get_run(run_id: str) -> PlaygroundRun | None:
    with _connect() as conn:
        row = conn.execute(
            "select run_json from playground_runs where run_id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    return PlaygroundRun.model_validate_json(row["run_json"])


init_db()
