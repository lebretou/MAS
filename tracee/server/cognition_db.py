"""SQLite storage for cognition layer results."""

import sqlite3

from backbone.models.cognition import CognitionLog, TraceCognition
from server.trace_db import TRACE_DB_PATH


def _connect() -> sqlite3.Connection:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            create table if not exists trace_cognitions (
                trace_id text primary key,
                cognition_json text not null,
                model_used text,
                created_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists cognition_logs (
                id integer primary key autoincrement,
                trace_id text not null,
                agent_id text,
                llm_input text not null,
                llm_output text not null,
                model text,
                tokens_used integer,
                created_at text not null
            )
            """
        )
        conn.execute(
            "create index if not exists idx_cognition_logs_trace_id on cognition_logs(trace_id)"
        )
        conn.commit()


def upsert_cognition(cognition: TraceCognition, model_used: str | None = None) -> None:
    """insert or replace cognition results for a trace."""
    with _connect() as conn:
        conn.execute(
            """
            insert into trace_cognitions (trace_id, cognition_json, model_used, created_at)
            values (?, ?, ?, ?)
            on conflict(trace_id) do update set
                cognition_json = excluded.cognition_json,
                model_used = excluded.model_used,
                created_at = excluded.created_at
            """,
            (
                cognition.trace_id,
                cognition.model_dump_json(),
                model_used,
                cognition.created_at,
            ),
        )
        conn.commit()


def get_cognition(trace_id: str) -> TraceCognition | None:
    with _connect() as conn:
        row = conn.execute(
            "select cognition_json from trace_cognitions where trace_id = ?",
            (trace_id,),
        ).fetchone()
    if not row:
        return None
    # discard stale cached data that doesn't match current schema
    try:
        return TraceCognition.model_validate_json(row["cognition_json"])
    except Exception:
        delete_cognition(trace_id)
        return None


def insert_cognition_logs(logs: list[CognitionLog]) -> int:
    if not logs:
        return 0
    rows = [
        (
            log.trace_id,
            log.agent_id,
            log.llm_input,
            log.llm_output,
            log.model,
            log.tokens_used,
            log.created_at,
        )
        for log in logs
    ]
    with _connect() as conn:
        conn.executemany(
            """
            insert into cognition_logs (trace_id, agent_id, llm_input, llm_output, model, tokens_used, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def delete_cognition(trace_id: str) -> None:
    """delete cognition results and logs for a trace."""
    with _connect() as conn:
        conn.execute("delete from trace_cognitions where trace_id = ?", (trace_id,))
        conn.execute("delete from cognition_logs where trace_id = ?", (trace_id,))
        conn.commit()
