"""SQLite storage for traces and events."""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backbone.models.trace_event import TraceEvent


DEFAULT_DB_PATH = Path(__file__).parent / "data" / "tracee.db"
TRACE_DB_PATH = Path(os.getenv("TRACE_DB_PATH", str(DEFAULT_DB_PATH)))


@dataclass
class TraceRow:
    trace_id: str
    event_count: int
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            create table if not exists traces (
                trace_id text primary key,
                event_count integer not null default 0,
                created_at text not null,
                updated_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists trace_events (
                id integer primary key autoincrement,
                trace_id text not null,
                event_json text not null,
                event_type text,
                timestamp text,
                sequence integer
            )
            """
        )
        conn.execute(
            "create index if not exists idx_trace_events_trace_id on trace_events(trace_id)"
        )
        conn.execute(
            "create index if not exists idx_trace_events_timestamp on trace_events(timestamp)"
        )
        conn.commit()


def upsert_trace(trace_id: str, event_count_delta: int) -> None:
    now = _utc_now()
    with _connect() as conn:
        existing = conn.execute(
            "select trace_id from traces where trace_id = ?",
            (trace_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                update traces
                set event_count = event_count + ?, updated_at = ?
                where trace_id = ?
                """,
                (event_count_delta, now, trace_id),
            )
        else:
            conn.execute(
                """
                insert into traces (trace_id, event_count, created_at, updated_at)
                values (?, ?, ?, ?)
                """,
                (trace_id, event_count_delta, now, now),
            )
        conn.commit()


def insert_events(trace_id: str, events: list[TraceEvent]) -> int:
    if not events:
        return 0
    rows = []
    for event in events:
        rows.append(
            (
                trace_id,
                event.model_dump_json(),
                event.event_type,
                event.timestamp,
                event.sequence,
            )
        )
    with _connect() as conn:
        conn.executemany(
            """
            insert into trace_events (
                trace_id,
                event_json,
                event_type,
                timestamp,
                sequence
            )
            values (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    upsert_trace(trace_id, len(events))
    return len(events)


def list_traces(limit: int = 100, offset: int = 0) -> list[TraceRow]:
    with _connect() as conn:
        rows = conn.execute(
            """
            select trace_id, event_count, created_at, updated_at
            from traces
            order by updated_at desc
            limit ? offset ?
            """,
            (limit, offset),
        ).fetchall()
    return [
        TraceRow(
            trace_id=row["trace_id"],
            event_count=row["event_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def load_events(trace_id: str, limit: int | None = None, offset: int = 0) -> list[TraceEvent]:
    with _connect() as conn:
        if limit is None:
            rows = conn.execute(
                """
                select event_json
                from trace_events
                where trace_id = ?
                order by id asc
                """,
                (trace_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select event_json
                from trace_events
                where trace_id = ?
                order by id asc
                limit ? offset ?
                """,
                (trace_id, limit, offset),
            ).fetchall()
    events = []
    for row in rows:
        events.append(TraceEvent.model_validate_json(row["event_json"]))
    return events


def get_trace(trace_id: str) -> TraceRow | None:
    with _connect() as conn:
        row = conn.execute(
            """
            select trace_id, event_count, created_at, updated_at
            from traces
            where trace_id = ?
            """,
            (trace_id,),
        ).fetchone()
    if not row:
        return None
    return TraceRow(
        trace_id=row["trace_id"],
        event_count=row["event_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def delete_trace(trace_id: str) -> None:
    with _connect() as conn:
        conn.execute("delete from trace_events where trace_id = ?", (trace_id,))
        conn.execute("delete from traces where trace_id = ?", (trace_id,))
        conn.commit()


init_db()
