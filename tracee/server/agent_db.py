"""SQLite storage for the agent registry."""

import sqlite3

from backbone.models.agent_registry import AgentRegistryEntry
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
            create table if not exists agent_registry (
                agent_id text primary key,
                entry_json text not null,
                prompt_id text,
                updated_at text not null
            )
            """
        )
        conn.execute(
            "create index if not exists idx_agent_registry_prompt_id on agent_registry(prompt_id)"
        )
        conn.commit()


def upsert_agent(entry: AgentRegistryEntry) -> None:
    """insert or update an agent registry entry."""
    with _connect() as conn:
        conn.execute(
            """
            insert into agent_registry (agent_id, entry_json, prompt_id, updated_at)
            values (?, ?, ?, ?)
            on conflict(agent_id) do update set
                entry_json = excluded.entry_json,
                prompt_id = excluded.prompt_id,
                updated_at = excluded.updated_at
            """,
            (
                entry.agent_id,
                entry.model_dump_json(),
                entry.prompt_id,
                entry.updated_at,
            ),
        )
        conn.commit()


def get_agent(agent_id: str) -> AgentRegistryEntry | None:
    with _connect() as conn:
        row = conn.execute(
            "select entry_json from agent_registry where agent_id = ?",
            (agent_id,),
        ).fetchone()
    if not row:
        return None
    return AgentRegistryEntry.model_validate_json(row["entry_json"])


def list_agents() -> list[AgentRegistryEntry]:
    with _connect() as conn:
        rows = conn.execute(
            "select entry_json from agent_registry order by agent_id"
        ).fetchall()
    return [AgentRegistryEntry.model_validate_json(row["entry_json"]) for row in rows]


def delete_agent(agent_id: str) -> None:
    with _connect() as conn:
        conn.execute("delete from agent_registry where agent_id = ?", (agent_id,))
        conn.commit()
