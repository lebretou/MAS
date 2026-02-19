"""SQLite storage for graph topologies."""

import sqlite3

from backbone.models.graph_topology import GraphTopology
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
            create table if not exists graphs (
                graph_id text primary key,
                graph_json text not null,
                name text not null,
                created_at text not null,
                updated_at text not null
            )
            """
        )
        conn.commit()


def upsert_graph(graph: GraphTopology) -> None:
    """insert or update a graph topology."""
    with _connect() as conn:
        conn.execute(
            """
            insert into graphs (graph_id, graph_json, name, created_at, updated_at)
            values (?, ?, ?, ?, ?)
            on conflict(graph_id) do update set
                graph_json = excluded.graph_json,
                name = excluded.name,
                updated_at = excluded.updated_at
            """,
            (
                graph.graph_id,
                graph.model_dump_json(),
                graph.name,
                graph.created_at,
                graph.updated_at,
            ),
        )
        conn.commit()


def get_graph(graph_id: str) -> GraphTopology | None:
    with _connect() as conn:
        row = conn.execute(
            "select graph_json from graphs where graph_id = ?",
            (graph_id,),
        ).fetchone()
    if not row:
        return None
    return GraphTopology.model_validate_json(row["graph_json"])


def list_graphs() -> list[GraphTopology]:
    with _connect() as conn:
        rows = conn.execute(
            "select graph_json from graphs order by updated_at desc"
        ).fetchall()
    return [GraphTopology.model_validate_json(row["graph_json"]) for row in rows]


def delete_graph(graph_id: str) -> None:
    with _connect() as conn:
        conn.execute("delete from graphs where graph_id = ?", (graph_id,))
        conn.commit()
