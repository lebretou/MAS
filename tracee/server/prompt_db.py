"""SQLite storage for prompts and versions."""

import os
import sqlite3
from dataclasses import dataclass

from backbone.models.prompt_artifact import Prompt, PromptVersion
from server.trace_db import TRACE_DB_PATH


@dataclass
class PromptRow:
    prompt_id: str
    name: str
    description: str | None
    latest_version_id: str | None
    created_at: str
    updated_at: str


def _connect() -> sqlite3.Connection:
    TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            create table if not exists prompts (
                prompt_id text primary key,
                prompt_json text not null,
                name text not null,
                description text,
                latest_version_id text,
                created_at text not null,
                updated_at text not null
            )
            """
        )
        conn.execute(
            """
            create table if not exists prompt_versions (
                prompt_id text not null,
                version_id text not null,
                version_json text not null,
                name text not null,
                created_at text not null,
                primary key (prompt_id, version_id)
            )
            """
        )
        conn.execute(
            """
            create index if not exists idx_prompt_versions_prompt_id
            on prompt_versions(prompt_id)
            """
        )
        conn.commit()


def create_prompt(prompt: Prompt) -> None:
    with _connect() as conn:
        conn.execute(
            """
            insert into prompts (
                prompt_id,
                prompt_json,
                name,
                description,
                latest_version_id,
                created_at,
                updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt.prompt_id,
                prompt.model_dump_json(),
                prompt.name,
                prompt.description,
                prompt.latest_version_id,
                prompt.created_at,
                prompt.updated_at,
            ),
        )
        conn.commit()


def update_prompt(prompt: Prompt) -> None:
    with _connect() as conn:
        conn.execute(
            """
            update prompts
            set prompt_json = ?,
                name = ?,
                description = ?,
                latest_version_id = ?,
                created_at = ?,
                updated_at = ?
            where prompt_id = ?
            """,
            (
                prompt.model_dump_json(),
                prompt.name,
                prompt.description,
                prompt.latest_version_id,
                prompt.created_at,
                prompt.updated_at,
                prompt.prompt_id,
            ),
        )
        conn.commit()


def get_prompt(prompt_id: str) -> Prompt | None:
    with _connect() as conn:
        row = conn.execute(
            "select prompt_json from prompts where prompt_id = ?",
            (prompt_id,),
        ).fetchone()
    if not row:
        return None
    return Prompt.model_validate_json(row["prompt_json"])


def get_prompt_row(prompt_id: str) -> PromptRow | None:
    with _connect() as conn:
        row = conn.execute(
            """
            select prompt_id, name, description, latest_version_id, created_at, updated_at
            from prompts
            where prompt_id = ?
            """,
            (prompt_id,),
        ).fetchone()
    if not row:
        return None
    return PromptRow(
        prompt_id=row["prompt_id"],
        name=row["name"],
        description=row["description"],
        latest_version_id=row["latest_version_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_prompts() -> list[PromptRow]:
    with _connect() as conn:
        rows = conn.execute(
            """
            select prompt_id, name, description, latest_version_id, created_at, updated_at
            from prompts
            order by updated_at desc
            """
        ).fetchall()
    return [
        PromptRow(
            prompt_id=row["prompt_id"],
            name=row["name"],
            description=row["description"],
            latest_version_id=row["latest_version_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def delete_prompt(prompt_id: str) -> None:
    with _connect() as conn:
        conn.execute("delete from prompt_versions where prompt_id = ?", (prompt_id,))
        conn.execute("delete from prompts where prompt_id = ?", (prompt_id,))
        conn.commit()


def insert_version(version: PromptVersion) -> None:
    with _connect() as conn:
        conn.execute(
            """
            insert into prompt_versions (
                prompt_id,
                version_id,
                version_json,
                name,
                created_at
            )
            values (?, ?, ?, ?, ?)
            """,
            (
                version.prompt_id,
                version.version_id,
                version.model_dump_json(),
                version.name,
                version.created_at,
            ),
        )
        conn.commit()


def list_versions(prompt_id: str) -> list[PromptVersion]:
    with _connect() as conn:
        rows = conn.execute(
            """
            select version_json
            from prompt_versions
            where prompt_id = ?
            order by created_at desc
            """,
            (prompt_id,),
        ).fetchall()
    return [PromptVersion.model_validate_json(row["version_json"]) for row in rows]


def get_version(prompt_id: str, version_id: str) -> PromptVersion | None:
    with _connect() as conn:
        row = conn.execute(
            """
            select version_json
            from prompt_versions
            where prompt_id = ? and version_id = ?
            """,
            (prompt_id, version_id),
        ).fetchone()
    if not row:
        return None
    return PromptVersion.model_validate_json(row["version_json"])


def get_latest_version(prompt_id: str) -> PromptVersion | None:
    with _connect() as conn:
        row = conn.execute(
            "select latest_version_id from prompts where prompt_id = ?",
            (prompt_id,),
        ).fetchone()
    if not row:
        return None
    latest_id = row["latest_version_id"]
    if not latest_id:
        return None
    return get_version(prompt_id, latest_id)


