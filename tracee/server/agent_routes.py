"""API routes for the agent registry."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.models.agent_registry import AgentRegistryEntry
from backbone.utils.identifiers import utc_timestamp
from server.agent_db import (
    upsert_agent as db_upsert_agent,
    get_agent as db_get_agent,
    list_agents as db_list_agents,
    delete_agent as db_delete_agent,
)

router = APIRouter()


class UpsertAgentRequest(BaseModel):
    """request body for registering or updating an agent."""

    prompt_id: str | None = None
    prompt_version_id: str | None = None
    model: str | None = None
    temperature: float | None = None
    has_tools: bool = False
    metadata: dict | None = None


@router.get("/agents")
def list_agents() -> list[AgentRegistryEntry]:
    """list all registered agents."""
    return db_list_agents()


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str) -> AgentRegistryEntry:
    """get a single agent's registry entry."""
    entry = db_get_agent(agent_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return entry


@router.put("/agents/{agent_id}")
def upsert_agent(agent_id: str, request: UpsertAgentRequest) -> AgentRegistryEntry:
    """register or update an agent's configuration.

    Uses PUT for idempotent upsert — safe to call repeatedly
    (e.g. from PromptLoader on every get() call).
    """
    existing = db_get_agent(agent_id)
    now = utc_timestamp()

    # merge: keep existing fields that aren't provided in this request
    if existing:
        entry = AgentRegistryEntry(
            agent_id=agent_id,
            prompt_id=request.prompt_id if request.prompt_id is not None else existing.prompt_id,
            prompt_version_id=request.prompt_version_id if request.prompt_version_id is not None else existing.prompt_version_id,
            model=request.model if request.model is not None else existing.model,
            temperature=request.temperature if request.temperature is not None else existing.temperature,
            has_tools=request.has_tools,
            metadata=request.metadata if request.metadata is not None else existing.metadata,
            updated_at=now,
        )
    else:
        entry = AgentRegistryEntry(
            agent_id=agent_id,
            prompt_id=request.prompt_id,
            prompt_version_id=request.prompt_version_id,
            model=request.model,
            temperature=request.temperature,
            has_tools=request.has_tools,
            metadata=request.metadata,
            updated_at=now,
        )

    db_upsert_agent(entry)
    return entry


@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: str) -> dict:
    """remove an agent from the registry."""
    if not db_get_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    db_delete_agent(agent_id)
    return {"deleted": agent_id}
