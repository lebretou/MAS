"""API routes for graph topology management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.models.graph_topology import GraphEdge, GraphNode, GraphTopology
from backbone.utils.identifiers import utc_timestamp
from server.graph_db import (
    upsert_graph as db_upsert_graph,
    get_graph as db_get_graph,
    list_graphs as db_list_graphs,
    delete_graph as db_delete_graph,
)

router = APIRouter()


class UpsertGraphRequest(BaseModel):
    """request body for creating or updating a graph topology."""

    graph_id: str
    name: str
    description: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    state_schema: dict | None = None


@router.get("/graphs")
def list_graphs() -> list[GraphTopology]:
    """list all stored graph topologies."""
    return db_list_graphs()


@router.get("/graphs/{graph_id}")
def get_graph(graph_id: str) -> GraphTopology:
    """get a specific graph topology."""
    graph = db_get_graph(graph_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    return graph


@router.put("/graphs/{graph_id}")
def upsert_graph(graph_id: str, request: UpsertGraphRequest) -> GraphTopology:
    """create or update a graph topology.

    Uses PUT for idempotent upsert — safe to call on every compile.
    Also registers all agents that have prompt_id in their metadata.
    """
    from server.agent_db import upsert_agent as db_upsert_agent
    from backbone.models.agent_registry import AgentRegistryEntry

    now = utc_timestamp()

    existing = db_get_graph(graph_id)
    graph = GraphTopology(
        graph_id=graph_id,
        name=request.name,
        description=request.description,
        nodes=request.nodes,
        edges=request.edges,
        state_schema=request.state_schema,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )

    db_upsert_graph(graph)

    # auto-register all agent nodes that have metadata
    for node in request.nodes:
        if node.node_type != "agent":
            continue
        meta = node.metadata or {}
        entry = AgentRegistryEntry(
            agent_id=node.node_id,
            prompt_id=node.prompt_id,
            model=meta.get("model"),
            temperature=meta.get("temperature"),
            has_tools=meta.get("has_tools", False),
            metadata=meta,
            updated_at=now,
        )
        db_upsert_agent(entry)

    return graph


@router.delete("/graphs/{graph_id}")
def delete_graph(graph_id: str) -> dict:
    """delete a graph topology."""
    if not db_get_graph(graph_id):
        raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
    db_delete_graph(graph_id)
    return {"deleted": graph_id}
