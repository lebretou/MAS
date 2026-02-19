"""Data model for persisted graph topology.

Stores the full structure of a user's LangGraph workflow so the UI
can render the intent layer without recompilation.
"""

from pydantic import BaseModel


class GraphNode(BaseModel):
    """a node in the graph, typically an agent."""

    node_id: str
    label: str
    node_type: str = "agent"  # "agent", "start", "end"
    prompt_id: str | None = None
    metadata: dict | None = None  # model, temperature, has_tools, etc.


class GraphEdge(BaseModel):
    """a directed edge between two nodes."""

    source: str
    target: str
    conditional: bool = False
    label: str | None = None


class GraphTopology(BaseModel):
    """the full graph structure as defined by the user."""

    graph_id: str
    name: str
    description: str | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    created_at: str
    updated_at: str
