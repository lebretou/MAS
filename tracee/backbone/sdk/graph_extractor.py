"""Extract graph topology from a compiled LangGraph and register it on the server.

Usage in user code (e.g. after create_workflow()):

    from backbone.sdk.graph_extractor import extract_and_register
    app = create_workflow()
    extract_and_register(app, graph_id="data-analysis", name="Data Analysis Workflow")
"""

from __future__ import annotations

import httpx

from backbone.models.graph_topology import GraphEdge, GraphNode, GraphTopology
from backbone.utils.identifiers import utc_timestamp


# langgraph synthetic node ids
_START = "__start__"
_END = "__end__"


def extract_topology(
    compiled_graph,
    graph_id: str,
    name: str,
    description: str | None = None,
) -> GraphTopology:
    """Build a GraphTopology from a compiled LangGraph.

    Reads node metadata (prompt_id, model, etc.) that the user attached
    via add_node(..., metadata={...}).
    """
    lc_graph = compiled_graph.get_graph()

    nodes: list[GraphNode] = []
    for node_id, node in lc_graph.nodes.items():
        if node_id == _START:
            nodes.append(GraphNode(node_id=node_id, label="Start", node_type="start"))
            continue
        if node_id == _END:
            nodes.append(GraphNode(node_id=node_id, label="End", node_type="end"))
            continue

        meta = node.metadata or {}
        nodes.append(GraphNode(
            node_id=node_id,
            label=meta.get("label", node.name),
            node_type="agent",
            prompt_id=meta.get("prompt_id"),
            metadata=meta if meta else None,
        ))

    edges: list[GraphEdge] = []
    for edge in lc_graph.edges:
        label = str(edge.data) if edge.data is not None else None
        edges.append(GraphEdge(
            source=edge.source,
            target=edge.target,
            conditional=edge.conditional,
            label=label,
        ))

    now = utc_timestamp()
    return GraphTopology(
        graph_id=graph_id,
        name=name,
        description=description,
        nodes=nodes,
        edges=edges,
        created_at=now,
        updated_at=now,
    )


def extract_and_register(
    compiled_graph,
    graph_id: str,
    name: str,
    description: str | None = None,
    base_url: str = "http://localhost:8000",
    timeout: float = 10.0,
) -> GraphTopology:
    """Extract topology from a compiled LangGraph and PUT it to the server.

    This registers the graph and all its agents in one call.
    The server-side PUT handler also auto-registers each agent node
    into the agent registry.
    """
    topology = extract_topology(compiled_graph, graph_id, name, description)

    url = f"{base_url.rstrip('/')}/api/graphs/{graph_id}"
    with httpx.Client(timeout=timeout) as client:
        response = client.put(url, json={
            "graph_id": topology.graph_id,
            "name": topology.name,
            "description": topology.description,
            "nodes": [n.model_dump() for n in topology.nodes],
            "edges": [e.model_dump() for e in topology.edges],
        })
        response.raise_for_status()

    return topology
