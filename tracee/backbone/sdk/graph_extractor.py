"""Extract graph topology from a compiled LangGraph and register it on the server.

Usage in user code (e.g. after create_workflow()):

    from backbone.sdk.graph_extractor import extract_and_register
    app = create_workflow()
    extract_and_register(app, graph_id="data-analysis", name="Data Analysis Workflow")
"""

from __future__ import annotations

import re
import typing
import httpx

from backbone.models.graph_topology import GraphEdge, GraphNode, GraphTopology
from backbone.utils.identifiers import utc_timestamp


# langgraph synthetic node ids
_START = "__start__"
_END = "__end__"


def _normalize_id_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return normalized or "branch"


def _extract_state_schema(compiled_graph) -> dict | None:
    """best-effort extraction of the state TypedDict into a JSON-schema-like dict."""
    builder = getattr(compiled_graph, "builder", None)
    schema_cls = getattr(builder, "state_schema", None)
    if schema_cls is None:
        # fallback for older graph builder versions
        schema_cls = getattr(builder, "schema", None)
    if schema_cls is None:
        return None

    hints = typing.get_type_hints(schema_cls, include_extras=True)
    if not hints:
        return None

    properties: dict[str, dict] = {}
    for field_name, annotation in hints.items():
        origin = typing.get_origin(annotation)

        # unwrap Annotated
        if origin is typing.Annotated:
            args = typing.get_args(annotation)
            annotation = args[0] if args else annotation
            origin = typing.get_origin(annotation)

        type_str = getattr(annotation, "__name__", None) or str(annotation)
        if origin is not None:
            args = typing.get_args(annotation)
            arg_strs = [getattr(a, "__name__", str(a)) for a in args] if args else []
            base = getattr(origin, "__name__", str(origin))
            type_str = f"{base}[{', '.join(arg_strs)}]" if arg_strs else base

        properties[field_name] = {"type": type_str}

    return {
        "type": "object",
        "description": getattr(schema_cls, "__doc__", None) or "",
        "properties": properties,
    }


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

    existing_node_ids = {node.node_id for node in nodes}
    conditional_end_nodes: dict[tuple[str, str], str] = {}
    edges: list[GraphEdge] = []
    for idx, edge in enumerate(lc_graph.edges):
        label = str(edge.data) if edge.data is not None else None
        target = edge.target

        if edge.conditional and edge.target == _END:
            branch_key = label or f"branch-{idx}"
            branch_signature = (edge.source, branch_key)
            conditional_end_id = conditional_end_nodes.get(branch_signature)
            if conditional_end_id is None:
                source_part = _normalize_id_part(edge.source)
                branch_part = _normalize_id_part(branch_key)
                base_node_id = f"__end__conditional__{source_part}__{branch_part}"
                conditional_end_id = base_node_id
                suffix = 1
                while conditional_end_id in existing_node_ids:
                    conditional_end_id = f"{base_node_id}-{suffix}"
                    suffix += 1

                conditional_end_nodes[branch_signature] = conditional_end_id
                existing_node_ids.add(conditional_end_id)
                nodes.append(GraphNode(
                    node_id=conditional_end_id,
                    label="End",
                    node_type="end",
                ))

            target = conditional_end_id

        edges.append(GraphEdge(
            source=edge.source,
            target=target,
            conditional=edge.conditional,
            label=label,
        ))

    state_schema = _extract_state_schema(compiled_graph)

    now = utc_timestamp()
    return GraphTopology(
        graph_id=graph_id,
        name=name,
        description=description,
        nodes=nodes,
        edges=edges,
        state_schema=state_schema,
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
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.put(url, json={
                "graph_id": topology.graph_id,
                "name": topology.name,
                "description": topology.description,
                "nodes": [n.model_dump() for n in topology.nodes],
                "edges": [e.model_dump() for e in topology.edges],
                "state_schema": topology.state_schema,
            })
            response.raise_for_status()
    except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
        import warnings
        warnings.warn(
            f"failed to register graph topology with server at {base_url}: {exc}",
            stacklevel=2,
        )

    return topology
