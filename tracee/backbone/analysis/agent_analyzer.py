"""Segment extraction and grouping for cognition layer analysis.

Extracts per-node execution segments from raw trace events, producing
structured data that can be fed to the LLM judge. This is the server-side
equivalent of the frontend's computeOverlay logic.
"""

import json
from datetime import datetime
from typing import Any

from backbone.models.cognition import NodeSegment
from backbone.models.trace_event import TraceEvent


def _get_direct_node(event: TraceEvent) -> str | None:
    """get the agent/node id directly annotated on this event."""
    if "hint" in event.refs and "agent_id" in event.refs["hint"]:
        return event.refs["hint"]["agent_id"]
    if "langgraph" in event.refs and "node" in event.refs["langgraph"]:
        return event.refs["langgraph"]["node"]
    return event.agent_id


def _get_run_id(event: TraceEvent) -> str | None:
    lc = event.refs.get("langchain")
    if isinstance(lc, dict):
        return lc.get("run_id")
    return None


def _get_parent_run_id(event: TraceEvent) -> str | None:
    lc = event.refs.get("langchain")
    if isinstance(lc, dict):
        return lc.get("parent_run_id")
    return None


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _parse_ts_ms(ts: str) -> float:
    return _parse_ts(ts).timestamp() * 1000


def _truncate(value: Any, max_chars: int = 4000) -> Any:
    """truncate large string values for LLM context."""
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"... (truncated, {len(value) - max_chars} chars omitted)"
    if isinstance(value, dict):
        return {k: _truncate(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(item, max_chars) for item in value[:30]]
    return value


def _diff_state_keys(before: dict, after: dict) -> list[str]:
    """return sorted list of keys that differ between two state dicts."""
    all_keys = set(before.keys()) | set(after.keys())
    changed = []
    for key in all_keys:
        if json.dumps(before.get(key), sort_keys=True) != json.dumps(after.get(key), sort_keys=True):
            changed.append(key)
    return sorted(changed)


def _get_tags(event: TraceEvent) -> list[str]:
    raw = event.payload.get("tags")
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str)]


def _classify_tool(tool_name: str, tags: list[str]) -> str:
    """classify a tool call into a category."""
    tag_map = {
        "tracee:rag": "rag_retrieve",
        "tracee:code_exec": "code_exec",
        "tracee:tool": "tool_call",
    }
    for tag in tags:
        if tag in tag_map:
            return tag_map[tag]
    normalized = tool_name.lower()
    if any(p in normalized for p in ("retrieve", "rag", "vector_search", "embed")):
        return "rag_retrieve"
    if any(p in normalized for p in ("execute", "run_code", "python_repl", "bash", "shell")):
        return "code_exec"
    return "tool_call"


def _resolve_event_to_node(
    event: TraceEvent,
    node_id_set: set[str],
    run_to_node: dict[str, str],
    run_parent: dict[str, str],
) -> str | None:
    """resolve which graph node an event belongs to by walking the run_id parent chain.

    mirrors the frontend's resolveAgentNodeId logic:
    1. check direct annotations (hint.agent_id, langgraph.node, agent_id)
    2. check run_to_node for this event's run_id
    3. walk up parent_run_id chain until we find a mapped run
    """
    direct = _get_direct_node(event)
    if direct and direct in node_id_set:
        return direct

    run_id = _get_run_id(event)
    if run_id and run_id in run_to_node:
        return run_to_node[run_id]

    # walk up parent chain
    parent_id = _get_parent_run_id(event)
    visited: set[str] = set()
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        if parent_id in run_to_node:
            return run_to_node[parent_id]
        parent_id = run_parent.get(parent_id)

    # also check span_id parent chain
    if event.parent_span_id:
        # not tracked as deeply, but try direct
        pass

    return None


def group_events_by_agent(
    events: list[TraceEvent],
    node_ids: list[str] | None = None,
) -> dict[str, list[TraceEvent]]:
    """group trace events by agent node id using run_id parent chain resolution.

    if node_ids is provided, only events that resolve to those nodes are included.
    events that can't be resolved are grouped under 'unknown'.
    """
    node_id_set = set(node_ids) if node_ids else None

    # pass 1: build run_id -> parent_run_id map and run_id -> node map from direct annotations
    run_parent: dict[str, str] = {}
    run_to_node: dict[str, str] = {}

    for event in events:
        run_id = _get_run_id(event)
        parent_run_id = _get_parent_run_id(event)
        if run_id and parent_run_id:
            run_parent[run_id] = parent_run_id

        direct = _get_direct_node(event)
        if direct and run_id:
            if node_id_set is None or direct in node_id_set:
                run_to_node[run_id] = direct

    # pass 2: resolve every event to a node
    grouped: dict[str, list[TraceEvent]] = {}
    for event in events:
        node = _resolve_event_to_node(event, node_id_set or set(), run_to_node, run_parent)
        if node is None:
            node = "unknown"
        if node not in grouped:
            grouped[node] = []
        grouped[node].append(event)

    return grouped


def extract_node_segment(
    events: list[TraceEvent],
    agent_id: str,
    all_agent_ids: list[str],
    graph_edges: list[dict] | None = None,
) -> NodeSegment:
    """extract a structured execution segment for a single agent node.

    Args:
        events: trace events already filtered/grouped for this agent.
        agent_id: the node id being analyzed.
        all_agent_ids: all agent node ids in the graph (for upstream inference).
        graph_edges: optional graph edge dicts with "source" and "target" keys.
    """
    sorted_events = sorted(events, key=lambda e: (e.sequence or 0, _parse_ts_ms(e.timestamp)))

    # determine upstream agents from graph edges
    upstream_agents: list[str] = []
    if graph_edges:
        upstream_agents = [
            e["source"] for e in graph_edges
            if e.get("target") == agent_id and e["source"] in all_agent_ids
        ]

    input_state: dict[str, Any] | None = None
    output_state: dict[str, Any] | None = None
    operations: list[dict[str, Any]] = []

    # track start events for pairing with end events
    llm_starts: dict[str, TraceEvent] = {}  # span_id -> event
    tool_starts: dict[str, TraceEvent] = {}
    chain_starts: dict[str, TraceEvent] = {}  # run_id -> event

    for event in sorted_events:
        et = event.event_type

        if et == "on_chain_start":
            inputs = event.payload.get("inputs")
            if isinstance(inputs, dict) and input_state is None:
                input_state = _truncate(inputs)
            run_id = _get_run_id(event)
            if run_id:
                chain_starts[run_id] = event

        elif et == "on_chain_end":
            outputs = event.payload.get("outputs")
            if isinstance(outputs, dict):
                output_state = _truncate(outputs)

        elif et == "on_llm_start":
            if event.span_id:
                llm_starts[event.span_id] = event

        elif et == "on_llm_end":
            start = llm_starts.get(event.span_id or "") if event.span_id else None
            model_name = (
                (start.payload.get("model_name") if start else None)
                or event.payload.get("model_name")
                or "llm"
            )
            latency_ms = None
            if start:
                latency_ms = round(_parse_ts_ms(event.timestamp) - _parse_ts_ms(start.timestamp))

            llm_input = None
            if start:
                llm_input = _truncate(
                    start.payload.get("input")
                    or start.payload.get("messages")
                    or start.payload.get("prompts")
                )
            llm_output = _truncate(
                event.payload.get("output_text") or event.payload.get("output")
            )

            token_usage = event.payload.get("token_usage")
            operations.append({
                "type": "llm_call",
                "id": event.event_id,
                "label": str(model_name),
                "latency_ms": latency_ms,
                "input": llm_input,
                "output": llm_output,
                "token_usage": _truncate(token_usage) if token_usage else None,
            })

        elif et == "on_tool_start":
            if event.span_id:
                tool_starts[event.span_id] = event

        elif et == "on_tool_end":
            start = tool_starts.get(event.span_id or "") if event.span_id else None
            tool_name = (start.payload.get("tool_name") if start else None) or "tool"
            latency_ms = None
            if start:
                latency_ms = round(_parse_ts_ms(event.timestamp) - _parse_ts_ms(start.timestamp))

            all_tags = (_get_tags(start) if start else []) + _get_tags(event)
            op_type = _classify_tool(str(tool_name), all_tags)

            operations.append({
                "type": op_type,
                "id": event.event_id,
                "label": str(tool_name),
                "latency_ms": latency_ms,
                "input": _truncate(start.payload.get("input") if start else None),
                "output": _truncate(event.payload.get("output")),
            })

        elif et.endswith("_error"):
            operations.append({
                "type": "error",
                "id": event.event_id,
                "label": event.payload.get("error_type", et),
                "error_message": _truncate(event.payload.get("error_message", ""), 2000),
            })

    # compute changed keys
    changed_keys: list[str] = []
    if isinstance(input_state, dict) and isinstance(output_state, dict):
        changed_keys = _diff_state_keys(input_state, output_state)

    return NodeSegment(
        agent_id=agent_id,
        upstream_agents=upstream_agents,
        input_state=input_state,
        output_state=output_state,
        changed_keys=changed_keys,
        operations=operations,
    )


def extract_all_segments(
    events: list[TraceEvent],
    agent_ids: list[str],
    graph_edges: list[dict] | None = None,
) -> dict[str, NodeSegment]:
    """extract segments for all agent nodes in a trace."""
    grouped = group_events_by_agent(events, node_ids=agent_ids)
    segments: dict[str, NodeSegment] = {}

    for agent_id in agent_ids:
        agent_events = grouped.get(agent_id, [])
        if not agent_events:
            segments[agent_id] = NodeSegment(agent_id=agent_id)
            continue
        segments[agent_id] = extract_node_segment(
            agent_events, agent_id, agent_ids, graph_edges
        )

    return segments
