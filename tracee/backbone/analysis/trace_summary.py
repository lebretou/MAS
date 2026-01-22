"""This file has some helper functions that get basic statistics from raw trace events.

This module provides heuristic-based analysis of raw LangChain events.
For semantic analysis (agent messages, decisions), see agent_analyzer.py.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from backbone.models.trace_event import TraceEvent


@dataclass
class AgentEdge:
    """Represents inferred communication between two agents."""

    from_agent: str
    to_agent: str
    message_count: int


@dataclass
class ToolUsage:
    """Represents tool usage statistics."""

    tool_name: str
    call_count: int
    avg_latency_ms: float | None = None


@dataclass
class TraceSummary:
    """Summary of a trace with basic statistics.
    
    Note: Agent edges are inferred from event sequences, not explicit messages.
    For accurate agent communication analysis, use the agent_analyzer module.
    """

    execution_id: str
    trace_id: str
    agents: list[str]
    edges: list[AgentEdge]
    messages_by_edge: dict[tuple[str, str], int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    tool_usage: list[ToolUsage] = field(default_factory=list)
    llm_usage: list[ToolUsage] = field(default_factory=list)
    event_count: int = 0


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp."""
    # handle various ISO8601 formats
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def _get_agent_from_event(event: TraceEvent) -> str | None:
    """Extract agent identifier from an event.
    
    Checks multiple sources in order of priority:
    1. refs.hint.agent_id (explicitly set via metadata)
    2. refs.langgraph.node (from LangGraph workflow)
    3. event.agent_id field
    """
    # check hints from metadata
    if "hint" in event.refs:
        agent_id = event.refs["hint"].get("agent_id")
        if agent_id:
            return agent_id
    
    # check langgraph node
    if "langgraph" in event.refs:
        node = event.refs["langgraph"].get("node")
        if node:
            return node
    
    # fallback to agent_id field
    return event.agent_id


def trace_summary(events: list[TraceEvent]) -> TraceSummary:
    """Extract basic statistics from raw trace events.

    This provides heuristic-based analysis of the trace:
    - Tool/LLM usage counts and latencies
    - Error detection
    - Agent identification
    - Inferred agent transitions (based on event sequences)

    For semantic analysis (actual agent messages, decisions), use the
    agent_analyzer module instead.

    Args:
        events: List of TraceEvent objects from a single trace.

    Returns:
        TraceSummary with basic statistics and inferred information.
    """
    if not events:
        return TraceSummary(
            execution_id="",
            trace_id="",
            agents=[],
            edges=[],
        )

    # get execution_id and trace_id from first event
    execution_id = events[0].execution_id
    trace_id = events[0].trace_id

    # collect unique agents
    agents_set: set[str] = set()

    # track failures
    failures: list[dict] = []

    # track tool usage - tool_name -> {"count": N, "latencies": [...]}
    tool_calls: dict[str, dict] = defaultdict(lambda: {"count": 0, "latencies": []})
    tool_start_times: dict[str, tuple[str, datetime]] = {}  # span_id -> (tool_name, start_time)

    # track LLM usage separately
    llm_calls: dict[str, dict] = defaultdict(lambda: {"count": 0, "latencies": []})
    llm_start_times: dict[str, tuple[str, datetime]] = {}

    # track agent transitions for edge inference
    last_agent: str | None = None
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for event in events:
        # extract agent from event
        agent = _get_agent_from_event(event)
        if agent:
            agents_set.add(agent)
            # track agent transitions
            if last_agent and last_agent != agent:
                edge_counts[(last_agent, agent)] += 1
            last_agent = agent

        event_type = event.event_type

        # handle tool events
        if event_type == "on_tool_start":
            tool_name = event.payload.get("tool_name", "unknown")
            span_id = event.span_id or ""
            tool_start_times[span_id] = (tool_name, _parse_timestamp(event.timestamp))
            tool_calls[tool_name]["count"] += 1

        elif event_type == "on_tool_end":
            span_id = event.span_id or ""
            if span_id in tool_start_times:
                tool_name, start_time = tool_start_times[span_id]
                end_time = _parse_timestamp(event.timestamp)
                latency_ms = (end_time - start_time).total_seconds() * 1000
                tool_calls[tool_name]["latencies"].append(latency_ms)
                del tool_start_times[span_id]

        # handle LLM events
        elif event_type == "on_llm_start":
            model_name = event.payload.get("model_name", "unknown")
            span_id = event.span_id or ""
            llm_start_times[span_id] = (model_name, _parse_timestamp(event.timestamp))
            llm_calls[model_name]["count"] += 1

        elif event_type == "on_llm_end":
            span_id = event.span_id or ""
            if span_id in llm_start_times:
                model_name, start_time = llm_start_times[span_id]
                end_time = _parse_timestamp(event.timestamp)
                latency_ms = (end_time - start_time).total_seconds() * 1000
                llm_calls[model_name]["latencies"].append(latency_ms)
                del llm_start_times[span_id]

        # handle error events
        elif event_type in ("on_chain_error", "on_llm_error", "on_tool_error"):
            failures.append({
                "type": event_type,
                "agent_id": agent,
                "error_type": event.payload.get("error_type", "unknown"),
                "message": event.payload.get("error_message", ""),
                "timestamp": event.timestamp,
            })

    # build inferred edges list
    edges = [
        AgentEdge(from_agent=from_agent, to_agent=to_agent, message_count=count)
        for (from_agent, to_agent), count in edge_counts.items()
    ]

    # build tool usage list
    tool_usage = []
    for tool_name, data in tool_calls.items():
        latencies = data["latencies"]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        tool_usage.append(ToolUsage(
            tool_name=tool_name,
            call_count=data["count"],
            avg_latency_ms=avg_latency,
        ))

    # build LLM usage list
    llm_usage = []
    for model_name, data in llm_calls.items():
        latencies = data["latencies"]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        llm_usage.append(ToolUsage(
            tool_name=model_name,
            call_count=data["count"],
            avg_latency_ms=avg_latency,
        ))

    return TraceSummary(
        execution_id=execution_id,
        trace_id=trace_id,
        agents=sorted(a for a in agents_set if a),  # filter out None
        edges=edges,
        messages_by_edge=dict(edge_counts),
        failures=failures,
        tool_usage=tool_usage,
        llm_usage=llm_usage,
        event_count=len(events),
    )
