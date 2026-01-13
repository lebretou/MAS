"""Trace summary analysis - reconstructs agent graph and detects failures."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from mas_backbone.models.trace_event import EventType, TraceEvent


@dataclass
class AgentEdge:
    """Represents communication between two agents."""

    from_agent: str
    to_agent: str
    message_count: int


@dataclass
class FailedContract:
    """Represents a failed contract validation."""

    contract_id: str
    contract_version: str
    failure_count: int


@dataclass
class ToolUsage:
    """Represents tool usage statistics."""

    tool_name: str
    call_count: int
    avg_latency_ms: float | None = None


@dataclass
class TraceSummary:
    """Summary of a trace including agent communication and failures."""

    execution_id: str
    trace_id: str
    agents: list[str]
    edges: list[AgentEdge]
    messages_by_edge: dict[tuple[str, str], int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    failed_contracts: list[FailedContract] = field(default_factory=list)
    tool_usage: list[ToolUsage] = field(default_factory=list)
    event_count: int = 0


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp."""
    # handle various ISO8601 formats
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def trace_summary(events: list[TraceEvent]) -> TraceSummary:
    """Reconstruct agent graph and detect failures from trace events.

    Args:
        events: List of TraceEvent objects from a single trace.

    Returns:
        TraceSummary with agent communication graph and failure information.
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

    # track agent-to-agent messages
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    # track failures
    failures: list[dict] = []

    # track failed contracts
    contract_failures: dict[tuple[str, str], int] = defaultdict(int)

    # track tool usage - (tool_name, span_id) -> (start_time, end_time)
    tool_calls: dict[str, dict] = defaultdict(lambda: {"count": 0, "latencies": []})
    tool_start_times: dict[str, datetime] = {}

    for event in events:
        agents_set.add(event.agent_id)

        if event.event_type == EventType.agent_message:
            from_agent = event.agent_id
            to_agent = event.payload.get("to_agent_id", "unknown")
            agents_set.add(to_agent)
            edge_counts[(from_agent, to_agent)] += 1

        elif event.event_type == EventType.contract_validation:
            result = event.payload.get("validation_result", {})
            if not result.get("is_valid", True):
                contract_id = event.payload.get("contract_id", "unknown")
                contract_version = event.payload.get("contract_version", "unknown")
                contract_failures[(contract_id, contract_version)] += 1
                failures.append({
                    "type": "contract_validation",
                    "agent_id": event.agent_id,
                    "contract_id": contract_id,
                    "contract_version": contract_version,
                    "errors": result.get("errors", []),
                    "timestamp": event.timestamp,
                })

        elif event.event_type == EventType.error:
            failures.append({
                "type": "error",
                "agent_id": event.agent_id,
                "error_type": event.payload.get("error_type", "unknown"),
                "message": event.payload.get("message", ""),
                "timestamp": event.timestamp,
            })

        elif event.event_type == EventType.tool_call:
            tool_name = event.payload.get("tool_name", "unknown")
            phase = event.payload.get("phase", "")
            span_id = event.span_id or ""

            if phase == "start":
                tool_start_times[span_id] = _parse_timestamp(event.timestamp)
                tool_calls[tool_name]["count"] += 1
            elif phase == "end" and span_id in tool_start_times:
                start_time = tool_start_times[span_id]
                end_time = _parse_timestamp(event.timestamp)
                latency_ms = (end_time - start_time).total_seconds() * 1000
                tool_calls[tool_name]["latencies"].append(latency_ms)
                del tool_start_times[span_id]

    # build edges list
    edges = [
        AgentEdge(from_agent=from_agent, to_agent=to_agent, message_count=count)
        for (from_agent, to_agent), count in edge_counts.items()
    ]

    # build failed contracts list
    failed_contracts = [
        FailedContract(contract_id=cid, contract_version=cver, failure_count=count)
        for (cid, cver), count in contract_failures.items()
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

    return TraceSummary(
        execution_id=execution_id,
        trace_id=trace_id,
        agents=sorted(agents_set),
        edges=edges,
        messages_by_edge=dict(edge_counts),
        failures=failures,
        failed_contracts=failed_contracts,
        tool_usage=tool_usage,
        event_count=len(events),
    )
