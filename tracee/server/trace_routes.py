"""API routes for trace data."""

from dataclasses import asdict
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.analysis.trace_summary import TraceSummary, trace_summary
from backbone.models.trace_event import TraceEvent
from server.trace_db import (
    TRACE_DB_PATH,
    delete_trace,
    get_trace,
    insert_events,
    list_traces,
    load_events,
)

router = APIRouter()


class TraceMetadata(BaseModel):
    """Metadata about a trace."""

    trace_id: str
    event_count: int
    graph_id: str | None = None
    created_at: str
    updated_at: str


class TraceIngestRequest(BaseModel):
    """Request body for trace ingestion."""

    events: list[dict]


_EST = timezone(timedelta(hours=-5))


def _utc_to_est(iso_utc: str) -> str:
    """convert a UTC ISO timestamp to EST for display."""
    dt = datetime.fromisoformat(iso_utc)
    return dt.astimezone(_EST).isoformat()


def _summary_to_dict(summary: TraceSummary) -> dict:
    """Convert TraceSummary to a JSON-serializable dict."""
    d = asdict(summary)
    # convert tuple keys in messages_by_edge to string keys
    d["messages_by_edge"] = {
        f"{k[0]}->{k[1]}": v for k, v in summary.messages_by_edge.items()
    }
    return d


@router.get("/traces")
def list_traces_endpoint(limit: int = 100, offset: int = 0, graph_id: str | None = None) -> list[TraceMetadata]:
    """List traces with metadata, optionally filtered by graph_id."""
    rows = list_traces(limit=limit, offset=offset, graph_id=graph_id)
    return [
        TraceMetadata(
            trace_id=row.trace_id,
            event_count=row.event_count,
            graph_id=row.graph_id,
            created_at=_utc_to_est(row.created_at),
            updated_at=_utc_to_est(row.updated_at),
        )
        for row in rows
    ]


@router.post("/traces/{trace_id}/events")
def ingest_events(trace_id: str, request: TraceIngestRequest, graph_id: str | None = None) -> dict:
    """Append events to a trace."""
    if not request.events:
        return {"trace_id": trace_id, "inserted": 0}
    events: list[TraceEvent] = []
    for payload in request.events:
        event = TraceEvent.model_validate(payload)
        if event.trace_id != trace_id:
            raise HTTPException(
                status_code=400,
                detail="trace_id mismatch between path and event payload",
            )
        events.append(event)
    inserted = insert_events(trace_id, events, graph_id=graph_id)
    return {"trace_id": trace_id, "inserted": inserted}


@router.get("/traces/{trace_id}")
def get_trace_events(trace_id: str, limit: int | None = None, offset: int = 0) -> list[dict]:
    """Get raw events for a trace."""
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    events = load_events(trace_id, limit=limit, offset=offset)
    return [event.model_dump() for event in events]


@router.get("/traces/{trace_id}/summary")
def get_trace_summary(trace_id: str) -> dict:
    """Get computed summary for a trace."""
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    events = load_events(trace_id)
    if not events:
        raise HTTPException(status_code=400, detail="Trace has no events")
    summary = trace_summary(events)
    return _summary_to_dict(summary)


@router.post("/traces/{trace_id}/analyze")
async def analyze_trace_endpoint(trace_id: str, graph_id: str | None = None) -> dict:
    """Run cognition analysis on a trace."""
    from server.cognition_db import get_cognition, insert_cognition_logs, upsert_cognition
    from server.cognition_service import run_cognition_analysis
    from server.graph_db import get_graph as db_get_graph, list_graphs as db_list_graphs
    from server.agent_db import get_agent as db_get_agent
    from server.prompt_db import get_latest_version

    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    events = load_events(trace_id)
    if not events:
        raise HTTPException(status_code=400, detail="Trace has no events")

    # resolve graph
    graph = None
    if graph_id:
        graph = db_get_graph(graph_id)
    if not graph:
        graphs = db_list_graphs()
        if graphs:
            graph = graphs[0]
    if not graph:
        raise HTTPException(status_code=400, detail="No graph topology registered")

    # resolve agent system prompts from the registry + prompt DB
    agent_prompts: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_type != "agent":
            continue
        agent = db_get_agent(node.node_id)
        if agent and agent.prompt_id:
            version = get_latest_version(agent.prompt_id)
            if version:
                agent_prompts[node.node_id] = version.resolve()

    cognition, logs = await run_cognition_analysis(
        trace_id=trace_id,
        events=events,
        graph=graph,
        agent_prompts=agent_prompts,
    )

    upsert_cognition(cognition)
    insert_cognition_logs(logs)

    return cognition.model_dump()


@router.get("/traces/{trace_id}/cognition")
def get_trace_cognition(trace_id: str) -> dict:
    """Get cached cognition results for a trace."""
    from server.cognition_db import get_cognition

    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

    cognition = get_cognition(trace_id)
    if not cognition:
        raise HTTPException(status_code=404, detail="No cognition results for this trace. Run POST /analyze first.")

    return cognition.model_dump()


@router.delete("/traces/{trace_id}")
def delete_trace_endpoint(trace_id: str) -> dict:
    """Delete a trace and all events."""
    from server.cognition_db import delete_cognition

    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    delete_trace(trace_id)
    delete_cognition(trace_id)
    return {"deleted": trace_id}
