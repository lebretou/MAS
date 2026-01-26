"""API routes for trace data."""

from dataclasses import asdict

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
    created_at: str
    updated_at: str


class TraceIngestRequest(BaseModel):
    """Request body for trace ingestion."""

    events: list[dict]


def _summary_to_dict(summary: TraceSummary) -> dict:
    """Convert TraceSummary to a JSON-serializable dict."""
    d = asdict(summary)
    # convert tuple keys in messages_by_edge to string keys
    d["messages_by_edge"] = {
        f"{k[0]}->{k[1]}": v for k, v in summary.messages_by_edge.items()
    }
    return d


@router.get("/traces")
def list_traces_endpoint(limit: int = 100, offset: int = 0) -> list[TraceMetadata]:
    """List traces with metadata."""
    rows = list_traces(limit=limit, offset=offset)
    return [
        TraceMetadata(
            trace_id=row.trace_id,
            event_count=row.event_count,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


@router.post("/traces/{trace_id}/events")
def ingest_events(trace_id: str, request: TraceIngestRequest) -> dict:
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
    inserted = insert_events(trace_id, events)
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


@router.delete("/traces/{trace_id}")
def delete_trace_endpoint(trace_id: str) -> dict:
    """Delete a trace and all events."""
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    delete_trace(trace_id)
    return {"deleted": trace_id}
