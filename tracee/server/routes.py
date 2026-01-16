"""API routes for trace data."""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.analysis.trace_summary import TraceSummary, trace_summary
from backbone.models.trace_event import TraceEvent

router = APIRouter()

# configurable traces directory (stored alongside prompts in server/data/)
DEFAULT_TRACES_DIR = Path(__file__).parent / "data" / "traces"
TRACES_DIR = Path(os.getenv("TRACES_DIR", str(DEFAULT_TRACES_DIR)))


class TraceMetadata(BaseModel):
    """Metadata about a trace."""
    trace_id: str
    event_count: int
    created_at: str | None
    file_size_bytes: int


def _get_trace_file(trace_id: str) -> Path:
    """Get path to trace file, raise 404 if not found."""
    trace_dir = TRACES_DIR / trace_id
    trace_file = trace_dir / "trace_events.jsonl"
    
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    
    return trace_file


def _load_trace_events(trace_file: Path) -> list[TraceEvent]:
    """Load trace events from a JSONL file."""
    events = []
    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            event = TraceEvent.model_validate(data)
            events.append(event)
    return events


# def _summary_to_dict(summary: TraceSummary) -> dict:
#     """Convert TraceSummary to a JSON-serializable dict."""
#     d = asdict(summary)
#     # convert tuple keys in messages_by_edge to string keys
#     d["messages_by_edge"] = {
#         f"{k[0]}->{k[1]}": v for k, v in summary.messages_by_edge.items()
#     }
#     return d


@router.get("/traces")
def list_traces() -> list[TraceMetadata]:
    """List all available traces with metadata."""
    traces = []
    
    if not TRACES_DIR.exists():
        return traces
    
    for trace_dir in TRACES_DIR.iterdir():
        if not trace_dir.is_dir():
            continue
        
        trace_file = trace_dir / "trace_events.jsonl"
        if not trace_file.exists():
            continue
        
        # count events
        event_count = 0
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    event_count += 1
        
        # get file stats
        stat = trace_file.stat()
        created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        traces.append(TraceMetadata(
            trace_id=trace_dir.name,
            event_count=event_count,
            created_at=created_at,
            file_size_bytes=stat.st_size,
        ))
    
    # sort by created_at descending (newest first)
    traces.sort(key=lambda t: t.created_at or "", reverse=True)
    
    return traces


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> list[dict]:
    """Get all events for a specific trace."""
    trace_file = _get_trace_file(trace_id)
    events = _load_trace_events(trace_file)
    return [event.model_dump() for event in events]


# @router.get("/traces/{trace_id}/summary")
# def get_trace_summary(trace_id: str) -> dict:
#     """Get computed summary for a specific trace."""
#     trace_file = _get_trace_file(trace_id)
#     events = _load_trace_events(trace_file)
    
#     if not events:
#         raise HTTPException(status_code=400, detail="Trace has no events")
    
#     summary = trace_summary(events)
#     return _summary_to_dict(summary)
