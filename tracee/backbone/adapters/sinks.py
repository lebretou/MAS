"""Event sinks for trace storage."""

import json
import urllib.request
from pathlib import Path

from backbone.models.trace_event import TraceEvent


class EventSink:
    """Protocol for receiving trace events."""

    def append(self, event: TraceEvent) -> None:
        """Append an event to the sink."""
        raise NotImplementedError


class ListSink(EventSink):
    """Stores events in a list."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        """Append an event to the list."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()


class FileSink(EventSink):
    """Writes events to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        """Append an event to the file."""
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")


class HttpSink(EventSink):
    """Posts events to the trace API."""

    def __init__(self, base_url: str, trace_id: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.trace_id = trace_id
        self.timeout = timeout

    def append(self, event: TraceEvent) -> None:
        payload = {"events": [event.model_dump()]}
        url = f"{self.base_url}/api/traces/{self.trace_id}/events"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except (urllib.error.URLError, urllib.error.HTTPError):
            # Silently ignore network errors to avoid crashing the agent
            # In production, consider logging these errors
            pass
