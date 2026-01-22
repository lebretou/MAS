"""ID generation and timestamp utilities."""

import uuid
from datetime import datetime, timezone


def generate_execution_id() -> str:
    """Generate a unique execution ID (UUID4)."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID (UUID4)."""
    return str(uuid.uuid4())


def generate_event_id() -> str:
    """Generate a unique event ID (UUID4)."""
    return str(uuid.uuid4())


def generate_span_id() -> str:
    """Generate a span ID (16-char hex string)."""
    return uuid.uuid4().hex[:16]


def generate_run_id() -> str:
    """Generate a unique playground run ID (UUID4)."""
    return str(uuid.uuid4())


def generate_config_id() -> str:
    """Generate a unique model config ID (UUID4)."""
    return str(uuid.uuid4())


def utc_timestamp() -> str:
    """Generate an ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
