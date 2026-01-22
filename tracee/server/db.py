"""database initialization helpers."""

from server.playground_db import init_db as init_playground_db
from server.prompt_db import init_db as init_prompt_db
from server.trace_db import init_db as init_trace_db


def init_all() -> None:
    """initialize all sqlite tables."""
    init_trace_db()
    init_prompt_db()
    init_playground_db()
