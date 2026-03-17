"""
Trace event models for raw events (think of it as the log of actions performed by agents) during execution.

Events are captured directly from LangChain/LangGraph callbacks 

For the data models, we choose pydantic, a library for data type validation.
"""

from typing import Any

from pydantic import BaseModel


class TraceEvent(BaseModel):
    """
    Events are stored with their original LangChain event type names
    (e.g., 'on_llm_start', 'on_chain_end', 'on_tool_start').
    """

    model_config = {"extra": "forbid"}

    event_id: str  # UUID for deduping
    trace_id: str
    execution_id: str
    timestamp: str
    sequence: int | None = None  # monotonic ordering within trace

    event_type: str

    # optional identifiers - may not be available for all events
    agent_id: str | None = None

    span_id: str | None = None
    parent_span_id: str | None = None

    refs: dict[str, Any]  # namespaced: refs["langchain"], refs["langgraph"], etc.
    payload: dict[str, Any]
