"""
Trace event models for raw events (think of it as the log of actions performed by agents) during execution.

Events are captured directly from LangChain/LangGraph callbacks 

For the data models, we choose pydantic, a library for data type validation.
"""

from typing import Any, Self

from pydantic import BaseModel, model_validator


# the only custom event type we define - auto-emitted by prompt SDK
# having this because LangChain doesn't tell you which prompt version/id was 
# being executed only the raw input
PROMPT_RESOLVED = "prompt_resolved"


class TraceEvent(BaseModel):
    """
    Events are stored with their original LangChain event type names
    (e.g., 'on_llm_start', 'on_chain_end', 'on_tool_start').
    
    The only custom event type is 'prompt_resolved', which is auto-emitted
    by the prompt SDK when a prompt is loaded.
    """

    model_config = {"extra": "forbid"}

    event_id: str  # UUID for deduping
    trace_id: str
    execution_id: str
    timestamp: str
    sequence: int | None = None  # monotonic ordering within trace

    event_type: str  # raw LangChain event name or 'prompt_resolved'

    # optional identifiers - may not be available for all events
    agent_id: str | None = None

    span_id: str | None = None
    parent_span_id: str | None = None

    refs: dict[str, Any]  # namespaced: refs["langchain"], refs["langgraph"], etc.
    payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_payload_invariants(self) -> Self:
        """Validate payload structure for custom event types."""
        if self.event_type == PROMPT_RESOLVED:
            self._validate_prompt_resolved(self.payload)
        # raw LangChain events pass through without validation
        return self

    def _validate_prompt_resolved(self, payload: dict) -> None:
        """prompt_resolved requires prompt_id, version_id, and resolved_text."""
        if "prompt_id" not in payload:
            raise ValueError("prompt_resolved payload must contain 'prompt_id'")
        if "version_id" not in payload:
            raise ValueError("prompt_resolved payload must contain 'version_id'")
        if "resolved_text" not in payload:
            raise ValueError("prompt_resolved payload must contain 'resolved_text'")
