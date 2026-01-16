"""
Trace event models for semantic events during execution.

For the data models, we choose pydantic, a library for data type validation

"""

from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator


class EventType(str, Enum):
    """Types of events in a trace."""

    agent_input = "agent_input"
    agent_output = "agent_output"
    agent_message = "agent_message"
    agent_decision = "agent_decision"
    tool_call = "tool_call"
    contract_validation = "contract_validation"
    prompt_resolved = "prompt_resolved"  # emitted when a prompt is loaded for an agent
    error = "error"


# valid error types for error events
ERROR_TYPES = {"schema", "tool", "model", "infra", "logic"}

# valid phases for tool_call events
TOOL_CALL_PHASES = {"start", "end"}


class TraceEvent(BaseModel):
    """A structured semantic event describing what happened during execution."""

    model_config = {"extra": "forbid"}

    event_id: str  # UUID for deduping
    trace_id: str
    execution_id: str
    timestamp: str
    sequence: int | None = None  # monotonic ordering within trace

    event_type: EventType
    agent_id: str

    span_id: str | None = None
    parent_span_id: str | None = None

    refs: dict[str, Any]  # namespaced: refs["langchain"], refs["langgraph"], etc.
    payload: dict[str, Any]

    @model_validator(mode="after")
    def validate_payload_invariants(self) -> Self:
        """Validate payload structure based on event_type."""
        payload = self.payload
        event_type = self.event_type

        if event_type == EventType.agent_message:
            self._validate_agent_message(payload)
        elif event_type == EventType.contract_validation:
            self._validate_contract_validation(payload)
        elif event_type == EventType.tool_call:
            self._validate_tool_call(payload)
        elif event_type == EventType.prompt_resolved:
            self._validate_prompt_resolved(payload)
        elif event_type == EventType.error:
            self._validate_error(payload)

        return self

    def _validate_agent_message(self, payload: dict) -> None:
        """agent_message requires to_agent_id and (message_summary OR payload_ref)."""
        if "to_agent_id" not in payload:
            raise ValueError("agent_message payload must contain 'to_agent_id'")
        if "message_summary" not in payload and "payload_ref" not in payload:
            raise ValueError(
                "agent_message payload must contain 'message_summary' or 'payload_ref'"
            )

    def _validate_contract_validation(self, payload: dict) -> None:
        """contract_validation requires validation_result with is_valid and errors."""
        if "validation_result" not in payload:
            raise ValueError(
                "contract_validation payload must contain 'validation_result'"
            )

        result = payload["validation_result"]
        if not isinstance(result, dict):
            raise ValueError("validation_result must be a dict")
        if "is_valid" not in result:
            raise ValueError("validation_result must contain 'is_valid'")
        if "errors" not in result:
            raise ValueError("validation_result must contain 'errors'")
        if not isinstance(result["errors"], list):
            raise ValueError("validation_result.errors must be a list")

        # contract identity required
        if "contract_id" not in payload:
            raise ValueError("contract_validation payload must contain 'contract_id'")
        if "contract_version" not in payload:
            raise ValueError(
                "contract_validation payload must contain 'contract_version'"
            )

    def _validate_tool_call(self, payload: dict) -> None:
        """tool_call requires tool_name and phase (start|end)."""
        if "tool_name" not in payload:
            raise ValueError("tool_call payload must contain 'tool_name'")
        if "phase" not in payload:
            raise ValueError("tool_call payload must contain 'phase'")
        if payload["phase"] not in TOOL_CALL_PHASES:
            raise ValueError(f"tool_call phase must be one of {TOOL_CALL_PHASES}")

    def _validate_prompt_resolved(self, payload: dict) -> None:
        """prompt_resolved requires prompt_id, version_id, and resolved_text."""
        if "prompt_id" not in payload:
            raise ValueError("prompt_resolved payload must contain 'prompt_id'")
        if "version_id" not in payload:
            raise ValueError("prompt_resolved payload must contain 'version_id'")
        if "resolved_text" not in payload:
            raise ValueError("prompt_resolved payload must contain 'resolved_text'")

    def _validate_error(self, payload: dict) -> None:
        """error requires error_type and message."""
        if "error_type" not in payload:
            raise ValueError("error payload must contain 'error_type'")
        if payload["error_type"] not in ERROR_TYPES:
            raise ValueError(f"error_type must be one of {ERROR_TYPES}")
        if "message" not in payload:
            raise ValueError("error payload must contain 'message'")
