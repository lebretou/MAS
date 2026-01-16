"""Event sink and manual event emission API."""

import json
from pathlib import Path
from typing import Protocol

from backbone.models.trace_event import EventType, TraceEvent
from backbone.utils.identifiers import generate_event_id, generate_span_id, utc_timestamp


class EventSink(Protocol):
    """Protocol for receiving trace events."""

    def append(self, event: TraceEvent) -> None:
        """Append an event to the sink."""
        ...


class ListSink:
    """stores events in a list."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        """Append an event to the list."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()


class FileSink:
    """ writes events to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        """Append an event to the file."""
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")


class EventEmitter:
    """Manual event emission API - shares sink with callback handler."""

    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        event_sink: EventSink,
    ) -> None:
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.event_sink = event_sink
        self._sequence = 0

    def _next_sequence(self) -> int:
        """Get the next sequence number."""
        seq = self._sequence
        self._sequence += 1
        return seq

    def emit(
        self,
        event_type: EventType,
        agent_id: str,
        refs: dict | None = None,
        payload: dict | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceEvent:
        """Emit a trace event with the given parameters."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=self.trace_id,
            execution_id=self.execution_id,
            timestamp=utc_timestamp(),
            sequence=self._next_sequence(),
            event_type=event_type,
            agent_id=agent_id,
            span_id=span_id or generate_span_id(),
            parent_span_id=parent_span_id,
            refs=refs or {},
            payload=payload or {},
        )
        self.event_sink.append(event)
        return event

    def emit_input(
        self,
        agent_id: str,
        input_data: dict | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit an agent_input event."""
        payload = {"input": input_data} if input_data else {}
        return self.emit(EventType.agent_input, agent_id, refs=refs, payload=payload)

    def emit_output(
        self,
        agent_id: str,
        output_data: dict | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit an agent_output event."""
        payload = {"output": output_data} if output_data else {}
        return self.emit(EventType.agent_output, agent_id, refs=refs, payload=payload)

    def emit_message(
        self,
        from_agent: str,
        to_agent: str,
        summary: str | None = None,
        payload_ref: str | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit an explicit agent-to-agent handoff message."""
        payload: dict = {"to_agent_id": to_agent}
        if summary:
            payload["message_summary"] = summary
        if payload_ref:
            payload["payload_ref"] = payload_ref
        return self.emit(EventType.agent_message, from_agent, refs=refs, payload=payload)

    def emit_decision(
        self,
        agent_id: str,
        decision: str,
        reasoning: str | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit an agent decision event."""
        payload: dict = {"decision": decision}
        if reasoning:
            payload["reasoning"] = reasoning
        return self.emit(EventType.agent_decision, agent_id, refs=refs, payload=payload)

    def emit_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        phase: str,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
        refs: dict | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceEvent:
        """Emit a tool call event (start or end phase)."""
        payload: dict = {"tool_name": tool_name, "phase": phase}
        if tool_input:
            payload["input"] = tool_input
        if tool_output:
            payload["output"] = tool_output
        return self.emit(
            EventType.tool_call,
            agent_id,
            refs=refs,
            payload=payload,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

    def emit_validation(
        self,
        agent_id: str,
        contract_id: str,
        contract_version: str,
        is_valid: bool,
        errors: list[dict] | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit a contract validation event."""
        payload = {
            "contract_id": contract_id,
            "contract_version": contract_version,
            "validation_result": {
                "is_valid": is_valid,
                "errors": errors or [],
            },
        }
        return self.emit(EventType.contract_validation, agent_id, refs=refs, payload=payload)

    def emit_error(
        self,
        agent_id: str,
        error_type: str,
        message: str,
        details: dict | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit an error event."""
        payload: dict = {"error_type": error_type, "message": message}
        if details:
            payload["details"] = details
        return self.emit(EventType.error, agent_id, refs=refs, payload=payload)

    def emit_prompt_resolved(
        self,
        agent_id: str,
        prompt_id: str,
        version_id: str,
        resolved_text: str,
        components: list[dict] | None = None,
        variables_used: dict[str, str] | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit a prompt_resolved event when a prompt is loaded for an agent.
        
        This captures a snapshot of the exact prompt text used at runtime,
        enabling trace visualization to show which prompt was used.
        
        Args:
            agent_id: The agent that loaded this prompt
            prompt_id: The prompt identifier
            version_id: The version of the prompt
            resolved_text: The full resolved prompt text (all enabled components)
            components: Optional list of component dicts with type, content, enabled
            variables_used: Optional dict of template variables that were substituted
            refs: Optional additional references
        """
        payload: dict = {
            "prompt_id": prompt_id,
            "version_id": version_id,
            "resolved_text": resolved_text,
        }
        if components:
            payload["components"] = components
        if variables_used:
            payload["variables_used"] = variables_used
        return self.emit(EventType.prompt_resolved, agent_id, refs=refs, payload=payload)
