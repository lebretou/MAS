"""Event sink and manual event emission API."""

from pathlib import Path

from backbone.models.trace_event import PROMPT_RESOLVED, TraceEvent
from backbone.utils.identifiers import generate_event_id, generate_span_id, utc_timestamp


class FileSink:
    """Writes events to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        """Append an event to the file."""
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")


class EventEmitter:
    """Manual event emission API.
    
    This is primarily used for emitting prompt_resolved events. 
    Raw LangChain events are captured automatically by the RawCallbackHandler.
    
    For most use cases, prefer using the enable_tracing() context manager
    from backbone.sdk.tracing instead of creating an EventEmitter directly.
    """

    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        event_sink,  # EventSink protocol
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
        event_type: str,
        payload: dict,
        agent_id: str | None = None,
        refs: dict | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraceEvent:
        """Emit a trace event with the given parameters.
        
        Args:
            event_type: The event type string
            payload: Event payload data
            agent_id: Optional agent identifier
            refs: Optional references dict
            span_id: Optional span ID
            parent_span_id: Optional parent span ID
            
        Returns:
            The created TraceEvent
        """
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
            payload=payload,
        )
        self.event_sink.append(event)
        return event

    def emit_prompt_resolved(
        self,
        prompt_id: str,
        version_id: str,
        resolved_text: str,
        agent_id: str | None = None,
        components: list[dict] | None = None,
        variables_used: dict[str, str] | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit a prompt_resolved event when a prompt is loaded.
        
        This captures a snapshot of the exact prompt text used at runtime,
        enabling trace visualization to show which prompt was used.
        
        Args:
            prompt_id: The prompt identifier
            version_id: The version of the prompt
            resolved_text: The full resolved prompt text (all enabled components)
            agent_id: Optional agent that loaded this prompt
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
        return self.emit(PROMPT_RESOLVED, payload, agent_id=agent_id, refs=refs)
