"""LangChain callback handler for raw trace event capture.

Captures LangChain/LangGraph events with minimal transformation.
Events are stored with their original event type names (e.g., 'on_llm_start').

Semantic analysis (agent messages, decisions, etc.) is performed by a separate
analysis layer after capture.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from backbone.models.trace_event import TraceEvent
from backbone.utils.identifiers import generate_event_id, generate_span_id, utc_timestamp


def _sanitize_for_json(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to string representations."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # for other types, convert to string
    return str(obj)


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


class RawCallbackHandler(BaseCallbackHandler):
    """Captures raw LangChain/LangGraph events with minimal transformation.
    
    Events are stored with their original event type names. No semantic inference
    is performed - that's handled by the analysis layer.
    
    Event types captured:
    - on_chain_start, on_chain_end, on_chain_error
    - on_llm_start, on_llm_end, on_llm_error
    - on_tool_start, on_tool_end, on_tool_error
    - on_chat_model_start (if available)
    """

    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        event_sink: EventSink,
    ) -> None:
        super().__init__()
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.event_sink = event_sink
        self._sequence = 0
        # track span IDs for correlating start/end events
        self._run_span_map: dict[str, str] = {}

    def _next_sequence(self) -> int:
        """Get the next sequence number."""
        seq = self._sequence
        self._sequence += 1
        return seq

    def _get_or_create_span(self, run_id: UUID | None) -> str:
        """Get existing span ID or create a new one for this run."""
        if run_id:
            run_key = str(run_id)
            if run_key not in self._run_span_map:
                self._run_span_map[run_key] = generate_span_id()
            return self._run_span_map[run_key]
        return generate_span_id()

    def _make_refs(
        self,
        run_id: UUID | None,
        parent_run_id: UUID | None,
        metadata: dict | None = None,
    ) -> dict:
        """Create namespaced refs dict from LangChain run info."""
        refs: dict = {}

        # langchain namespace
        lc_refs: dict = {}
        if run_id:
            lc_refs["run_id"] = str(run_id)
        if parent_run_id:
            lc_refs["parent_run_id"] = str(parent_run_id)
        if lc_refs:
            refs["langchain"] = lc_refs

        # langgraph namespace (if present in metadata)
        if metadata:
            lg_refs: dict = {}
            if "langgraph_node" in metadata:
                lg_refs["node"] = metadata["langgraph_node"]
            if "langgraph_state_keys" in metadata:
                lg_refs["state_keys"] = metadata["langgraph_state_keys"]
            if lg_refs:
                refs["langgraph"] = lg_refs

            # store any agent hint from metadata
            if "agent" in metadata or "agent_id" in metadata:
                refs["hint"] = {
                    "agent_id": metadata.get("agent_id", metadata.get("agent"))
                }

        return refs

    def _emit(
        self,
        event_type: str,
        run_id: UUID | None,
        parent_run_id: UUID | None,
        payload: dict,
        metadata: dict | None = None,
    ) -> TraceEvent:
        """Emit a raw event."""
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None
        refs = self._make_refs(run_id, parent_run_id, metadata)

        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=self.trace_id,
            execution_id=self.execution_id,
            timestamp=utc_timestamp(),
            sequence=self._next_sequence(),
            event_type=event_type,
            agent_id=None,  # not inferred - let analysis layer determine
            span_id=span_id,
            parent_span_id=parent_span,
            refs=refs,
            payload=payload,
        )
        self.event_sink.append(event)
        return event

    # --- chain callbacks ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture chain start event."""
        chain_name = serialized.get("name", "unknown") if serialized else "unknown"
        sanitized_inputs = _sanitize_for_json(inputs)

        self._emit(
            event_type="on_chain_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            payload={
                "chain_name": chain_name,
                "inputs": sanitized_inputs,
                "tags": tags,
            },
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture chain end event."""
        sanitized_outputs = _sanitize_for_json(outputs)

        self._emit(
            event_type="on_chain_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "outputs": sanitized_outputs,
                "tags": tags,
            },
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture chain error event."""
        self._emit(
            event_type="on_chain_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "tags": tags,
            },
        )

    # --- LLM callbacks ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM start event."""
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown") if serialized else "unknown"

        self._emit(
            event_type="on_llm_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            payload={
                "model_name": model_name,
                "prompts": prompts[:1],  # truncate for brevity
                "tags": tags,
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM end event."""
        # extract generation text (truncated)
        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text[:500]

        # extract token usage if available
        token_usage = None
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]

        self._emit(
            event_type="on_llm_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "output_text": output_text,
                "token_usage": token_usage,
                "tags": tags,
            },
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM error event."""
        self._emit(
            event_type="on_llm_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "tags": tags,
            },
        )

    # --- tool callbacks ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool start event."""
        tool_name = serialized.get("name", "unknown") if serialized else "unknown"

        self._emit(
            event_type="on_tool_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            payload={
                "tool_name": tool_name,
                "input": inputs or {"raw": input_str[:500]},
                "tags": tags,
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool end event."""
        output_str = str(output)[:500] if output else ""

        self._emit(
            event_type="on_tool_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "output": output_str,
                "tags": tags,
            },
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool error event."""
        self._emit(
            event_type="on_tool_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "tags": tags,
            },
        )


# keep old name as alias for backwards compatibility during migration
MASCallbackHandler = RawCallbackHandler
