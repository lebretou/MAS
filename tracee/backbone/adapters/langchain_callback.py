"""LangChain callback handler for automatic trace event emission.

Brief idea of callback in LangChain/LangGraph is a list of events happened during LLM system runs
it describes what the agents did (e.g., tool calls, actions, prompts)

This file translates LangChain callback events into our TraceEvent objects
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from backbone.adapters.event_api import EventEmitter, EventSink
from backbone.models.trace_event import EventType
from backbone.utils.identifiers import generate_span_id


def _classify_error(error: BaseException) -> str:
    """Classify an error into one of the valid error types."""
    error_name = type(error).__name__.lower()

    # model-related errors
    if any(x in error_name for x in ["openai", "anthropic", "llm", "api", "rate"]):
        return "model"

    # tool-related errors
    if any(x in error_name for x in ["tool", "function"]):
        return "tool"

    # infrastructure errors
    if any(x in error_name for x in ["timeout", "connection", "network", "http"]):
        return "infra"

    # schema/validation errors
    if any(x in error_name for x in ["validation", "schema", "parse", "json", "type"]):
        return "schema"

    # default to logic error
    return "logic"


class MASCallbackHandler(BaseCallbackHandler):
    """Emits low-level execution events from LangChain/LangGraph.

    Does NOT infer agent-to-agent messages (use manual API for that).

    Mapping:
    - on_chain_start  -> agent_input (only if chain = agent node)
    - on_chain_end    -> agent_output (only if chain = agent node)
    - on_llm_start    -> tool_call (phase=start, tool_name="llm.generate")
    - on_llm_end      -> tool_call (phase=end, tool_name="llm.generate")
    - on_tool_start   -> tool_call (phase=start)
    - on_tool_end     -> tool_call (phase=end)
    - on_chain_error  -> error (classified)
    """

    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        event_sink: EventSink,
        default_agent_id: str = "unknown",
    ) -> None:
        super().__init__()
        self.emitter = EventEmitter(execution_id, trace_id, event_sink)
        self.default_agent_id = default_agent_id
        # track span IDs for correlating start/end events
        self._run_span_map: dict[str, str] = {}
        # track agent IDs from chain metadata
        self._run_agent_map: dict[str, str] = {}

    def _get_agent_id(self, run_id: UUID | None, metadata: dict | None = None) -> str:
        """Get agent ID from run tracking or metadata."""
        if run_id and str(run_id) in self._run_agent_map:
            return self._run_agent_map[str(run_id)]
        if metadata:
            return metadata.get("agent_id", metadata.get("agent", self.default_agent_id))
        return self.default_agent_id

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

        return refs

    # --- chain callbacks (agent boundaries) ---

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
        """Handle chain start - emit agent_input if this is an agent node."""
        # store agent ID for this run if provided
        if metadata and ("agent_id" in metadata or "agent" in metadata):
            agent_id = metadata.get("agent_id", metadata.get("agent", self.default_agent_id))
            self._run_agent_map[str(run_id)] = agent_id

        agent_id = self._get_agent_id(run_id, metadata)
        refs = self._make_refs(run_id, parent_run_id, metadata)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        self.emitter.emit(
            EventType.agent_input,
            agent_id,
            refs=refs,
            payload={"input": inputs, "chain_name": serialized.get("name", "unknown")},
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle chain end - emit agent_output."""
        agent_id = self._get_agent_id(run_id)
        refs = self._make_refs(run_id, parent_run_id)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        self.emitter.emit(
            EventType.agent_output,
            agent_id,
            refs=refs,
            payload={"output": outputs},
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle chain error - emit error event with classification."""
        agent_id = self._get_agent_id(run_id)
        refs = self._make_refs(run_id, parent_run_id)
        error_type = _classify_error(error)

        self.emitter.emit_error(
            agent_id,
            error_type=error_type,
            message=str(error),
            details={"exception_type": type(error).__name__},
            refs=refs,
        )

    # --- LLM callbacks (treated as tool calls) ---

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
        """Handle LLM start - emit tool_call with phase=start."""
        agent_id = self._get_agent_id(parent_run_id, metadata)
        refs = self._make_refs(run_id, parent_run_id, metadata)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        # store model info in refs
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        refs["llm"] = {"model": model_name}

        self.emitter.emit_tool_call(
            agent_id,
            tool_name="llm.generate",
            phase="start",
            tool_input={"prompts": prompts[:1]},  # truncate for brevity
            refs=refs,
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle LLM end - emit tool_call with phase=end."""
        agent_id = self._get_agent_id(parent_run_id)
        refs = self._make_refs(run_id, parent_run_id)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        # extract generation text (truncated)
        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text[:200]

        self.emitter.emit_tool_call(
            agent_id,
            tool_name="llm.generate",
            phase="end",
            tool_output={"text": output_text},
            refs=refs,
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle LLM error - emit error event."""
        agent_id = self._get_agent_id(parent_run_id)
        refs = self._make_refs(run_id, parent_run_id)

        self.emitter.emit_error(
            agent_id,
            error_type="model",
            message=str(error),
            details={"exception_type": type(error).__name__},
            refs=refs,
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
        """Handle tool start - emit tool_call with phase=start."""
        agent_id = self._get_agent_id(parent_run_id, metadata)
        refs = self._make_refs(run_id, parent_run_id, metadata)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        tool_name = serialized.get("name", "unknown_tool")

        self.emitter.emit_tool_call(
            agent_id,
            tool_name=tool_name,
            phase="start",
            tool_input=inputs or {"raw": input_str[:500]},
            refs=refs,
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle tool end - emit tool_call with phase=end."""
        agent_id = self._get_agent_id(parent_run_id)
        refs = self._make_refs(run_id, parent_run_id)
        span_id = self._get_or_create_span(run_id)
        parent_span = self._run_span_map.get(str(parent_run_id)) if parent_run_id else None

        # convert output to string if needed
        output_str = str(output)[:500] if output else ""

        self.emitter.emit_tool_call(
            agent_id,
            tool_name="unknown_tool",  # we don't have tool name in on_tool_end
            phase="end",
            tool_output={"result": output_str},
            refs=refs,
            span_id=span_id,
            parent_span_id=parent_span,
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
        """Handle tool error - emit error event."""
        agent_id = self._get_agent_id(parent_run_id)
        refs = self._make_refs(run_id, parent_run_id)

        self.emitter.emit_error(
            agent_id,
            error_type="tool",
            message=str(error),
            details={"exception_type": type(error).__name__},
            refs=refs,
        )
