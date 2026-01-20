"""Tracing SDK - single entry point for enabling trace capture.

This module provides a simple way to enable tracing for LangChain/LangGraph
applications. Just wrap your code with enable_tracing() and pass the callbacks
to your LangChain components.

Example:
    from tracee.backbone.sdk import enable_tracing
    
    with enable_tracing() as ctx:
        result = graph.invoke(state, config={"callbacks": ctx.callbacks})
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from backbone.adapters.langchain_callback import RawCallbackHandler, EventSink, ListSink
from backbone.adapters.event_api import FileSink
from backbone.models.trace_event import PROMPT_RESOLVED, TraceEvent
from backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_span_id,
    generate_trace_id,
    utc_timestamp,
)


# global active tracing context (set by enable_tracing context manager)
_active_context: TracingContext | None = None


def get_active_context() -> TracingContext | None:
    """Get the currently active tracing context, if any.
    
    This is used by the prompt SDK to auto-emit prompt_resolved events.
    """
    return _active_context


@dataclass
class TracingContext:
    """Context object returned by enable_tracing().
    
    Provides access to callbacks for LangChain and methods for emitting
    custom events (primarily prompt_resolved).
    """
    
    trace_id: str
    execution_id: str
    callback_handler: RawCallbackHandler
    event_sink: EventSink
    _sequence: int = 0
    
    @property
    def callbacks(self) -> list:
        """Get the list of callbacks to pass to LangChain.
        
        Usage:
            graph.invoke(state, config={"callbacks": ctx.callbacks})
        """
        return [self.callback_handler]
    
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
    ) -> TraceEvent:
        """Emit a custom trace event.
        
        This is primarily used for prompt_resolved events, but can be used
        for any custom event type.
        """
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=self.trace_id,
            execution_id=self.execution_id,
            timestamp=utc_timestamp(),
            sequence=self._next_sequence(),
            event_type=event_type,
            agent_id=agent_id,
            span_id=generate_span_id(),
            parent_span_id=None,
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
    ) -> TraceEvent:
        """Emit a prompt_resolved event.
        
        This is called automatically by the prompt SDK when load_prompt() is used.
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
        return self.emit(PROMPT_RESOLVED, payload, agent_id=agent_id)


@contextmanager
def enable_tracing(
    trace_id: str | None = None,
    execution_id: str | None = None,
    output_dir: str | Path | None = None,
    output_file: str = "trace_events.jsonl",
) -> Generator[TracingContext, None, None]:
    """Enable tracing for LangChain/LangGraph code.
    
    This is the single entry point for enabling tracing. It creates all necessary
    components (sink, callback handler) and provides a context object for accessing
    callbacks and emitting custom events.
    
    Args:
        trace_id: Optional trace ID (auto-generated if not provided)
        execution_id: Optional execution ID (auto-generated if not provided)
        output_dir: Directory to write trace files to (creates {trace_id}/ subdirectory)
                   If not provided, traces are stored in memory only.
        output_file: Name of the trace events file (default: trace_events.jsonl)
    
    Yields:
        TracingContext with callbacks and emit methods
    
    Example:
        # basic usage - traces stored in memory
        with enable_tracing() as ctx:
            result = graph.invoke(state, config={"callbacks": ctx.callbacks})
        
        # write traces to file
        with enable_tracing(output_dir="./traces") as ctx:
            result = graph.invoke(state, config={"callbacks": ctx.callbacks})
        
        # custom trace ID
        with enable_tracing(trace_id="my-trace-123", output_dir="./traces") as ctx:
            result = graph.invoke(state, config={"callbacks": ctx.callbacks})
    """
    global _active_context
    
    # generate IDs if not provided
    tid = trace_id or generate_trace_id()
    eid = execution_id or generate_execution_id()
    
    # create sink
    if output_dir:
        output_path = Path(output_dir) / tid / output_file
        sink: EventSink = FileSink(output_path)
    else:
        sink = ListSink()
    
    # create callback handler
    handler = RawCallbackHandler(
        execution_id=eid,
        trace_id=tid,
        event_sink=sink,
    )
    
    # create context
    context = TracingContext(
        trace_id=tid,
        execution_id=eid,
        callback_handler=handler,
        event_sink=sink,
    )
    
    # set global context for prompt SDK
    _active_context = context
    
    try:
        yield context
    finally:
        _active_context = None


def load_prompt(
    prompt_id: str,
    version_id: str = "latest",
    agent_id: str | None = None,
    base_url: str = "http://localhost:8000",
) -> str:
    """Load a prompt and auto-emit prompt_resolved if tracing is active.
    
    This is a convenience function that combines prompt loading with automatic
    trace emission. If tracing is active (inside enable_tracing()), a
    prompt_resolved event is automatically emitted.
    
    Args:
        prompt_id: The prompt identifier
        version_id: The version to load ("latest" for most recent)
        agent_id: Optional agent ID to associate with the event
        base_url: Base URL of the tracee server
    
    Returns:
        The resolved prompt text
    
    Example:
        with enable_tracing(output_dir="./traces") as ctx:
            system_prompt = load_prompt("planner-system", agent_id="planner")
            # ^ this automatically emits a prompt_resolved event
    """
    from backbone.sdk.prompt_loader import PromptLoader
    
    loader = PromptLoader(base_url=base_url)
    version = loader.get_version(prompt_id, version_id)
    resolved_text = version.resolve()
    
    # auto-emit if tracing is active
    ctx = get_active_context()
    if ctx:
        ctx.emit_prompt_resolved(
            prompt_id=prompt_id,
            version_id=version.version_id,
            resolved_text=resolved_text,
            agent_id=agent_id,
            components=[
                {
                    "type": c.type.value,
                    "content": c.content,
                    "enabled": c.enabled,
                }
                for c in version.components
            ],
            variables_used=version.variables,
        )
    
    return resolved_text
