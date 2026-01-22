"""Tracing SDK -  provides a simple way to enable tracing for LangChain/LangGraph
applications.

Example (so we do this in the config file of langgraph):
    from tracee.backbone.sdk import enable_tracing
    
    with enable_tracing() as ctx:
        result = graph.invoke(state, config={"callbacks": ctx.callbacks})
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from backbone.adapters.event_api import EventEmitter
from backbone.adapters.langchain_callback import RawCallbackHandler
from backbone.adapters.sinks import EventSink, FileSink, HttpSink, ListSink
from backbone.models.trace_event import TraceEvent
from backbone.utils.identifiers import (
    generate_execution_id,
    generate_trace_id,
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
    emitter: EventEmitter
    
    @property
    def callbacks(self) -> list:
        """Get the list of callbacks to pass to LangChain.
        
        Usage:
            graph.invoke(state, config={"callbacks": ctx.callbacks})
        """
        return [self.callback_handler]
    
    def emit(
        self,
        event_type: str,
        payload: dict,
        agent_id: str | None = None,
        refs: dict | None = None,
    ) -> TraceEvent:
        """Emit a custom trace event.
        
        This is currently only used for prompt_resolved events, but can be used
        for any custom event type if we have more in the future.
        """
        return self.emitter.emit(
            event_type,
            payload,
            agent_id=agent_id,
            refs=refs,
        )
    
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
        
        This is called automatically by the prompt SDK when PromptLoader is used.
        """
        return self.emitter.emit_prompt_resolved(
            prompt_id=prompt_id,
            version_id=version_id,
            resolved_text=resolved_text,
            agent_id=agent_id,
            components=components,
            variables_used=variables_used,
        )


def _create_tracing_components(
    trace_id: str | None = None,
    execution_id: str | None = None,
    output_dir: str | Path | None = None,
    output_file: str = "trace_events.jsonl",
    base_url: str | None = None,
) -> tuple[str, str, EventSink, RawCallbackHandler, EventEmitter]:
    """create sink, handler, and emitter for tracing."""
    tid = trace_id or generate_trace_id()
    eid = execution_id or generate_execution_id()

    if base_url:
        sink: EventSink = HttpSink(base_url=base_url, trace_id=tid)
    elif output_dir:
        output_path = Path(output_dir) / tid / output_file
        sink = FileSink(output_path)
    else:
        sink = ListSink()

    handler = RawCallbackHandler(
        execution_id=eid,
        trace_id=tid,
        event_sink=sink,
    )
    emitter = EventEmitter(
        execution_id=eid,
        trace_id=tid,
        event_sink=sink,
    )
    return tid, eid, sink, handler, emitter


@contextmanager
def enable_tracing(
    trace_id: str | None = None,
    execution_id: str | None = None,
    output_dir: str | Path | None = None,
    output_file: str = "trace_events.jsonl",
    base_url: str | None = None,
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
        base_url: Base URL for the trace API (posts events to server)
    
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

        # send events to server
        with enable_tracing(base_url="http://localhost:8000") as ctx:
            result = graph.invoke(state, config={"callbacks": ctx.callbacks})
    """
    global _active_context
    
    tid, eid, sink, handler, emitter = _create_tracing_components(
        trace_id=trace_id,
        execution_id=execution_id,
        output_dir=output_dir,
        output_file=output_file,
        base_url=base_url,
    )
    
    # create context
    context = TracingContext(
        trace_id=tid,
        execution_id=eid,
        callback_handler=handler,
        event_sink=sink,
        emitter=emitter,
    )
    
    # set global context for prompt SDK
    _active_context = context
    
    try:
        yield context
    finally:
        _active_context = None

