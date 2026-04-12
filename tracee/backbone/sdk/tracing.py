"""Tracing SDK -  provides a simple way to enable tracing for LangChain/LangGraph
applications.

Example (so we do this in the config file of langgraph):
    from tracee.backbone.sdk import enable_tracing
    
    with enable_tracing() as ctx:
        result = graph.invoke(state, config={"callbacks": ctx.callbacks})
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from backbone.adapters.langchain_callback import RawCallbackHandler
from backbone.adapters.sinks import EventSink, FileSink, HttpSink, ListSink
from backbone.utils.identifiers import (
    generate_execution_id,
    generate_trace_id,
)

_active_context: ContextVar[TracingContext | None] = ContextVar(
    "tracee_active_context",
    default=None,
)


def get_active_context() -> TracingContext | None:
    """Get the currently active tracing context, if any."""
    return _active_context.get()


@dataclass
class TracingContext:
    """Context object returned by enable_tracing().
    """
    
    trace_id: str
    execution_id: str
    callback_handler: RawCallbackHandler
    event_sink: EventSink
    
    @property
    def callbacks(self) -> list:
        """Get the list of callbacks to pass to LangChain.
        
        Usage:
            graph.invoke(state, config={"callbacks": ctx.callbacks})
        """
        return [self.callback_handler]


def _create_tracing_components(
    trace_id: str | None = None,
    execution_id: str | None = None,
    output_dir: str | Path | None = None,
    output_file: str = "trace_events.jsonl",
    base_url: str | None = None,
    graph_id: str | None = None,
) -> tuple[str, str, EventSink, RawCallbackHandler]:
    """create sink and handler for tracing."""
    tid = trace_id or generate_trace_id()
    eid = execution_id or generate_execution_id()

    if base_url:
        sink: EventSink = HttpSink(base_url=base_url, trace_id=tid, graph_id=graph_id)
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
    return tid, eid, sink, handler


@contextmanager
def enable_tracing(
    trace_id: str | None = None,
    execution_id: str | None = None,
    output_dir: str | Path | None = None,
    output_file: str = "trace_events.jsonl",
    base_url: str | None = None,
    graph_id: str | None = None,
) -> Generator[TracingContext, None, None]:
    """Enable tracing for LangChain/LangGraph code.
    
    This is the single entry point for enabling tracing. It creates the sink and
    callback handler and provides a context object for accessing callbacks.
    
    Args:
        trace_id: Optional trace ID (auto-generated if not provided)
        execution_id: Optional execution ID (auto-generated if not provided)
        output_dir: Directory to write trace files to (creates {trace_id}/ subdirectory)
                   If not provided, traces are stored in memory only.
        output_file: Name of the trace events file (default: trace_events.jsonl)
        base_url: Base URL for the trace API (posts events to server)
        graph_id: Optional graph ID to associate traces with a specific graph
    
    Yields:
        TracingContext with callbacks
    
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
    tid, eid, sink, handler = _create_tracing_components(
        trace_id=trace_id,
        execution_id=execution_id,
        output_dir=output_dir,
        output_file=output_file,
        base_url=base_url,
        graph_id=graph_id,
    )
    
    # create context
    context = TracingContext(
        trace_id=tid,
        execution_id=eid,
        callback_handler=handler,
        event_sink=sink,
    )
    
    token = _active_context.set(context)
    
    try:
        yield context
    finally:
        _active_context.reset(token)

