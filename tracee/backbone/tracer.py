"""Convenience wrapper for MAS tracing.

Provides a simple interface to set up tracing with minimal boilerplate.

For new code, prefer using the `enable_tracing()` context manager from
`backbone.sdk.tracing` for a cleaner interface.
"""

from pathlib import Path

from backbone.adapters.event_api import EventEmitter
from backbone.adapters.langchain_callback import RawCallbackHandler
from backbone.adapters.sinks import EventSink, ListSink
from backbone.sdk.tracing import _create_tracing_components


class Tracer:
    """High-level tracing interface for LangChain/LangGraph integration.
    
    Usage:
        from backbone import Tracer
        
        tracer = Tracer(output_dir="./traces")
        graph.invoke(input, config={"callbacks": [tracer.callback]})
    
    For new code, prefer the `enable_tracing()` context manager:
        
        from backbone.sdk import enable_tracing
        
        with enable_tracing(output_dir="./traces") as ctx:
            graph.invoke(input, config={"callbacks": ctx.callbacks})
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize a new tracer.
        
        Args:
            output_dir: directory to store trace files. If None, traces are kept in memory.
        """
        tid, eid, sink, handler, emitter = _create_tracing_components(
            output_dir=output_dir,
        )
        self.trace_id = tid
        self.execution_id = eid
        self._sink = sink
        if output_dir:
            self._output_path = Path(output_dir) / self.trace_id
        else:
            self._output_path = None
        self._callback = handler
        self._emitter = emitter

    @property
    def callback(self) -> RawCallbackHandler:
        """LangChain callback handler for automatic event capture."""
        return self._callback

    @property
    def emitter(self) -> EventEmitter:
        """Event emitter for manual event emission (primarily prompt_resolved)."""
        return self._emitter

    @property
    def sink(self) -> EventSink:
        """The underlying event sink (ListSink or FileSink)."""
        return self._sink

    @property
    def output_path(self) -> Path | None:
        """Path to trace output directory, or None if in-memory."""
        return self._output_path

    @property
    def events(self) -> list:
        """Get events if using in-memory sink (ListSink).
        
        Raises:
            TypeError: if using FileSink (events are written to disk).
        """
        if isinstance(self._sink, ListSink):
            return self._sink.events
        raise TypeError("events property only available with in-memory sink (no output_dir)")

    def __repr__(self) -> str:
        sink_type = "file" if self._output_path else "memory"
        return f"Tracer(trace_id={self.trace_id!r}, sink={sink_type})"
