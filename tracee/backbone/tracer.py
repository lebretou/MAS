"""Convenience wrapper for MAS tracing.

Provides a simple interface to set up tracing with minimal boilerplate.
"""

from pathlib import Path

from backbone.adapters.event_api import EventEmitter, EventSink, FileSink, ListSink
from backbone.adapters.langchain_callback import MASCallbackHandler
from backbone.utils.identifiers import generate_execution_id, generate_trace_id


class Tracer:
    """High-level tracing interface for LangChain/LangGraph integration.
    
    Usage:
        from backbone import Tracer
        
        tracer = Tracer(output_dir="./traces")
        graph.invoke(input, config={"callbacks": [tracer.callback]})
        
        # manual events
        tracer.emitter.emit_message("planner", "coder", summary="Here's the plan")
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        default_agent_id: str = "workflow",
    ) -> None:
        """Initialize a new tracer.
        
        Args:
            output_dir: directory to store trace files. If None, traces are kept in memory.
            default_agent_id: default agent ID for events without explicit agent context.
        """
        self.trace_id = generate_trace_id()
        self.execution_id = generate_execution_id()
        self._default_agent_id = default_agent_id

        if output_dir:
            self._output_path = Path(output_dir) / self.trace_id
            self._output_path.mkdir(parents=True, exist_ok=True)
            self._sink: EventSink = FileSink(self._output_path / "trace_events.jsonl")
        else:
            self._output_path = None
            self._sink = ListSink()

        self._emitter = EventEmitter(self.execution_id, self.trace_id, self._sink)
        self._callback = MASCallbackHandler(
            execution_id=self.execution_id,
            trace_id=self.trace_id,
            event_sink=self._sink,
            default_agent_id=default_agent_id,
        )

    @property
    def callback(self) -> MASCallbackHandler:
        """LangChain callback handler for automatic event capture."""
        return self._callback

    @property
    def emitter(self) -> EventEmitter:
        """Event emitter for manual event emission."""
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
