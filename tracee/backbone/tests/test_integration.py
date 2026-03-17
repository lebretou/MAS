"""Integration tests for the backbone workflow."""

import json
from pathlib import Path

from backbone.adapters.sinks import FileSink, ListSink
from backbone.analysis.trace_summary import trace_summary
from backbone.models.trace_event import TraceEvent
from backbone.sdk.tracing import enable_tracing, get_active_context
from backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_span_id,
    generate_trace_id,
    utc_timestamp,
)


class TestEnableTracing:
    """Test the enable_tracing context manager."""

    def test_creates_context_with_generated_ids(self):
        """enable_tracing should generate trace_id and execution_id if not provided."""
        with enable_tracing() as ctx:
            assert ctx.trace_id is not None
            assert ctx.execution_id is not None
            assert len(ctx.trace_id) > 0
            assert len(ctx.execution_id) > 0

    def test_uses_provided_trace_id(self):
        """enable_tracing should use provided trace_id."""
        with enable_tracing(trace_id="my-trace-123") as ctx:
            assert ctx.trace_id == "my-trace-123"

    def test_provides_callbacks_list(self):
        """enable_tracing context should provide callbacks list."""
        with enable_tracing() as ctx:
            assert ctx.callbacks is not None
            assert len(ctx.callbacks) == 1

    def test_sets_active_context(self):
        """enable_tracing should set the active context."""
        assert get_active_context() is None
        
        with enable_tracing() as ctx:
            assert get_active_context() is ctx
        
        assert get_active_context() is None

    def test_nested_context_restores_outer_context(self):
        """Nested tracing should restore the outer context on exit."""
        with enable_tracing() as outer:
            assert get_active_context() is outer

            with enable_tracing() as inner:
                assert get_active_context() is inner

            assert get_active_context() is outer

        assert get_active_context() is None

class TestTraceSummaryWithRawEvents:
    """Test trace summary with raw LangChain events."""

    def _make_raw_event(
        self,
        event_type: str,
        trace_id: str,
        execution_id: str,
        sequence: int,
        payload: dict | None = None,
        refs: dict | None = None,
        span_id: str | None = None,
    ) -> TraceEvent:
        """Create a raw trace event for testing."""
        from backbone.utils.identifiers import generate_event_id, generate_span_id
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=trace_id,
            execution_id=execution_id,
            timestamp=utc_timestamp(),
            sequence=sequence,
            event_type=event_type,
            agent_id=None,
            span_id=span_id or generate_span_id(),
            parent_span_id=None,
            refs=refs or {},
            payload=payload or {},
        )

    def test_summary_extracts_tool_usage(self):
        """TraceSummary should extract tool usage from on_tool_start/end events."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        span_id = "test-span-123"
        
        events = [
            self._make_raw_event(
                "on_tool_start",
                trace_id, execution_id, 0,
                payload={"tool_name": "search", "input": {"query": "test"}},
                span_id=span_id,
            ),
            self._make_raw_event(
                "on_tool_end",
                trace_id, execution_id, 1,
                payload={"output": "result"},
                span_id=span_id,
            ),
        ]
        
        summary = trace_summary(events)
        
        assert len(summary.tool_usage) == 1
        assert summary.tool_usage[0].tool_name == "search"
        assert summary.tool_usage[0].call_count == 1

    def test_summary_extracts_llm_usage(self):
        """TraceSummary should extract LLM usage from on_llm_start/end events."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        span_id = "llm-span-123"
        
        events = [
            self._make_raw_event(
                "on_llm_start",
                trace_id, execution_id, 0,
                payload={"model_name": "gpt-4", "prompts": ["test"]},
                span_id=span_id,
            ),
            self._make_raw_event(
                "on_llm_end",
                trace_id, execution_id, 1,
                payload={"output_text": "hello"},
                span_id=span_id,
            ),
        ]
        
        summary = trace_summary(events)
        
        assert len(summary.llm_usage) == 1
        assert summary.llm_usage[0].tool_name == "gpt-4"
        assert summary.llm_usage[0].call_count == 1

    def test_summary_detects_errors(self):
        """TraceSummary should detect error events."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        
        events = [
            self._make_raw_event(
                "on_chain_error",
                trace_id, execution_id, 0,
                payload={"error_type": "ValueError", "error_message": "invalid input"},
            ),
        ]
        
        summary = trace_summary(events)
        
        assert len(summary.failures) == 1
        assert summary.failures[0]["type"] == "on_chain_error"
        assert summary.failures[0]["error_type"] == "ValueError"

    def test_summary_extracts_agents_from_hints(self):
        """TraceSummary should extract agents from refs.hint.agent_id."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        
        events = [
            self._make_raw_event(
                "on_chain_start",
                trace_id, execution_id, 0,
                refs={"hint": {"agent_id": "planner"}},
            ),
            self._make_raw_event(
                "on_chain_end",
                trace_id, execution_id, 1,
                refs={"hint": {"agent_id": "planner"}},
            ),
            self._make_raw_event(
                "on_chain_start",
                trace_id, execution_id, 2,
                refs={"hint": {"agent_id": "executor"}},
            ),
        ]
        
        summary = trace_summary(events)
        
        assert "planner" in summary.agents
        assert "executor" in summary.agents

    def test_summary_extracts_agents_from_langgraph_node(self):
        """TraceSummary should extract agents from refs.langgraph.node."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        
        events = [
            self._make_raw_event(
                "on_chain_start",
                trace_id, execution_id, 0,
                refs={"langgraph": {"node": "interaction"}},
            ),
        ]
        
        summary = trace_summary(events)
        
        assert "interaction" in summary.agents

    def test_summary_infers_agent_transitions(self):
        """TraceSummary should infer agent transitions from event sequences."""
        trace_id = generate_trace_id()
        execution_id = generate_execution_id()
        
        events = [
            self._make_raw_event(
                "on_chain_start",
                trace_id, execution_id, 0,
                refs={"hint": {"agent_id": "planner"}},
            ),
            self._make_raw_event(
                "on_chain_end",
                trace_id, execution_id, 1,
                refs={"hint": {"agent_id": "planner"}},
            ),
            self._make_raw_event(
                "on_chain_start",
                trace_id, execution_id, 2,
                refs={"hint": {"agent_id": "executor"}},
            ),
        ]
        
        summary = trace_summary(events)
        
        # should have an edge from planner to executor
        assert len(summary.edges) == 1
        assert summary.edges[0].from_agent == "planner"
        assert summary.edges[0].to_agent == "executor"


class TestFileSink:
    """Test FileSink writes events correctly."""

    def setup_method(self):
        """Set up test output directory."""
        self.output_dir = Path(__file__).parent / "test_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_writes_events_to_file(self):
        """FileSink should write events to a JSONL file."""
        trace_id = generate_trace_id()
        output_path = self.output_dir / trace_id / "trace_events.jsonl"
        
        sink = FileSink(output_path)

        sink.append(TraceEvent(
            event_id=generate_event_id(),
            trace_id=trace_id,
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=0,
            event_type="on_chain_start",
            agent_id=None,
            span_id=generate_span_id(),
            parent_span_id=None,
            refs={},
            payload={"chain_name": "test"},
        ))
        sink.append(TraceEvent(
            event_id=generate_event_id(),
            trace_id=trace_id,
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=1,
            event_type="on_chain_end",
            agent_id=None,
            span_id=generate_span_id(),
            parent_span_id=None,
            refs={},
            payload={"outputs": {}},
        ))
        
        # read the file
        with open(output_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        event1 = json.loads(lines[0])
        assert event1["event_type"] == "on_chain_start"
        
        event2 = json.loads(lines[1])
        assert event2["event_type"] == "on_chain_end"
