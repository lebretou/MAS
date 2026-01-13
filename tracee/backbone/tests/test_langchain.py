"""Tests for LangChain callback handler."""

from uuid import uuid4

import pytest

from backbone.adapters.event_api import ListSink
from backbone.adapters.langchain_callback import MASCallbackHandler, _classify_error
from backbone.models.trace_event import EventType
from backbone.utils.identifiers import generate_execution_id, generate_trace_id


class TestErrorClassification:
    """Test error classification logic."""

    def test_classifies_openai_errors_as_model(self):
        """OpenAI-related errors should be classified as 'model'."""
        class OpenAIError(Exception):
            pass

        assert _classify_error(OpenAIError("rate limit")) == "model"

    def test_classifies_timeout_errors_as_infra(self):
        """Timeout errors should be classified as 'infra'."""
        class TimeoutError(Exception):
            pass

        assert _classify_error(TimeoutError("connection timeout")) == "infra"

    def test_classifies_validation_errors_as_schema(self):
        """Validation errors should be classified as 'schema'."""
        class ValidationError(Exception):
            pass

        assert _classify_error(ValidationError("invalid input")) == "schema"

    def test_classifies_tool_errors_as_tool(self):
        """Tool-related errors should be classified as 'tool'."""
        class ToolExecutionError(Exception):
            pass

        assert _classify_error(ToolExecutionError("tool failed")) == "tool"

    def test_defaults_to_logic(self):
        """Unknown errors should default to 'logic'."""
        class UnknownError(Exception):
            pass

        assert _classify_error(UnknownError("something")) == "logic"


class TestMASCallbackHandler:
    """Test MASCallbackHandler event emission."""

    def setup_method(self):
        """Set up test fixtures."""
        self.execution_id = generate_execution_id()
        self.trace_id = generate_trace_id()
        self.sink = ListSink()
        self.handler = MASCallbackHandler(
            execution_id=self.execution_id,
            trace_id=self.trace_id,
            event_sink=self.sink,
            default_agent_id="test_agent",
        )

    def test_on_llm_start_emits_tool_call_start(self):
        """on_llm_start should emit tool_call with phase=start."""
        run_id = uuid4()
        self.handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4"}},
            prompts=["Hello, world!"],
            run_id=run_id,
        )

        assert len(self.sink.events) == 1
        event = self.sink.events[0]
        assert event.event_type == EventType.tool_call
        assert event.payload["tool_name"] == "llm.generate"
        assert event.payload["phase"] == "start"
        assert event.refs.get("llm", {}).get("model") == "gpt-4"

    def test_on_llm_end_emits_tool_call_end(self):
        """on_llm_end should emit tool_call with phase=end."""
        from langchain_core.outputs import LLMResult, Generation

        run_id = uuid4()

        # simulate start first
        self.handler.on_llm_start(
            serialized={},
            prompts=["test"],
            run_id=run_id,
        )

        # now end
        result = LLMResult(generations=[[Generation(text="Hello!")]])
        self.handler.on_llm_end(response=result, run_id=run_id)

        assert len(self.sink.events) == 2
        end_event = self.sink.events[1]
        assert end_event.event_type == EventType.tool_call
        assert end_event.payload["phase"] == "end"
        assert "text" in end_event.payload.get("output", {})

    def test_on_tool_start_emits_tool_call_start(self):
        """on_tool_start should emit tool_call with phase=start."""
        run_id = uuid4()
        self.handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="query string",
            run_id=run_id,
        )

        assert len(self.sink.events) == 1
        event = self.sink.events[0]
        assert event.event_type == EventType.tool_call
        assert event.payload["tool_name"] == "search_tool"
        assert event.payload["phase"] == "start"

    def test_on_tool_end_emits_tool_call_end(self):
        """on_tool_end should emit tool_call with phase=end."""
        run_id = uuid4()

        # start first
        self.handler.on_tool_start(
            serialized={"name": "search"},
            input_str="test",
            run_id=run_id,
        )

        # end
        self.handler.on_tool_end(output="search result", run_id=run_id)

        assert len(self.sink.events) == 2
        end_event = self.sink.events[1]
        assert end_event.event_type == EventType.tool_call
        assert end_event.payload["phase"] == "end"

    def test_on_chain_error_emits_error_event(self):
        """on_chain_error should emit error event with classification."""
        run_id = uuid4()

        class TestValidationError(Exception):
            pass

        self.handler.on_chain_error(
            error=TestValidationError("invalid schema"),
            run_id=run_id,
        )

        assert len(self.sink.events) == 1
        event = self.sink.events[0]
        assert event.event_type == EventType.error
        assert event.payload["error_type"] == "schema"
        assert "invalid schema" in event.payload["message"]

    def test_stores_langchain_run_id_in_refs(self):
        """LangChain run_id should be stored in refs['langchain']['run_id']."""
        run_id = uuid4()
        parent_run_id = uuid4()

        self.handler.on_chain_start(
            serialized={"name": "test_chain"},
            inputs={"query": "test"},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        event = self.sink.events[0]
        assert event.refs["langchain"]["run_id"] == str(run_id)
        assert event.refs["langchain"]["parent_run_id"] == str(parent_run_id)

    def test_does_not_emit_agent_message(self):
        """Callback handler should NOT emit agent_message (that's manual API)."""
        run_id = uuid4()

        # simulate a full chain execution
        self.handler.on_chain_start(
            serialized={"name": "chain"},
            inputs={},
            run_id=run_id,
        )
        self.handler.on_chain_end(outputs={}, run_id=run_id)

        # check no agent_message events were emitted
        message_events = [e for e in self.sink.events if e.event_type == EventType.agent_message]
        assert len(message_events) == 0

    def test_uses_agent_id_from_metadata(self):
        """Should use agent_id from metadata if provided."""
        run_id = uuid4()

        self.handler.on_chain_start(
            serialized={"name": "chain"},
            inputs={},
            run_id=run_id,
            metadata={"agent_id": "planner"},
        )

        event = self.sink.events[0]
        assert event.agent_id == "planner"

    def test_stores_langgraph_context_in_refs(self):
        """LangGraph metadata should be stored in refs['langgraph']."""
        run_id = uuid4()

        self.handler.on_chain_start(
            serialized={"name": "chain"},
            inputs={},
            run_id=run_id,
            metadata={
                "agent_id": "planner",
                "langgraph_node": "plan_node",
                "langgraph_state_keys": ["messages", "plan"],
            },
        )

        event = self.sink.events[0]
        assert event.refs["langgraph"]["node"] == "plan_node"
        assert event.refs["langgraph"]["state_keys"] == ["messages", "plan"]

    def test_all_events_share_execution_and_trace_id(self):
        """All emitted events should share the same execution_id and trace_id."""
        run_id = uuid4()

        self.handler.on_chain_start(serialized={}, inputs={}, run_id=run_id)
        self.handler.on_llm_start(serialized={}, prompts=["test"], run_id=uuid4(), parent_run_id=run_id)
        self.handler.on_chain_end(outputs={}, run_id=run_id)

        for event in self.sink.events:
            assert event.execution_id == self.execution_id
            assert event.trace_id == self.trace_id
