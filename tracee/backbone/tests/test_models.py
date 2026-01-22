"""Tests for model serialization and round-trip."""

import json

import pytest

from backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.models.trace_event import PROMPT_RESOLVED, TraceEvent
from backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_trace_id,
    utc_timestamp,
)


class TestPromptArtifactRoundTrip:
    """Test PromptComponent and PromptVersion serialization."""

    def test_prompt_component_round_trip(self):
        """PromptComponent should serialize and deserialize cleanly."""
        component = PromptComponent(
            type=PromptComponentType.role,
            content="You are a helpful assistant.",
            enabled=True,
        )
        json_str = component.model_dump_json()
        restored = PromptComponent.model_validate_json(json_str)

        assert restored.type == component.type
        assert restored.content == component.content
        assert restored.enabled == component.enabled

    def test_prompt_version_round_trip(self):
        """PromptVersion should serialize and deserialize cleanly."""
        version = PromptVersion(
            prompt_id="test-prompt-001",
            version_id="v1.0.0",
            name="Test Prompt",
            components=[
                PromptComponent(type=PromptComponentType.role, content="Role content"),
                PromptComponent(type=PromptComponentType.goal, content="Goal content"),
            ],
            variables={"var1": "value1"},
            created_at=utc_timestamp(),
        )
        json_str = version.model_dump_json()
        restored = PromptVersion.model_validate_json(json_str)

        assert restored.prompt_id == version.prompt_id
        assert restored.version_id == version.version_id
        assert restored.name == version.name
        assert len(restored.components) == 2
        assert restored.variables == version.variables


class TestTraceEventRoundTrip:
    """Test TraceEvent serialization."""

    def test_raw_event_round_trip(self):
        """Raw LangChain event should serialize and deserialize cleanly."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=0,
            event_type="on_chain_start",
            agent_id=None,
            refs={"langchain": {"run_id": "abc123"}},
            payload={"chain_name": "test_chain", "inputs": {"query": "test"}},
        )
        json_str = event.model_dump_json()
        restored = TraceEvent.model_validate_json(json_str)

        assert restored.event_id == event.event_id
        assert restored.event_type == "on_chain_start"
        assert restored.sequence == 0
        assert restored.refs["langchain"]["run_id"] == "abc123"

    def test_prompt_resolved_event_round_trip(self):
        """prompt_resolved event should serialize and deserialize cleanly."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=1,
            event_type=PROMPT_RESOLVED,
            agent_id="planner",
            refs={},
            payload={
                "prompt_id": "planner-system",
                "version_id": "v1",
                "resolved_text": "You are a planning agent.",
            },
        )
        json_str = event.model_dump_json()
        restored = TraceEvent.model_validate_json(json_str)

        assert restored.event_type == PROMPT_RESOLVED
        assert restored.payload["prompt_id"] == "planner-system"
        assert restored.payload["version_id"] == "v1"

    def test_namespaced_refs_survive_serialization(self):
        """Namespaced refs should survive JSON round-trip."""
        refs = {
            "langchain": {"run_id": "lc-123", "parent_run_id": "lc-000"},
            "langgraph": {"node": "planner", "state_keys": ["messages", "plan"]},
            "hint": {"agent_id": "planner"},
        }
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type="on_chain_start",
            agent_id=None,
            refs=refs,
            payload={},
        )
        json_str = event.model_dump_json()
        restored = TraceEvent.model_validate_json(json_str)

        assert restored.refs["langchain"]["run_id"] == "lc-123"
        assert restored.refs["langgraph"]["state_keys"] == ["messages", "plan"]
        assert restored.refs["hint"]["agent_id"] == "planner"

    def test_agent_id_optional(self):
        """agent_id should be optional (None) for raw events."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type="on_llm_start",
            agent_id=None,
            refs={},
            payload={"model_name": "gpt-4"},
        )
        assert event.agent_id is None
