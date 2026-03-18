"""Tests for model serialization and round-trip."""

import json

import pytest

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptTool,
    PromptToolArgument,
    PromptVersion,
)
from backbone.models.trace_event import TraceEvent
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
                PromptComponent(type=PromptComponentType.task, content="Task content"),
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


class TestOutputSchema:
    """Tests for PromptVersion.output_schema and resolve() injection."""

    def _make_version(self, output_schema=None):
        return PromptVersion(
            prompt_id="test-prompt",
            version_id="v1",
            name="Test",
            components=[
                PromptComponent(type=PromptComponentType.role, content="You are a helpful assistant."),
                PromptComponent(type=PromptComponentType.task, content="Answer the question."),
            ],
            created_at=utc_timestamp(),
            output_schema=output_schema,
        )

    def test_resolve_without_schema_is_unchanged(self):
        """resolve() with no output_schema returns the same plain text as before."""
        version = self._make_version()
        assert version.resolve() == "You are a helpful assistant.\n\nAnswer the question."

    def test_resolve_with_schema_appends_block(self):
        """resolve() appends a JSON Schema block after components when output_schema is set."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "age":  {"type": "number"},
            },
            "required": ["name"],
        }
        version = self._make_version(output_schema=schema)
        resolved = version.resolve()

        assert resolved.startswith("You are a helpful assistant.\n\nAnswer the question.\n\n")
        assert "Respond with a JSON object that conforms to the following JSON Schema:" in resolved
        assert "```json" in resolved
        assert '"type": "object"' in resolved
        assert '"name"' in resolved

    def test_invalid_schema_missing_type_raises(self):
        """output_schema missing 'type' raises ValueError at construction."""
        with pytest.raises(ValueError, match="output_schema must have top-level"):
            self._make_version(output_schema={"properties": {"name": {"type": "string"}}})

    def test_invalid_schema_missing_properties_raises(self):
        """output_schema missing 'properties' raises ValueError at construction."""
        with pytest.raises(ValueError, match="output_schema must have top-level"):
            self._make_version(output_schema={"type": "object"})

    def test_model_validate_with_null_schema(self):
        """model_validate with output_schema: null (existing payloads) works without error."""
        data = {
            "prompt_id": "p1",
            "version_id": "v1",
            "name": "Test",
            "components": [{"type": "role", "content": "Hello", "enabled": True}],
            "output_schema": None,
            "created_at": utc_timestamp(),
        }
        version = PromptVersion.model_validate(data)
        assert version.output_schema is None

    def test_model_validate_with_valid_schema(self):
        """model_validate with a valid output_schema dict deserializes correctly."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        data = {
            "prompt_id": "p1",
            "version_id": "v1",
            "name": "Test",
            "components": [{"type": "role", "content": "Hello", "enabled": True}],
            "output_schema": schema,
            "created_at": utc_timestamp(),
        }
        version = PromptVersion.model_validate(data)
        assert version.output_schema == schema

    def test_round_trip_with_schema(self):
        """PromptVersion with output_schema serializes and deserializes cleanly."""
        schema = {"type": "object", "properties": {"result": {"type": "boolean"}}}
        version = self._make_version(output_schema=schema)
        restored = PromptVersion.model_validate_json(version.model_dump_json())
        assert restored.output_schema == schema

    def test_round_trip_with_tools(self):
        """PromptVersion with custom tools serializes and deserializes cleanly."""
        version = self._make_version()
        version.tools = [
            PromptTool(
                name="explore_dataset",
                description="return basic information about a dataset",
                arguments=[
                    PromptToolArgument(
                        name="dataset_name",
                        description="name of the dataset to inspect",
                        required=True,
                    )
                ],
            )
        ]
        restored = PromptVersion.model_validate_json(version.model_dump_json())
        assert restored.tools[0].name == "explore_dataset"
        assert restored.tools[0].arguments[0].name == "dataset_name"

    def test_duplicate_tool_names_raise(self):
        """PromptVersion rejects duplicate tool names."""
        with pytest.raises(ValueError, match="tool names must be unique"):
            PromptVersion(
                prompt_id="test-prompt",
                version_id="v1",
                name="Test",
                components=[
                    PromptComponent(type=PromptComponentType.role, content="You are a helpful assistant."),
                ],
                tools=[
                    PromptTool(name="lookup", description="first tool"),
                    PromptTool(name="lookup", description="second tool"),
                ],
                created_at=utc_timestamp(),
            )

    def test_reserved_tool_name_raises(self):
        """PromptVersion rejects reserved internal tool names."""
        with pytest.raises(ValueError, match="reserved"):
            PromptVersion(
                prompt_id="test-prompt",
                version_id="v1",
                name="Test",
                components=[
                    PromptComponent(type=PromptComponentType.role, content="You are a helpful assistant."),
                ],
                tools=[
                    PromptTool(name="structured_output", description="conflicts with internal tool"),
                ],
                created_at=utc_timestamp(),
            )

    def test_playground_run_accepts_output_schema(self):
        """PlaygroundRun accepts output_schema field without extra-field rejection."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        run = PlaygroundRun(
            run_id="run-1",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4",
            provider="openai",
            input_variables={},
            resolved_prompt="Hello",
            output_schema=schema,
            output="{}",
        )
        assert run.output_schema == schema

    def test_playground_run_output_schema_defaults_none(self):
        """PlaygroundRun.output_schema defaults to None for existing records."""
        run = PlaygroundRun(
            run_id="run-2",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4",
            provider="openai",
            input_variables={},
            resolved_prompt="Hello",
            output="{}",
        )
        assert run.output_schema is None

    def test_playground_run_create_accepts_output_schema(self):
        """PlaygroundRunCreate accepts output_schema in the request body."""
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        req = PlaygroundRunCreate(prompt_id="p1", output_schema=schema)
        assert req.output_schema == schema

    def test_playground_run_accepts_tool_calls(self):
        """PlaygroundRun accepts persisted tool call metadata."""
        run = PlaygroundRun(
            run_id="run-3",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4o",
            provider="openai",
            input_variables={},
            resolved_prompt="Hello",
            tools=[
                PromptTool(
                    name="explore_dataset",
                    description="return basic information about a dataset",
                )
            ],
            tool_calls=[
                {
                    "call_id": "call-1",
                    "name": "explore_dataset",
                    "arguments": {"dataset_name": "sales"},
                }
            ],
            output='[{"name": "explore_dataset"}]',
        )
        assert run.tools[0].name == "explore_dataset"
        assert run.tool_calls[0].name == "explore_dataset"


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
