"""Tests for output_schema feature on PromptVersion.

TDD: these tests are written before the implementation.
Covers model serialization, backward compatibility, SDK integration,
and playground structured output plumbing.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backbone.models.prompt_artifact import (
    Prompt,
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.utils.identifiers import utc_timestamp


SAMPLE_SCHEMA = {
    "type": "object",
    "title": "AnalysisPlan",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "number",
        },
    },
    "required": ["steps", "confidence"],
}


def _make_version(output_schema: dict | None = None) -> PromptVersion:
    """Helper to build a PromptVersion with optional output_schema."""
    return PromptVersion(
        prompt_id="test-prompt",
        version_id="v1",
        name="Test Version",
        components=[
            PromptComponent(type=PromptComponentType.role, content="You are a planner."),
            PromptComponent(type=PromptComponentType.task, content="Create a plan."),
        ],
        variables={"max_steps": "5"},
        output_schema=output_schema,
        created_at=utc_timestamp(),
    )


class TestPromptVersionOutputSchema:
    """Test output_schema field on PromptVersion."""

    def test_output_schema_defaults_to_none(self):
        """output_schema should default to None when not provided."""
        version = PromptVersion(
            prompt_id="p1",
            version_id="v1",
            name="V1",
            components=[],
            created_at=utc_timestamp(),
        )
        assert version.output_schema is None

    def test_output_schema_stores_json_schema(self):
        """output_schema should store a JSON Schema dict."""
        version = _make_version(output_schema=SAMPLE_SCHEMA)
        assert version.output_schema == SAMPLE_SCHEMA
        assert version.output_schema["type"] == "object"
        assert "steps" in version.output_schema["properties"]

    def test_round_trip_with_output_schema(self):
        """PromptVersion with output_schema should serialize and deserialize cleanly."""
        version = _make_version(output_schema=SAMPLE_SCHEMA)
        json_str = version.model_dump_json()
        restored = PromptVersion.model_validate_json(json_str)

        assert restored.output_schema == SAMPLE_SCHEMA
        assert restored.prompt_id == version.prompt_id
        assert restored.version_id == version.version_id
        assert len(restored.components) == 2

    def test_round_trip_without_output_schema(self):
        """PromptVersion without output_schema should serialize with null."""
        version = _make_version(output_schema=None)
        json_str = version.model_dump_json()
        restored = PromptVersion.model_validate_json(json_str)

        assert restored.output_schema is None

    def test_backward_compat_missing_key(self):
        """Old JSON without output_schema key should deserialize to None."""
        raw = {
            "prompt_id": "p1",
            "version_id": "v1",
            "name": "V1",
            "components": [],
            "variables": None,
            "created_at": utc_timestamp(),
        }
        version = PromptVersion.model_validate(raw)
        assert version.output_schema is None

    def test_resolve_unaffected_by_output_schema(self):
        """resolve() should still concatenate enabled components regardless of schema."""
        version = _make_version(output_schema=SAMPLE_SCHEMA)
        resolved = version.resolve()
        assert "You are a planner." in resolved
        assert "Create a plan." in resolved

    def test_model_dump_includes_output_schema(self):
        """model_dump() should include output_schema in the dict."""
        version = _make_version(output_schema=SAMPLE_SCHEMA)
        data = version.model_dump()
        assert "output_schema" in data
        assert data["output_schema"]["title"] == "AnalysisPlan"

    def test_complex_nested_schema(self):
        """output_schema should support deeply nested JSON Schema."""
        nested_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "metric": {"type": "string"},
                                    "value": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        }
        version = _make_version(output_schema=nested_schema)
        json_str = version.model_dump_json()
        restored = PromptVersion.model_validate_json(json_str)
        assert restored.output_schema["properties"]["analysis"]["properties"]["findings"]["type"] == "array"


class TestPlaygroundRunOutputSchemaFlag:
    """Test output_schema_used flag on PlaygroundRun."""

    def test_output_schema_used_defaults_false(self):
        """output_schema_used should default to False."""
        run = PlaygroundRun(
            run_id="run-1",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4o",
            provider="openai",
            temperature=0.7,
            input_variables={},
            resolved_prompt="You are a test agent.",
            output="Hello!",
        )
        assert run.output_schema_used is False

    def test_output_schema_used_true(self):
        """output_schema_used can be set to True."""
        run = PlaygroundRun(
            run_id="run-2",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4o",
            provider="openai",
            temperature=0.0,
            input_variables={},
            resolved_prompt="You are a test agent.",
            output='{"steps": ["step1"], "confidence": 0.95}',
            output_schema_used=True,
        )
        assert run.output_schema_used is True

    def test_round_trip_with_flag(self):
        """PlaygroundRun should serialize/deserialize output_schema_used."""
        run = PlaygroundRun(
            run_id="run-3",
            created_at=utc_timestamp(),
            prompt_id="p1",
            version_id="v1",
            model="gpt-4o",
            provider="openai",
            temperature=0.7,
            input_variables={},
            resolved_prompt="test",
            output="test",
            output_schema_used=True,
        )
        json_str = run.model_dump_json()
        restored = PlaygroundRun.model_validate_json(json_str)
        assert restored.output_schema_used is True


class TestPromptLoaderGetWithSchema:
    """Test PromptLoader.get_with_schema() method."""

    def test_get_with_schema_returns_tuple(self):
        """get_with_schema should return (resolved_text, output_schema)."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)

        loader = PromptLoader(base_url="http://localhost:8000")
        # mock _fetch_latest to avoid network call
        loader._fetch_latest = MagicMock(return_value=version)

        text, schema = loader.get_with_schema("test-prompt")
        assert "You are a planner." in text
        assert schema == SAMPLE_SCHEMA

    def test_get_with_schema_none_when_no_schema(self):
        """get_with_schema should return None for schema when not set."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=None)

        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)

        text, schema = loader.get_with_schema("test-prompt")
        assert "You are a planner." in text
        assert schema is None

    def test_get_with_schema_stays_pure_during_tracing(self):
        """get_with_schema should not emit trace events when tracing is active."""
        from backbone.sdk.prompt_loader import PromptLoader
        from backbone.sdk.tracing import enable_tracing

        version = _make_version(output_schema=SAMPLE_SCHEMA)

        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock()

        with enable_tracing() as ctx:
            text, schema = loader.get_with_schema("test-prompt", agent_id="planner")

        assert "You are a planner." in text
        assert schema == SAMPLE_SCHEMA
        assert ctx.event_sink.events == []

    def test_get_registers_prompt_with_agent_registry(self):
        """get should sync prompt ownership when agent_id is provided."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock()

        text = loader.get("test-prompt", agent_id="planner")

        assert "You are a planner." in text
        loader._upsert_agent_registry.assert_called_once_with(
            agent_id="planner",
            prompt_id="test-prompt",
            prompt_version_id="v1",
        )

    def test_get_with_schema_registers_prompt_with_agent_registry(self):
        """get_with_schema should sync prompt ownership when agent_id is provided."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock()

        text, schema = loader.get_with_schema("test-prompt", agent_id="planner")

        assert "You are a planner." in text
        assert schema == SAMPLE_SCHEMA
        loader._upsert_agent_registry.assert_called_once_with(
            agent_id="planner",
            prompt_id="test-prompt",
            prompt_version_id="v1",
        )

    def test_get_without_agent_id_does_not_register_prompt(self):
        """get should not sync registry when no agent_id is provided."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock()

        loader.get("test-prompt")

        loader._upsert_agent_registry.assert_not_called()

    def test_get_skips_registry_sync_when_agent_prompt_is_unchanged(self):
        """get should avoid repeated registry writes for the same resolved version."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock()

        loader.get("test-prompt", agent_id="planner")
        loader.get("test-prompt", agent_id="planner")

        loader._upsert_agent_registry.assert_called_once_with(
            agent_id="planner",
            prompt_id="test-prompt",
            prompt_version_id="v1",
        )

    def test_get_retries_registry_sync_after_failed_upsert(self):
        """failed registry sync should not mark the agent prompt as synced."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock(side_effect=[False, True])

        loader.get("test-prompt", agent_id="planner")
        loader.get("test-prompt", agent_id="planner")

        assert loader._upsert_agent_registry.call_count == 2

    def test_clear_cache_resets_agent_prompt_sync_cache(self):
        """clear_cache should allow registry sync to run again."""
        from backbone.sdk.prompt_loader import PromptLoader

        version = _make_version(output_schema=SAMPLE_SCHEMA)
        loader = PromptLoader(base_url="http://localhost:8000")
        loader._fetch_latest = MagicMock(return_value=version)
        loader._upsert_agent_registry = MagicMock(return_value=True)

        loader.get("test-prompt", agent_id="planner")
        loader.clear_cache()
        loader.get("test-prompt", agent_id="planner")

        assert loader._upsert_agent_registry.call_count == 2

    def test_upsert_agent_registry_returns_false_on_request_error(self, monkeypatch):
        """registry sync should report failure on transport errors."""
        from backbone.sdk.prompt_loader import PromptLoader

        loader = PromptLoader(base_url="http://localhost:8000")

        class FailingClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def put(self, *args, **kwargs):
                raise httpx.RequestError("boom")

        monkeypatch.setattr("backbone.sdk.prompt_loader.httpx.Client", FailingClient)

        with pytest.warns(UserWarning, match="failed to sync agent registry"):
            assert loader._upsert_agent_registry("planner", "test-prompt", "v1") is False


class TestOpenAIStructuredOutput:
    """Test OpenAI response_format construction for structured output."""

    def test_builds_response_format_from_schema(self):
        """_call_openai should add response_format when output_schema is provided."""
        from server.playground_routes import _build_openai_response_format

        result = _build_openai_response_format(SAMPLE_SCHEMA)

        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "AnalysisPlan"
        assert result["json_schema"]["schema"] == SAMPLE_SCHEMA
        assert result["json_schema"]["strict"] is True

    def test_uses_fallback_name_when_no_title(self):
        """Should use 'output' as name when schema has no title."""
        from server.playground_routes import _build_openai_response_format

        schema_no_title = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = _build_openai_response_format(schema_no_title)

        assert result["json_schema"]["name"] == "output"


class TestAnthropicStructuredOutput:
    """Test Anthropic tool-use extraction for structured output."""

    def test_builds_tool_definition_from_schema(self):
        """_build_anthropic_schema_tool should create a tool from schema."""
        from server.playground_routes import _build_anthropic_schema_tool

        tool = _build_anthropic_schema_tool(SAMPLE_SCHEMA)

        assert tool["name"] == "structured_output"
        assert tool["input_schema"] == SAMPLE_SCHEMA

    def test_extracts_json_from_tool_use_block(self):
        """_extract_anthropic_structured_content should pull args from tool_use block."""
        from server.playground_routes import _extract_anthropic_structured_content

        # simulate anthropic response content blocks
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.input = {"steps": ["step1"], "confidence": 0.9}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here is the result."

        content = _extract_anthropic_structured_content([tool_use_block, text_block])
        parsed = json.loads(content)
        assert parsed["steps"] == ["step1"]
        assert parsed["confidence"] == 0.9
