"""Tests for payload invariants and validation rules."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mas_backbone.models.execution_record import ExecutionRecord, ModelConfig
from mas_backbone.models.trace_event import EventType, TraceEvent
from mas_backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_trace_id,
    utc_timestamp,
)


class TestExecutionRecordInvariants:
    """Test ExecutionRecord validation rules."""

    def test_rejects_empty_resolved_prompt_text(self):
        """ExecutionRecord must reject empty resolved_prompt_text."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionRecord(
                execution_id=generate_execution_id(),
                trace_id=generate_trace_id(),
                origin="playground",
                created_at=utc_timestamp(),
                llm_config=ModelConfig(
                    provider="openai",
                    model_name="gpt-4",
                    temperature=0.0,
                    max_tokens=1000,
                ),
                input_payload={},
                resolved_prompt_text="",  # empty - should fail
            )
        assert "resolved_prompt_text must not be empty" in str(exc_info.value)

    def test_rejects_whitespace_only_resolved_prompt_text(self):
        """ExecutionRecord must reject whitespace-only resolved_prompt_text."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionRecord(
                execution_id=generate_execution_id(),
                trace_id=generate_trace_id(),
                origin="playground",
                created_at=utc_timestamp(),
                llm_config=ModelConfig(
                    provider="openai",
                    model_name="gpt-4",
                    temperature=0.0,
                    max_tokens=1000,
                ),
                input_payload={},
                resolved_prompt_text="   \n\t  ",  # whitespace only
            )
        assert "resolved_prompt_text must not be empty" in str(exc_info.value)

    def test_requires_trace_id(self):
        """ExecutionRecord must have a trace_id (not None)."""
        # pydantic will reject None for a non-optional str field
        with pytest.raises(ValidationError):
            ExecutionRecord(
                execution_id=generate_execution_id(),
                trace_id=None,  # type: ignore
                origin="playground",
                created_at=utc_timestamp(),
                llm_config=ModelConfig(
                    provider="openai",
                    model_name="gpt-4",
                    temperature=0.0,
                    max_tokens=1000,
                ),
                input_payload={},
                resolved_prompt_text="Valid prompt",
            )


class TestAgentMessageInvariants:
    """Test agent_message payload invariants."""

    def _make_event(self, payload: dict) -> TraceEvent:
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=EventType.agent_message,
            agent_id="test",
            refs={},
            payload=payload,
        )

    def test_requires_to_agent_id(self):
        """agent_message must have to_agent_id."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"message_summary": "Hello"})  # missing to_agent_id
        assert "to_agent_id" in str(exc_info.value)

    def test_requires_summary_or_payload_ref(self):
        """agent_message must have message_summary OR payload_ref."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"to_agent_id": "executor"})  # missing both
        assert "message_summary" in str(exc_info.value) or "payload_ref" in str(exc_info.value)

    def test_accepts_message_summary(self):
        """agent_message with to_agent_id and message_summary should pass."""
        event = self._make_event({
            "to_agent_id": "executor",
            "message_summary": "Sending plan",
        })
        assert event.payload["to_agent_id"] == "executor"

    def test_accepts_payload_ref(self):
        """agent_message with to_agent_id and payload_ref should pass."""
        event = self._make_event({
            "to_agent_id": "executor",
            "payload_ref": "blob://storage/12345",
        })
        assert event.payload["payload_ref"] == "blob://storage/12345"

    def test_accepts_both_summary_and_payload_ref(self):
        """agent_message with both summary and payload_ref should pass."""
        event = self._make_event({
            "to_agent_id": "executor",
            "message_summary": "Sending plan",
            "payload_ref": "blob://storage/12345",
        })
        assert event.payload["message_summary"] == "Sending plan"


class TestContractValidationInvariants:
    """Test contract_validation payload invariants."""

    def _make_event(self, payload: dict) -> TraceEvent:
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=EventType.contract_validation,
            agent_id="test",
            refs={},
            payload=payload,
        )

    def test_requires_validation_result(self):
        """contract_validation must have validation_result."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "contract_id": "c1",
                "contract_version": "1.0.0",
            })
        assert "validation_result" in str(exc_info.value)

    def test_requires_is_valid_in_result(self):
        """validation_result must have is_valid."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "contract_id": "c1",
                "contract_version": "1.0.0",
                "validation_result": {"errors": []},  # missing is_valid
            })
        assert "is_valid" in str(exc_info.value)

    def test_requires_errors_in_result(self):
        """validation_result must have errors list."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "contract_id": "c1",
                "contract_version": "1.0.0",
                "validation_result": {"is_valid": True},  # missing errors
            })
        assert "errors" in str(exc_info.value)

    def test_requires_contract_id(self):
        """contract_validation must have contract_id."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "contract_version": "1.0.0",
                "validation_result": {"is_valid": True, "errors": []},
            })
        assert "contract_id" in str(exc_info.value)

    def test_requires_contract_version(self):
        """contract_validation must have contract_version."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "contract_id": "c1",
                "validation_result": {"is_valid": True, "errors": []},
            })
        assert "contract_version" in str(exc_info.value)

    def test_valid_contract_validation(self):
        """Valid contract_validation should pass."""
        event = self._make_event({
            "contract_id": "c1",
            "contract_version": "1.0.0",
            "validation_result": {
                "is_valid": False,
                "errors": [{"path": "$.field", "message": "required"}],
            },
        })
        assert not event.payload["validation_result"]["is_valid"]


class TestToolCallInvariants:
    """Test tool_call payload invariants."""

    def _make_event(self, payload: dict) -> TraceEvent:
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=EventType.tool_call,
            agent_id="test",
            refs={},
            payload=payload,
        )

    def test_requires_tool_name(self):
        """tool_call must have tool_name."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"phase": "start"})
        assert "tool_name" in str(exc_info.value)

    def test_requires_phase(self):
        """tool_call must have phase."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"tool_name": "search"})
        assert "phase" in str(exc_info.value)

    def test_rejects_invalid_phase(self):
        """tool_call must have phase in {start, end}."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"tool_name": "search", "phase": "middle"})
        assert "phase" in str(exc_info.value)

    def test_accepts_start_phase(self):
        """tool_call with phase=start should pass."""
        event = self._make_event({"tool_name": "search", "phase": "start"})
        assert event.payload["phase"] == "start"

    def test_accepts_end_phase(self):
        """tool_call with phase=end should pass."""
        event = self._make_event({"tool_name": "search", "phase": "end"})
        assert event.payload["phase"] == "end"


class TestErrorInvariants:
    """Test error payload invariants."""

    def _make_event(self, payload: dict) -> TraceEvent:
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=EventType.error,
            agent_id="test",
            refs={},
            payload=payload,
        )

    def test_requires_error_type(self):
        """error must have error_type."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"message": "Something went wrong"})
        assert "error_type" in str(exc_info.value)

    def test_requires_message(self):
        """error must have message."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"error_type": "schema"})
        assert "message" in str(exc_info.value)

    def test_rejects_invalid_error_type(self):
        """error must have valid error_type."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({"error_type": "unknown", "message": "test"})
        assert "error_type" in str(exc_info.value)

    def test_accepts_valid_error_types(self):
        """error with valid error_type should pass."""
        valid_types = ["schema", "tool", "model", "infra", "logic"]
        for error_type in valid_types:
            event = self._make_event({"error_type": error_type, "message": "test"})
            assert event.payload["error_type"] == error_type


class TestSequenceOrdering:
    """Test sequence ordering invariants."""

    def test_sequence_can_be_none(self):
        """sequence field can be None."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=None,
            event_type=EventType.agent_input,
            agent_id="test",
            refs={},
            payload={},
        )
        assert event.sequence is None

    def test_sequence_can_be_zero(self):
        """sequence field can be 0."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            sequence=0,
            event_type=EventType.agent_input,
            agent_id="test",
            refs={},
            payload={},
        )
        assert event.sequence == 0


class TestContractFixtureValidation:
    """Test contract fixture validation."""

    def test_sample_contract_is_valid_json(self):
        """Sample contract fixture should be valid JSON."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_contract.json"
        with open(fixture_path) as f:
            contract = json.load(f)

        assert contract["contract_id"] == "plan-input-v1"
        assert contract["contract_version"] == "1.0.0"
        assert "schema" in contract
        assert contract["schema"]["type"] == "object"

    def test_contract_validation_event_references_contract(self):
        """contract_validation event should correctly reference the contract."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_contract.json"
        with open(fixture_path) as f:
            contract = json.load(f)

        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=EventType.contract_validation,
            agent_id="executor",
            refs={"contract": {"contract_id": contract["contract_id"], "version": contract["contract_version"]}},
            payload={
                "contract_id": contract["contract_id"],
                "contract_version": contract["contract_version"],
                "validation_result": {
                    "is_valid": False,
                    "errors": [{"path": "$.steps[0].tool", "message": "missing field"}],
                },
            },
        )

        assert event.payload["contract_id"] == "plan-input-v1"
        assert event.refs["contract"]["version"] == "1.0.0"
