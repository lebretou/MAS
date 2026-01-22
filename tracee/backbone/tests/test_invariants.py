"""Tests for payload invariants and validation rules."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backbone.models.trace_event import PROMPT_RESOLVED, TraceEvent
from backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_trace_id,
    utc_timestamp,
)


class TestPromptResolvedInvariants:
    """Test prompt_resolved payload invariants.
    
    This is the only custom event type that has validation rules.
    Raw LangChain events pass through without payload validation.
    """

    def _make_event(self, payload: dict) -> TraceEvent:
        return TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type=PROMPT_RESOLVED,
            agent_id="test",
            refs={},
            payload=payload,
        )

    def test_requires_prompt_id(self):
        """prompt_resolved must have prompt_id."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "version_id": "v1",
                "resolved_text": "You are an agent.",
            })
        assert "prompt_id" in str(exc_info.value)

    def test_requires_version_id(self):
        """prompt_resolved must have version_id."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "prompt_id": "planner-prompt",
                "resolved_text": "You are an agent.",
            })
        assert "version_id" in str(exc_info.value)

    def test_requires_resolved_text(self):
        """prompt_resolved must have resolved_text."""
        with pytest.raises(ValidationError) as exc_info:
            self._make_event({
                "prompt_id": "planner-prompt",
                "version_id": "v1",
            })
        assert "resolved_text" in str(exc_info.value)

    def test_valid_prompt_resolved(self):
        """Valid prompt_resolved event should pass."""
        event = self._make_event({
            "prompt_id": "planner-prompt",
            "version_id": "v1",
            "resolved_text": "You are an expert planning agent.\n\nCreate step-by-step plans.",
        })
        assert event.payload["prompt_id"] == "planner-prompt"
        assert event.payload["version_id"] == "v1"
        assert "planning agent" in event.payload["resolved_text"]

    def test_accepts_optional_components(self):
        """prompt_resolved can include components list."""
        event = self._make_event({
            "prompt_id": "planner-prompt",
            "version_id": "v1",
            "resolved_text": "You are an agent.",
            "components": [
                {"type": "role", "content": "You are an agent.", "enabled": True},
            ],
        })
        assert len(event.payload["components"]) == 1

    def test_accepts_optional_variables_used(self):
        """prompt_resolved can include variables_used dict."""
        event = self._make_event({
            "prompt_id": "planner-prompt",
            "version_id": "v1",
            "resolved_text": "Max 5 steps.",
            "variables_used": {"max_steps": "5"},
        })
        assert event.payload["variables_used"]["max_steps"] == "5"


class TestRawEventNoValidation:
    """Test that raw LangChain events pass through without payload validation."""

    def test_raw_event_accepts_any_payload(self):
        """Raw LangChain events should accept any payload."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type="on_llm_start",
            agent_id=None,
            refs={},
            payload={"arbitrary": "data", "nested": {"values": [1, 2, 3]}},
        )
        assert event.payload["arbitrary"] == "data"

    def test_raw_event_accepts_empty_payload(self):
        """Raw LangChain events should accept empty payload."""
        event = TraceEvent(
            event_id=generate_event_id(),
            trace_id=generate_trace_id(),
            execution_id=generate_execution_id(),
            timestamp=utc_timestamp(),
            event_type="on_chain_end",
            agent_id=None,
            refs={},
            payload={},
        )
        assert event.payload == {}


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
            event_type="on_chain_start",
            agent_id=None,
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
            event_type="on_chain_start",
            agent_id=None,
            refs={},
            payload={},
        )
        assert event.sequence == 0


class TestContractFixtureValidation:
    """Test contract fixture is valid JSON (for future use)."""

    def test_sample_contract_is_valid_json(self):
        """Sample contract fixture should be valid JSON."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_contract.json"
        with open(fixture_path) as f:
            contract = json.load(f)

        assert contract["contract_id"] == "plan-input-v1"
        assert contract["contract_version"] == "1.0.0"
        assert "schema" in contract
        assert contract["schema"]["type"] == "object"
