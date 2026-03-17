"""Tests for payload invariants and validation rules."""

import json
from pathlib import Path

from backbone.models.trace_event import TraceEvent
from backbone.utils.identifiers import (
    generate_event_id,
    generate_execution_id,
    generate_trace_id,
    utc_timestamp,
)


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
