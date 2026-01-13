"""Integration tests for the full backbone workflow."""

import json
from pathlib import Path

import pytest

from mas_backbone.adapters.event_api import EventEmitter, ListSink
from mas_backbone.analysis.trace_summary import trace_summary
from mas_backbone.models.execution_record import ExecutionRecord
from mas_backbone.models.trace_event import EventType, TraceEvent
from mas_backbone.scripts.generate_dummy_run import (
    generate_scenario_a_failure,
    generate_scenario_b_success,
)
from mas_backbone.utils.identifiers import generate_execution_id, generate_trace_id


class TestDummyScenarios:
    """Test the dummy scenario generators."""

    def setup_method(self):
        """Set up test output directory."""
        self.output_dir = Path(__file__).parent / "test_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_scenario_a_generates_execution_record(self):
        """Scenario A should generate a valid ExecutionRecord."""
        record, events = generate_scenario_a_failure(self.output_dir)

        assert record.execution_id
        assert record.trace_id
        assert record.resolved_prompt_text
        assert record.prompt_refs is not None
        assert len(record.prompt_refs) > 0
        assert record.contract_refs is not None

    def test_scenario_a_generates_events(self):
        """Scenario A should generate expected events."""
        record, events = generate_scenario_a_failure(self.output_dir)

        assert len(events) == 9  # as specified in the scenario

        # check event types in order
        assert events[0].event_type == EventType.agent_input
        assert events[1].event_type == EventType.tool_call
        assert events[2].event_type == EventType.tool_call
        assert events[3].event_type == EventType.agent_decision
        assert events[4].event_type == EventType.agent_output
        assert events[5].event_type == EventType.agent_message
        assert events[6].event_type == EventType.agent_input
        assert events[7].event_type == EventType.contract_validation
        assert events[8].event_type == EventType.error

    def test_scenario_a_all_events_share_ids(self):
        """All events in Scenario A should share execution_id and trace_id."""
        record, events = generate_scenario_a_failure(self.output_dir)

        for event in events:
            assert event.execution_id == record.execution_id
            assert event.trace_id == record.trace_id

    def test_scenario_a_sequence_is_increasing(self):
        """Event sequence numbers should be strictly increasing."""
        record, events = generate_scenario_a_failure(self.output_dir)

        sequences = [e.sequence for e in events if e.sequence is not None]
        assert sequences == list(range(len(sequences)))

    def test_scenario_b_generates_success(self):
        """Scenario B should generate successful execution."""
        record, events = generate_scenario_b_success(self.output_dir)

        # find contract_validation event
        validation_events = [e for e in events if e.event_type == EventType.contract_validation]
        assert len(validation_events) == 1
        assert validation_events[0].payload["validation_result"]["is_valid"] is True

        # no error events in success scenario
        error_events = [e for e in events if e.event_type == EventType.error]
        assert len(error_events) == 0

    def test_scenario_b_generates_events(self):
        """Scenario B should generate expected events (same count, different outcome)."""
        record, events = generate_scenario_b_success(self.output_dir)

        assert len(events) == 9

        # last event should be agent_output, not error
        assert events[-1].event_type == EventType.agent_output

    def test_scenarios_write_files(self):
        """Both scenarios should write output files."""
        record_a, _ = generate_scenario_a_failure(self.output_dir)
        record_b, _ = generate_scenario_b_success(self.output_dir)

        # check scenario A files
        scenario_a_dir = self.output_dir / record_a.trace_id
        assert (scenario_a_dir / "execution_record.json").exists()
        assert (scenario_a_dir / "trace_events.jsonl").exists()
        assert (scenario_a_dir / "prompt_version.json").exists()

        # check scenario B files
        scenario_b_dir = self.output_dir / record_b.trace_id
        assert (scenario_b_dir / "execution_record.json").exists()
        assert (scenario_b_dir / "trace_events.jsonl").exists()


class TestTraceSummary:
    """Test trace summary analysis."""

    def setup_method(self):
        """Set up test output directory."""
        self.output_dir = Path(__file__).parent / "test_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_summary_reconstructs_planner_executor_edge(self):
        """TraceSummary should reconstruct planner -> executor edge."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        # check agents
        assert "planner" in summary.agents
        assert "executor" in summary.agents

        # check edge exists
        edge = next((e for e in summary.edges if e.from_agent == "planner" and e.to_agent == "executor"), None)
        assert edge is not None
        assert edge.message_count == 1

    def test_summary_detects_schema_failure(self):
        """TraceSummary should detect schema failure in Scenario A."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        # check failures
        assert len(summary.failures) > 0

        # check for schema error
        error_failures = [f for f in summary.failures if f.get("type") == "error"]
        assert any(f.get("error_type") == "schema" for f in error_failures)

    def test_summary_detects_failed_contract(self):
        """TraceSummary should detect failed contract validation."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        # check failed contracts
        assert len(summary.failed_contracts) > 0
        assert any(fc.contract_id == "plan-input-v1" for fc in summary.failed_contracts)

    def test_summary_shows_no_failures_for_success(self):
        """TraceSummary should show no failures for Scenario B."""
        record, events = generate_scenario_b_success(self.output_dir)
        summary = trace_summary(events)

        # no error failures
        error_failures = [f for f in summary.failures if f.get("type") == "error"]
        assert len(error_failures) == 0

        # no failed contracts
        assert len(summary.failed_contracts) == 0

    def test_summary_computes_tool_usage(self):
        """TraceSummary should compute tool usage statistics."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        # check tool usage
        assert len(summary.tool_usage) > 0

        # check llm.generate usage
        llm_usage = next((t for t in summary.tool_usage if t.tool_name == "llm.generate"), None)
        assert llm_usage is not None
        assert llm_usage.call_count >= 1

    def test_summary_computes_messages_by_edge(self):
        """TraceSummary should compute messages by edge."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        # check messages_by_edge
        assert ("planner", "executor") in summary.messages_by_edge
        assert summary.messages_by_edge[("planner", "executor")] == 1

    def test_summary_event_count(self):
        """TraceSummary should track total event count."""
        record, events = generate_scenario_a_failure(self.output_dir)
        summary = trace_summary(events)

        assert summary.event_count == len(events)


class TestExecutionRecordPromptRefLink:
    """Test that ExecutionRecord correctly links to PromptVersion."""

    def setup_method(self):
        """Set up test output directory."""
        self.output_dir = Path(__file__).parent / "test_outputs"
        self.output_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test outputs."""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_execution_record_links_to_prompt_version(self):
        """ExecutionRecord.prompt_refs should link to PromptVersion."""
        record, events = generate_scenario_a_failure(self.output_dir)

        # read the generated prompt version
        scenario_dir = self.output_dir / record.trace_id
        with open(scenario_dir / "prompt_version.json") as f:
            prompt_version = json.load(f)

        # verify link
        assert record.prompt_refs is not None
        assert len(record.prompt_refs) > 0

        prompt_ref = record.prompt_refs[0]
        assert prompt_ref.prompt_id == prompt_version["prompt_id"]
        assert prompt_ref.version_id == prompt_version["version_id"]

    def test_execution_record_resolved_prompt_matches_version(self):
        """ExecutionRecord.resolved_prompt_text should match PromptVersion components."""
        record, events = generate_scenario_a_failure(self.output_dir)

        # read the generated prompt version
        scenario_dir = self.output_dir / record.trace_id
        with open(scenario_dir / "prompt_version.json") as f:
            prompt_version = json.load(f)

        # the resolved text should contain content from all enabled components
        for component in prompt_version["components"]:
            if component.get("enabled", True):
                assert component["content"] in record.resolved_prompt_text


class TestEventEmitterSequence:
    """Test EventEmitter sequence generation."""

    def test_sequence_starts_at_zero(self):
        """EventEmitter sequence should start at 0."""
        sink = ListSink()
        emitter = EventEmitter(generate_execution_id(), generate_trace_id(), sink)

        emitter.emit_input("test", {})

        assert sink.events[0].sequence == 0

    def test_sequence_increments(self):
        """EventEmitter sequence should increment with each emit."""
        sink = ListSink()
        emitter = EventEmitter(generate_execution_id(), generate_trace_id(), sink)

        for i in range(5):
            emitter.emit_input("test", {})

        for i, event in enumerate(sink.events):
            assert event.sequence == i
