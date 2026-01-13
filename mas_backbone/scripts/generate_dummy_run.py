"""Generate dummy execution records and trace events for testing.

Implements two mandatory scenarios:
- Scenario A (failure): planner -> executor, contract fails
- Scenario B (success): planner -> executor, contract passes

Both scenarios include PromptArtifactRef and ContractRef links.
"""

import json
import time
from pathlib import Path

from mas_backbone.adapters.event_api import EventEmitter, FileSink, ListSink
from mas_backbone.models.execution_record import (
    ContractRef,
    ExecutionRecord,
    ModelConfig,
    PromptArtifactRef,
)
from mas_backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from mas_backbone.utils.identifiers import (
    generate_execution_id,
    generate_trace_id,
    utc_timestamp,
)


def create_sample_prompt_version() -> PromptVersion:
    """Create a sample prompt version for testing."""
    return PromptVersion(
        prompt_id="prompt-planner-001",
        version_id="v1.0.0",
        name="Planner Agent Prompt",
        components=[
            PromptComponent(
                type=PromptComponentType.role,
                content="You are a planning agent that creates execution plans.",
            ),
            PromptComponent(
                type=PromptComponentType.goal,
                content="Analyze the user request and create a step-by-step plan.",
            ),
            PromptComponent(
                type=PromptComponentType.constraints,
                content="Plans must have at most 5 steps. Each step must be actionable.",
            ),
            PromptComponent(
                type=PromptComponentType.io_rules,
                content="Output must be valid JSON with 'steps' array.",
            ),
        ],
        variables={"max_steps": "5"},
        created_at=utc_timestamp(),
    )


def generate_scenario_a_failure(output_dir: Path) -> tuple[ExecutionRecord, list]:
    """Generate Scenario A: planner -> executor, contract fails.

    Timeline:
    1. planner: agent_input
    2. planner: tool_call (llm.generate, start)
    3. planner: tool_call (llm.generate, end)
    4. planner: agent_decision
    5. planner: agent_output
    6. planner: agent_message -> executor
    7. executor: agent_input
    8. executor: contract_validation (is_valid=false)
    9. executor: error (type=schema)
    """
    execution_id = generate_execution_id()
    trace_id = generate_trace_id()

    # create output directory
    scenario_dir = output_dir / trace_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # create event sink
    sink = ListSink()
    emitter = EventEmitter(execution_id, trace_id, sink)

    # create sample prompt version
    prompt_version = create_sample_prompt_version()

    # 1. planner receives input
    emitter.emit_input(
        "planner",
        input_data={"user_request": "Analyze sales data and create report"},
        refs={"langgraph": {"node": "planner"}},
    )
    time.sleep(0.01)  # small delay for timestamp ordering

    # 2. planner: tool_call (llm.generate, start)
    llm_span_id = emitter.emit_tool_call(
        "planner",
        tool_name="llm.generate",
        phase="start",
        tool_input={"prompt": "Create a plan for: Analyze sales data"},
        refs={"llm": {"model": "gpt-4"}},
    ).span_id
    time.sleep(0.01)

    # 3. planner: tool_call (llm.generate, end)
    emitter.emit_tool_call(
        "planner",
        tool_name="llm.generate",
        phase="end",
        tool_output={"text": '{"steps": [{"action": "load_data"}, {"action": "analyze"}]}'},
        span_id=llm_span_id,
    )
    time.sleep(0.01)

    # 4. planner: agent_decision
    emitter.emit_decision(
        "planner",
        decision="execute_plan",
        reasoning="Plan looks valid, sending to executor",
    )
    time.sleep(0.01)

    # 5. planner: agent_output
    emitter.emit_output(
        "planner",
        output_data={"plan": {"steps": [{"action": "load_data"}, {"action": "analyze"}]}},
    )
    time.sleep(0.01)

    # 6. planner: agent_message -> executor
    emitter.emit_message(
        "planner",
        "executor",
        summary="Sending execution plan with 2 steps",
        refs={"contract": {"contract_id": "plan-output-v1", "version": "1.0.0"}},
    )
    time.sleep(0.01)

    # 7. executor: agent_input
    emitter.emit_input(
        "executor",
        input_data={"plan": {"steps": [{"action": "load_data"}, {"action": "analyze"}]}},
        refs={"langgraph": {"node": "executor"}},
    )
    time.sleep(0.01)

    # 8. executor: contract_validation (is_valid=false)
    emitter.emit_validation(
        "executor",
        contract_id="plan-input-v1",
        contract_version="1.0.0",
        is_valid=False,
        errors=[
            {"path": "$.steps[0].tool", "message": "missing required field 'tool'"},
            {"path": "$.steps[1].tool", "message": "missing required field 'tool'"},
        ],
    )
    time.sleep(0.01)

    # 9. executor: error (type=schema)
    emitter.emit_error(
        "executor",
        error_type="schema",
        message="Plan validation failed: missing required field 'tool' in steps",
        details={"failed_paths": ["$.steps[0].tool", "$.steps[1].tool"]},
    )

    # create execution record
    execution_record = ExecutionRecord(
        execution_id=execution_id,
        trace_id=trace_id,
        origin="playground",
        created_at=utc_timestamp(),
        llm_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            max_tokens=1000,
            seed=42,
        ),
        input_payload={"user_request": "Analyze sales data and create report"},
        resolved_prompt_text="\n".join(
            c.content for c in prompt_version.components if c.enabled
        ),
        prompt_refs=[
            PromptArtifactRef(
                prompt_id=prompt_version.prompt_id,
                version_id=prompt_version.version_id,
                agent_id="planner",
            )
        ],
        contract_refs=[
            ContractRef(
                contract_id="plan-output-v1",
                contract_version="1.0.0",
                agent_id="planner",
            ),
            ContractRef(
                contract_id="plan-input-v1",
                contract_version="1.0.0",
                agent_id="executor",
            ),
        ],
        env="dev",
        tags=["scenario-a", "failure", "test"],
    )

    # write outputs
    with open(scenario_dir / "execution_record.json", "w") as f:
        f.write(execution_record.model_dump_json(indent=2))

    with open(scenario_dir / "trace_events.jsonl", "w") as f:
        for event in sink.events:
            f.write(event.model_dump_json() + "\n")

    with open(scenario_dir / "prompt_version.json", "w") as f:
        f.write(prompt_version.model_dump_json(indent=2))

    print(f"Scenario A (failure) generated: {scenario_dir}")

    return execution_record, sink.events


def generate_scenario_b_success(output_dir: Path) -> tuple[ExecutionRecord, list]:
    """Generate Scenario B: planner -> executor, contract passes.

    Timeline:
    1-6. Same as Scenario A
    7. executor: agent_input
    8. executor: contract_validation (is_valid=true)
    9. executor: agent_output
    """
    execution_id = generate_execution_id()
    trace_id = generate_trace_id()

    # create output directory
    scenario_dir = output_dir / trace_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # create event sink
    sink = ListSink()
    emitter = EventEmitter(execution_id, trace_id, sink)

    # create sample prompt version
    prompt_version = create_sample_prompt_version()

    # 1. planner receives input
    emitter.emit_input(
        "planner",
        input_data={"user_request": "Analyze sales data and create report"},
        refs={"langgraph": {"node": "planner"}},
    )
    time.sleep(0.01)

    # 2. planner: tool_call (llm.generate, start)
    llm_span_id = emitter.emit_tool_call(
        "planner",
        tool_name="llm.generate",
        phase="start",
        tool_input={"prompt": "Create a plan for: Analyze sales data"},
        refs={"llm": {"model": "gpt-4"}},
    ).span_id
    time.sleep(0.01)

    # 3. planner: tool_call (llm.generate, end)
    emitter.emit_tool_call(
        "planner",
        tool_name="llm.generate",
        phase="end",
        tool_output={
            "text": '{"steps": [{"action": "load_data", "tool": "csv_reader"}, {"action": "analyze", "tool": "pandas"}]}'
        },
        span_id=llm_span_id,
    )
    time.sleep(0.01)

    # 4. planner: agent_decision
    emitter.emit_decision(
        "planner",
        decision="execute_plan",
        reasoning="Plan looks valid with tool specifications, sending to executor",
    )
    time.sleep(0.01)

    # 5. planner: agent_output (valid plan with tool field)
    emitter.emit_output(
        "planner",
        output_data={
            "plan": {
                "steps": [
                    {"action": "load_data", "tool": "csv_reader"},
                    {"action": "analyze", "tool": "pandas"},
                ]
            }
        },
    )
    time.sleep(0.01)

    # 6. planner: agent_message -> executor
    emitter.emit_message(
        "planner",
        "executor",
        summary="Sending execution plan with 2 steps (including tools)",
        refs={"contract": {"contract_id": "plan-output-v1", "version": "1.0.0"}},
    )
    time.sleep(0.01)

    # 7. executor: agent_input
    emitter.emit_input(
        "executor",
        input_data={
            "plan": {
                "steps": [
                    {"action": "load_data", "tool": "csv_reader"},
                    {"action": "analyze", "tool": "pandas"},
                ]
            }
        },
        refs={"langgraph": {"node": "executor"}},
    )
    time.sleep(0.01)

    # 8. executor: contract_validation (is_valid=true)
    emitter.emit_validation(
        "executor",
        contract_id="plan-input-v1",
        contract_version="1.0.0",
        is_valid=True,
        errors=[],
    )
    time.sleep(0.01)

    # 9. executor: agent_output
    emitter.emit_output(
        "executor",
        output_data={"status": "success", "results": {"rows_processed": 1000}},
    )

    # create execution record
    execution_record = ExecutionRecord(
        execution_id=execution_id,
        trace_id=trace_id,
        origin="playground",
        created_at=utc_timestamp(),
        llm_config=ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.0,
            max_tokens=1000,
            seed=42,
        ),
        input_payload={"user_request": "Analyze sales data and create report"},
        resolved_prompt_text="\n".join(
            c.content for c in prompt_version.components if c.enabled
        ),
        prompt_refs=[
            PromptArtifactRef(
                prompt_id=prompt_version.prompt_id,
                version_id=prompt_version.version_id,
                agent_id="planner",
            )
        ],
        contract_refs=[
            ContractRef(
                contract_id="plan-output-v1",
                contract_version="1.0.0",
                agent_id="planner",
            ),
            ContractRef(
                contract_id="plan-input-v1",
                contract_version="1.0.0",
                agent_id="executor",
            ),
        ],
        env="dev",
        tags=["scenario-b", "success", "test"],
    )

    # write outputs
    with open(scenario_dir / "execution_record.json", "w") as f:
        f.write(execution_record.model_dump_json(indent=2))

    with open(scenario_dir / "trace_events.jsonl", "w") as f:
        for event in sink.events:
            f.write(event.model_dump_json() + "\n")

    with open(scenario_dir / "prompt_version.json", "w") as f:
        f.write(prompt_version.model_dump_json(indent=2))

    print(f"Scenario B (success) generated: {scenario_dir}")

    return execution_record, sink.events


def main():
    """Generate both scenarios."""
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("Generating dummy scenarios...")
    print()

    # generate both scenarios
    record_a, events_a = generate_scenario_a_failure(output_dir)
    print(f"  Execution ID: {record_a.execution_id}")
    print(f"  Trace ID: {record_a.trace_id}")
    print(f"  Events: {len(events_a)}")
    print()

    record_b, events_b = generate_scenario_b_success(output_dir)
    print(f"  Execution ID: {record_b.execution_id}")
    print(f"  Trace ID: {record_b.trace_id}")
    print(f"  Events: {len(events_b)}")
    print()

    print("Done!")


if __name__ == "__main__":
    main()
