"""Generate dummy execution records and trace events for testing.

NOTE: This script generates raw LangChain-style events. The old semantic
event types (agent_message, agent_decision, etc.) have been removed.

Generates two scenarios:
- Scenario A (failure): chain execution with error
- Scenario B (success): chain execution completes successfully
"""

import json
import time
from pathlib import Path

from backbone.adapters.event_api import EventEmitter, FileSink
from backbone.adapters.langchain_callback import ListSink
from backbone.models.execution_record import (
    ContractRef,
    ExecutionRecord,
    ModelConfig,
    PromptArtifactRef,
)
from backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.utils.identifiers import (
    generate_execution_id,
    generate_span_id,
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
    """Generate Scenario A: chain execution with error.

    Timeline (using raw LangChain event types):
    1. on_chain_start (planner)
    2. on_llm_start
    3. on_llm_end
    4. on_chain_end (planner)
    5. on_chain_start (executor)
    6. on_chain_error
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

    # 1. planner chain starts
    planner_span = generate_span_id()
    emitter.emit(
        "on_chain_start",
        payload={
            "chain_name": "planner",
            "inputs": {"user_request": "Analyze sales data and create report"},
        },
        refs={"hint": {"agent_id": "planner"}, "langgraph": {"node": "planner"}},
    )
    time.sleep(0.01)

    # 2. LLM call starts
    llm_span = generate_span_id()
    emitter.emit(
        "on_llm_start",
        payload={
            "model_name": "gpt-4",
            "prompts": ["Create a plan for: Analyze sales data"],
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 3. LLM call ends
    emitter.emit(
        "on_llm_end",
        payload={
            "output_text": '{"steps": [{"action": "load_data"}, {"action": "analyze"}]}',
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 30},
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 4. planner chain ends
    emitter.emit(
        "on_chain_end",
        payload={
            "outputs": {"plan": {"steps": [{"action": "load_data"}, {"action": "analyze"}]}},
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 5. executor chain starts
    emitter.emit(
        "on_chain_start",
        payload={
            "chain_name": "executor",
            "inputs": {"plan": {"steps": [{"action": "load_data"}, {"action": "analyze"}]}},
        },
        refs={"hint": {"agent_id": "executor"}, "langgraph": {"node": "executor"}},
    )
    time.sleep(0.01)

    # 6. executor chain fails
    emitter.emit(
        "on_chain_error",
        payload={
            "error_type": "ValidationError",
            "error_message": "Plan validation failed: missing required field 'tool' in steps",
        },
        refs={"hint": {"agent_id": "executor"}},
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
    """Generate Scenario B: chain execution completes successfully.

    Timeline (using raw LangChain event types):
    1. on_chain_start (planner)
    2. on_llm_start
    3. on_llm_end
    4. on_chain_end (planner)
    5. on_chain_start (executor)
    6. on_tool_start
    7. on_tool_end
    8. on_chain_end (executor)
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

    # 1. planner chain starts
    emitter.emit(
        "on_chain_start",
        payload={
            "chain_name": "planner",
            "inputs": {"user_request": "Analyze sales data and create report"},
        },
        refs={"hint": {"agent_id": "planner"}, "langgraph": {"node": "planner"}},
    )
    time.sleep(0.01)

    # 2. LLM call starts
    emitter.emit(
        "on_llm_start",
        payload={
            "model_name": "gpt-4",
            "prompts": ["Create a plan for: Analyze sales data"],
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 3. LLM call ends
    emitter.emit(
        "on_llm_end",
        payload={
            "output_text": '{"steps": [{"action": "load_data", "tool": "csv_reader"}, {"action": "analyze", "tool": "pandas"}]}',
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 40},
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 4. planner chain ends
    emitter.emit(
        "on_chain_end",
        payload={
            "outputs": {
                "plan": {
                    "steps": [
                        {"action": "load_data", "tool": "csv_reader"},
                        {"action": "analyze", "tool": "pandas"},
                    ]
                }
            },
        },
        refs={"hint": {"agent_id": "planner"}},
    )
    time.sleep(0.01)

    # 5. executor chain starts
    emitter.emit(
        "on_chain_start",
        payload={
            "chain_name": "executor",
            "inputs": {
                "plan": {
                    "steps": [
                        {"action": "load_data", "tool": "csv_reader"},
                        {"action": "analyze", "tool": "pandas"},
                    ]
                }
            },
        },
        refs={"hint": {"agent_id": "executor"}, "langgraph": {"node": "executor"}},
    )
    time.sleep(0.01)

    # 6. tool call starts
    tool_span = generate_span_id()
    emitter.emit(
        "on_tool_start",
        payload={"tool_name": "csv_reader", "input": {"path": "sales.csv"}},
        refs={"hint": {"agent_id": "executor"}},
    )
    time.sleep(0.01)

    # 7. tool call ends
    emitter.emit(
        "on_tool_end",
        payload={"output": "Loaded 1000 rows from sales.csv"},
        refs={"hint": {"agent_id": "executor"}},
    )
    time.sleep(0.01)

    # 8. executor chain ends
    emitter.emit(
        "on_chain_end",
        payload={
            "outputs": {"status": "success", "results": {"rows_processed": 1000}},
        },
        refs={"hint": {"agent_id": "executor"}},
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
