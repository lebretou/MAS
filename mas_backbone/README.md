# MAS Backbone

Semantic backbone for MAS research prototype - multi-agent system tracing.

## Overview

This module provides the core data models and utilities for tracing and analyzing LLM-based multi-agent systems. It implements:

- **PromptArtifact**: Versioned, structured prompts for the authoring layer
- **ExecutionRecord**: Factual snapshots of executions with resolved prompt text
- **TraceEvent**: Semantic events describing agent communication and behavior
- **TraceSummary**: Analysis utilities for reconstructing agent graphs and detecting failures

## Installation

```bash
uv pip install -e "./mas_backbone[dev]"
```

## Usage

### Creating Events

```python
from mas_backbone.adapters.event_api import EventEmitter, ListSink
from mas_backbone.utils.identifiers import generate_execution_id, generate_trace_id

# create event sink and emitter
sink = ListSink()
emitter = EventEmitter(
    execution_id=generate_execution_id(),
    trace_id=generate_trace_id(),
    event_sink=sink,
)

# emit events
emitter.emit_input("planner", input_data={"query": "analyze data"})
emitter.emit_message("planner", "executor", summary="Sending plan")
emitter.emit_validation("executor", "contract-v1", "1.0.0", is_valid=True, errors=[])
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from mas_backbone.adapters.langchain_callback import MASCallbackHandler
from mas_backbone.adapters.event_api import ListSink

sink = ListSink()
handler = MASCallbackHandler(
    execution_id="exec-123",
    trace_id="trace-456",
    event_sink=sink,
    default_agent_id="my_agent",
)

llm = ChatOpenAI(callbacks=[handler])
```

### Analyzing Traces

```python
from mas_backbone.analysis.trace_summary import trace_summary

summary = trace_summary(sink.events)
print(f"Agents: {summary.agents}")
print(f"Edges: {summary.edges}")
print(f"Failures: {summary.failures}")
```

## Running Tests

```bash
cd mas_backbone
uv run pytest tests/ -v
```

## Design Principles

1. **Separate authoring from execution**: Prompts are artifacts, executions are facts
2. **Execution records store ground truth**: Always store resolved_prompt_text
3. **Traces represent communication**: Agent-to-agent IO is the primary unit of meaning
4. **OpenTelemetry is optional transport**: We define our own event semantics
