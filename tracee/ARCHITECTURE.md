# Tracee Architecture

Tracee is a tracing and prompt-debugging toolkit for multi-agent systems built on LangChain and LangGraph.

## Core Idea

Tracee has two user-facing surfaces:

- `tracee serve` runs the FastAPI backend and serves the built UI.
- `import tracee` exposes the public SDK for registering a compiled graph and tracing runs.

At runtime, Tracee combines:

- a persisted intent layer: graph topology, agent registry, prompts, model configs
- a runtime execution layer: LangChain/LangGraph callback events captured during a run

## System Overview

```text
Agent code
  ├─ import tracee
  ├─ tracee.init(app, ...)
  ├─ PromptLoader.get(..., agent_id=...)
  └─ with tracee.trace(): app.invoke(...)
            |
            v
FastAPI server
  ├─ /api/traces
  ├─ /api/prompts
  ├─ /api/graphs
  ├─ /api/agents
  ├─ /api/playground
  └─ /api/model-configs
            |
            v
SQLite + JSON artifacts
  ├─ traces / events
  ├─ prompts / versions
  ├─ graph topologies
  ├─ agent registry
  └─ playground runs
```

## Directory Structure

```text
tracee/
├── __init__.py                    # local-dev import shim: import tracee
├── backbone/
│   ├── adapters/
│   │   ├── langchain_callback.py  # RawCallbackHandler
│   │   └── sinks.py               # ListSink, FileSink, HttpSink
│   ├── analysis/
│   │   └── trace_summary.py       # trace summarization helpers
│   ├── models/
│   │   ├── agent_registry.py
│   │   ├── graph_topology.py
│   │   ├── prompt_artifact.py
│   │   ├── playground_run.py
│   │   ├── saved_model_config.py
│   │   └── trace_event.py
│   └── sdk/
│       ├── graph_extractor.py     # extract_topology(), extract_and_register()
│       ├── instrument.py          # public init() / trace()
│       ├── prompt_loader.py       # PromptLoader
│       └── tracing.py             # low-level enable_tracing()
├── playground-ui/                 # built frontend served by the backend
├── sample_mas/
│   ├── backend/agents/            # agents load prompts with agent_id
│   ├── backend/graph/workflow.py  # graph definition + tracee.init(...)
│   ├── backend/telemetry/         # LangSmith + server URL config
│   ├── main.py                    # sample CLI entry
│   └── seed_prompts.py            # optional prompt seeding
├── server/
│   ├── app.py                     # FastAPI app and UI serving
│   ├── agent_routes.py
│   ├── graph_routes.py
│   ├── model_config_routes.py
│   ├── playground_routes.py
│   ├── prompt_routes.py
│   └── trace_routes.py
└── tracee/
    ├── __init__.py                # installed package entrypoint
    └── cli.py                     # `tracee serve`
```

## Install And Run

This repo uses `uv`. Install optional extras from `pyproject.toml` depending on what you need:

- `.[server]` for the server and UI
- `.[dev]` for local development and the sample MAS

### Server / UI

```bash
cd tracee
uv venv
source .venv/bin/activate
uv pip install -e ".[server]"

export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key  # optional

tracee serve
# equivalent: python -m tracee.cli serve
```

Health check:

```bash
curl http://localhost:8000/api/health
```

`tracee serve` is the preferred entrypoint. Internally it runs the FastAPI app in `server.app`.

### Sample MAS

The sample flow expects the repo environment to already contain the development dependencies.

```bash
cd tracee
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Then:

```bash
# 1) start the tracee server in one terminal
tracee serve

# 2) optionally seed prompts into the local tracee server
python sample_mas/seed_prompts.py

# 3) set API keys and server URL
export OPENAI_API_KEY=your_key
export TRACE_API_URL=http://localhost:8000
export LANGSMITH_API_KEY=your_langsmith_key   # optional
export LANGSMITH_TRACING=true                 # optional
export LANGSMITH_PROJECT=data-analysis-agents # optional

# 4) run the sample MAS from sample_mas/
cd sample_mas
python main.py --sample
python main.py --sample --query "Create a correlation heatmap"
```

`sample_mas/seed_prompts.py` talks to `http://localhost:8000/api` by default, so the server must already be running if you use that step.

## Public SDK Usage

The main SDK path is now the top-level `tracee` package.

### Recommended Flow

```python
import tracee

app = workflow.compile()

tracee.init(
    app,
    graph_id="data-analysis-mas",
    name="Data Analysis MAS",
    description="Multi-agent system for interactive data analysis",
    server_url="http://localhost:8000",
)

with tracee.trace():
    result = app.invoke(initial_state)
```

What `tracee.init()` does:

- stores the default server URL used by `tracee.trace()`
- patches `invoke()` and `ainvoke()` so active trace callbacks are injected automatically
- if `graph_id` is provided, extracts and registers graph topology with the server

What `tracee.trace()` does:

- creates a tracing context backed by `enable_tracing()`
- sends callback events to the configured server
- lets `app.invoke(...)` run without manually threading `config={"callbacks": ...}`

### Prompt Loading

Prompts should be loaded through `PromptLoader` inside agent code, with `agent_id` set to the node ID:

```python
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")

system_prompt = loader.get("planner-prompt", agent_id="planner")
system_prompt, output_schema = loader.get_with_schema(
    "planner-prompt",
    agent_id="planner",
)
```

This is the basic prompt-to-agent linkage path. `PromptLoader.get(...)` and `get_with_schema(...)` resolve the prompt version and upsert the matching `agent_registry` entry through `/api/agents/{agent_id}`.

### Lower-Level Internals

The lower-level pieces still exist:

- `backbone.sdk.tracing.enable_tracing()` is the underlying tracing context manager
- `backbone.sdk.graph_extractor.extract_and_register()` performs explicit topology extraction

They are useful for internals and tests, but they are no longer the main documented user path.

## Prompt And Intent Linkage

Tracee no longer depends on a custom `prompt_resolved` event or an event-emitter layer for prompt association.

There are now two complementary linkage mechanisms:

### 1. Runtime prompt association

This is the minimum needed for prompt-aware traces:

- call `PromptLoader.get(..., agent_id=...)` or `get_with_schema(..., agent_id=...)`
- the loader fetches the prompt version
- the loader upserts `prompt_id` and `prompt_version_id` into the agent registry

This keeps prompt usage aligned with the actual prompt version resolved during execution.

### 2. Static graph metadata

This is optional, but improves the graph/intent view:

```python
workflow.add_node("planner", create_planner_agent, metadata={
    "prompt_id": "planner-prompt",
    "model": "o3-mini",
    "has_tools": True,
})
```

Metadata is read when graph topology is extracted and registered. It can enrich the UI with static information such as prompt IDs, model names, temperatures, tool flags, labels, and state schema. It is not required for the basic trace view or for prompt-version synchronization.

## Graph Visualization Architecture

The UI combines two layers:

### Intent layer

Persisted graph topology from `tracee.init(..., graph_id=...)` or `extract_and_register(...)`:

- nodes and edges from the compiled LangGraph
- optional node metadata from `add_node(..., metadata=...)`
- graph-level state schema extracted from the workflow

This layer gives the UI the complete workflow shape, including branches that may not execute in a particular run.

### Execution layer

Runtime `TraceEvent` records captured through LangChain callbacks while `with tracee.trace():` is active:

- chain / model / tool start and end events
- span relationships
- execution ordering and payloads

The execution layer is overlaid on top of the intent layer to show what actually ran.

### Registration Flow

```text
compile graph
  -> tracee.init(app, graph_id=...)
  -> extract_topology() reads nodes, edges, metadata, state schema
  -> PUT /api/graphs/{graph_id}
  -> server upserts graph and agent registry entries

run graph
  -> with tracee.trace():
  -> patched invoke()/ainvoke() inject callbacks
  -> RawCallbackHandler emits trace events
  -> HttpSink posts to /api/traces/{trace_id}/events

resolve prompts in agents
  -> PromptLoader.get(..., agent_id=...)
  -> PUT /api/agents/{agent_id}
  -> agent registry reflects resolved prompt + version
```

## Key Data Models

### TraceEvent

Represents a raw callback event captured during execution, including span IDs, refs, and provider-specific payloads.

### Prompt / PromptVersion

Stores structured prompt artifacts. `PromptVersion` resolves enabled components into the final system prompt and may also carry an `output_schema`.

### GraphTopology

Stores the static workflow shape:

- `nodes`
- `edges`
- optional `state_schema`

### AgentRegistryEntry

Stores the current per-agent association:

- `prompt_id`
- `prompt_version_id`
- optional model/tool metadata

The registry may be populated from graph registration, prompt loading, or both.

## API Surface

Main endpoints used by the current architecture:

- `/api/health` for process health
- `/api/traces` for trace storage and retrieval
- `/api/prompts` for prompt artifacts and versions
- `/api/graphs` for topology registration
- `/api/agents` for runtime prompt-to-agent linkage
- `/api/playground` for prompt playground execution
- `/api/model-configs` for saved model presets

## Running Notes

- `GET /` serves the built UI if `playground-ui/dist` exists; otherwise it falls back to health info.
- `tracee serve` is the preferred way to run the backend locally.
- The sample MAS currently uses `tracee.init(...)`, `with tracee.trace():`, and `PromptLoader.get_with_schema(..., agent_id=...)` in its workflow and agents.
