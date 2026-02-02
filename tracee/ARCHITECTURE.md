# Tracee Architecture

Tracee is a developer tool for building and debugging multi-agent systems (MAS).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Core Features](#core-features)
3. [System Overview](#system-overview)
4. [LangChain and LangGraph Primer](#langchain-and-langgraph-primer)
5. [Directory Structure](#directory-structure)
6. [Quickstart](#quickstart)
7. [Core Components](#core-components)
8. [Data Models](#data-models)
9. [SDK Usage](#sdk-usage)
10. [API Reference](#api-reference)
11. [UI Integration](#ui-integration)

---

## Motivation

Based on the interviews with developers experienced in building multi-agent systems, two pain points came up repeatedly:

1. **Prompt iteration** Developers spend a lot of time tweaking prompts, but testing changes requires running the full agent pipeline. They prefer isolated testing—being able to test a single prompt against an LLM without running the entire system.

2. **Debugging means reading logs.** When something goes wrong, developers resort to manually reading through LLM inputs/outputs in terminal logs. 

## The 2 Core Features

Tracee addresses these pain points with and build on two core features (many of the existing tools have these two features):

### Playground

A prompt testing environment for rapid iteration. Developers can:
- Edit prompts in blocks (role, goal, constraints, etc.)
- Test prompts against different models individually. Please refer to LangSmith's similar feature page: https://docs.langchain.com/langsmith/run-evaluation-from-prompt-playground
- Version prompts and compare outputs across versions
- Load prompts in agent code via SDK with one line: `loader.get("my-prompt", "v2")`

### Trace Viewer

A structured view of agent execution for debugging. Instead of reading raw logs:
- See all LLM calls, tool invocations, and agent transitions in one place
- Correlate events with span IDs to understand parent-child relationships
- Identify which prompt version was used at runtime
- Analyze traces post-hoc to reconstruct what happened


## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           UI (Future)                               │
│            Trace Viewer · Prompt Editor · Playground                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                               │
│   /api/traces · /api/prompts · /api/playground · /api/model-configs │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SQLite Storage                               │
│              traces · prompts · prompt_versions · playground_runs   │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                        SDK (Agent Code)                             │
│          enable_tracing() · PromptLoader · RawCallbackHandler       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LangChain and LangGraph Primer

### What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It provides:

- **Abstractions for LLMs**: Unified interfaces for different model providers (OpenAI, Anthropic, etc.)
- **Prompt Management**: Templates and composition for complex prompts
- **Tool Integration**: Ability to give LLMs access to external tools and APIs
- **Callbacks**: Hooks into the execution lifecycle for logging, tracing, and monitoring

Key LangChain components:

```python
from langchain_openai import ChatOpenAI           # LLM wrapper for OpenAI models
from langchain_core.messages import (
    SystemMessage,   # system instructions for the LLM
    HumanMessage,    # user input messages
    AIMessage,       # LLM responses
    ToolMessage      # results from tool executions
)
from langchain_core.tools import tool, StructuredTool  # tool decorators and classes
from langchain_core.callbacks import BaseCallbackHandler  # for custom callbacks
```

### What is LangGraph?

**LangGraph** is built on top of LangChain and provides a way to build **stateful, multi-agent workflows** as directed graphs:

- **Nodes** are agents or functions that process and transform state
- **Edges** define the flow between nodes (can be conditional)
- **State** is a typed dictionary that flows through the graph

Key LangGraph components:

```python
from langgraph.graph import StateGraph, END

# create a graph with a typed state schema
workflow = StateGraph(AnalysisState)

# add nodes (agents)
workflow.add_node("agent_name", agent_function)

# define flow
workflow.set_entry_point("first_agent")
workflow.add_edge("agent_a", "agent_b")  # always go from A to B
workflow.add_conditional_edges(           # conditional routing
    "agent_a",
    routing_function,
    {"option1": "agent_b", "option2": END}
)

# compile and run
app = workflow.compile()
result = app.invoke(initial_state)
```

### Key Concepts

#### State Management

LangGraph uses a `TypedDict` to define the state schema. State flows through nodes and can be modified at each step:

```python
from typing import TypedDict, Annotated
from operator import add

class AnalysisState(TypedDict):
    dataset: pd.DataFrame
    user_query: str
    messages: Annotated[list[BaseMessage], add]  # messages are appended
    next_agent: str
```

The `Annotated[list, add]` pattern tells LangGraph to **append** to the list rather than replace it.

#### Conditional Edges

Routing decisions are made by functions that examine the state:

```python
def should_continue(state: AnalysisState) -> str:
    if state["relevance_decision"] == "relevant":
        return "planner"
    return "end"
```

#### Tool Binding

LLMs can be given tools that they can choose to call:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([tool1, tool2])
response = llm_with_tools.invoke(messages)

# check if the LLM wants to call a tool
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
```

#### Callbacks

LangChain/LangGraph callbacks are **hooks into the execution lifecycle**. They receive events like:
- Chain/agent started/ended
- LLM call started/ended
- Tool call started/ended
- Errors occurred

Tracee's `RawCallbackHandler` subscribes to these callbacks and captures them as `TraceEvent` objects.

---

## Directory Structure

```
tracee/
├── backbone/                    # Core tracing library
│   ├── adapters/
│   │   ├── event_api.py         # EventEmitter for manual events
│   │   ├── langchain_callback.py# RawCallbackHandler for LangChain
│   │   └── sinks.py             # ListSink, FileSink, HttpSink
│   ├── analysis/
│   │   └── trace_summary.py     # Reconstruct agent graphs from events
│   ├── models/
│   │   ├── prompt_artifact.py   # Prompt, PromptVersion, PromptComponent
│   │   ├── playground_run.py    # PlaygroundRun model
│   │   ├── saved_model_config.py# SavedModelConfig model
│   │   └── trace_event.py       # TraceEvent model
│   ├── sdk/
│   │   ├── tracing.py           # enable_tracing() context manager
│   │   └── prompt_loader.py     # PromptLoader for agent code
│   ├── utils/
│   │   └── identifiers.py       # UUID generation, timestamps
│   └── tracer.py                # Tracer class (legacy wrapper)
├── server/
│   ├── app.py                   # FastAPI application
│   ├── db.py                    # Database initialization
│   ├── routes.py                # Trace endpoints
│   ├── prompt_routes.py         # Prompt endpoints
│   ├── playground_routes.py     # Playground endpoints
│   ├── model_config_routes.py   # Model config endpoints
│   ├── trace_db.py              # Trace SQLite operations
│   ├── prompt_db.py             # Prompt SQLite operations
│   └── playground_db.py         # Playground SQLite operations
└── sample_mas/                  # Example multi-agent system
```

---

## Quickstart

This project uses **uv**, a fast Python package manager written in Rust. It's 10-100x faster than pip and handles virtual environments automatically. Install it from https://docs.astral.sh/uv/getting-started/installation/.

### 1. Run the Server

The server provides the REST API for the playground and trace viewer.

```bash
cd tracee

# create venv and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# set API keys for playground LLM calls
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key  # optional

# start server
uvicorn server.app:app --reload --port 8000

# verify it's running
curl http://localhost:8000/
```

### 2. Run the Sample MAS

The `sample_mas/` folder contains a data analysis multi-agent system for testing.

```bash
cd tracee/sample_mas

# install dependencies (uv auto-creates venv if needed)
uv pip install -r requirements.txt

# set OpenAI key
export OPENAI_API_KEY=your_key

# set trace API URL so traces are visible in /api/traces (requires server running)
export TRACE_API_URL=http://localhost:8000
```

**Enable LangSmith tracing (recommended):**

```bash
export LANGSMITH_API_KEY=your_langsmith_key
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=data-analysis-agents
```

**Run with sample data:**

```bash
# interactive mode
python main.py --sample

# single query
python main.py --sample --query "Create a correlation heatmap"
```

**Run with your own data:**

```bash
python main.py --dataset your_data.csv --query "Plot the distribution of age"
```

Traces are written to `outputs/traces/<trace_id>/trace_events.jsonl` and (if LangSmith is enabled) visible at https://smith.langchain.com/.

---

## Core Components

### Event Sinks

Events can be written to different destinations:

| Sink | Use Case |
|------|----------|
| `ListSink` | In-memory storage for tests |
| `FileSink` | Write to JSONL file |
| `HttpSink` | Post to server API |

### RawCallbackHandler

Captures LangChain events with minimal transformation. Events retain their original names:
- `on_chain_start`, `on_chain_end`, `on_chain_error`
- `on_llm_start`, `on_llm_end`, `on_llm_error`
- `on_tool_start`, `on_tool_end`, `on_tool_error`

### EventEmitter

Manual event emission for custom events. Currently supports `prompt_resolved` for tracking which prompt version was used.

---

## Data Models

### TraceEvent

```python
class TraceEvent(BaseModel):
    event_id: str           # UUID
    trace_id: str           # groups events in a trace
    execution_id: str
    timestamp: str          # ISO8601
    sequence: int | None    # ordering within trace
    event_type: str         # raw LangChain event or "prompt_resolved"
    agent_id: str | None
    span_id: str | None     # for correlating start/end pairs
    parent_span_id: str | None
    refs: dict[str, Any]    # namespaced: refs["langchain"], refs["langgraph"]
    payload: dict[str, Any] # event-specific data
```

### Prompt & PromptVersion

```python
class Prompt(BaseModel):
    prompt_id: str          # unique identifier
    name: str               # display name
    description: str | None
    latest_version_id: str | None
    created_at: str
    updated_at: str

class PromptVersion(BaseModel):
    prompt_id: str
    version_id: str         # e.g., "v1", "v2"
    name: str
    components: list[PromptComponent]
    variables: dict[str, str] | None
    created_at: str

    def resolve(self) -> str:
        """Concatenate enabled components."""
```

### PromptComponent

```python
class PromptComponentType(str, Enum):
    role = "role"
    goal = "goal"
    constraints = "constraints"
    io_rules = "io_rules"
    examples = "examples"
    safety = "safety"
    tool_instructions = "tool_instructions"

class PromptComponent(BaseModel):
    type: PromptComponentType
    content: str
    enabled: bool = True
```

### PlaygroundRun

```python
class PlaygroundRun(BaseModel):
    run_id: str
    prompt_id: str
    version_id: str
    model: str              # e.g., "gpt-4"
    provider: str           # "openai" or "anthropic"
    temperature: float
    max_tokens: int | None
    input_variables: dict[str, str]
    resolved_prompt: str    # final text sent to LLM
    output: str             # LLM response
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
```

---

## SDK Usage

### Enable Tracing

```python
from backbone.sdk import enable_tracing

# send events to server
with enable_tracing(base_url="http://localhost:8000") as ctx:
    result = graph.invoke(state, config={"callbacks": ctx.callbacks})

# or write to file
with enable_tracing(output_dir="./traces") as ctx:
    result = graph.invoke(state, config={"callbacks": ctx.callbacks})
```

### Load Prompts

```python
from backbone.sdk import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")

# get resolved text (auto-emits prompt_resolved event if tracing active)
system_prompt = loader.get("planner-prompt", "v2")

# get full version object
version = loader.get_version("planner-prompt", "latest")
```

### Combined Usage

```python
from backbone.sdk import enable_tracing, PromptLoader

loader = PromptLoader()

with enable_tracing(base_url="http://localhost:8000") as ctx:
    # prompt_resolved event auto-emitted
    system_prompt = loader.get("my-agent-prompt", "v1", agent_id="planner")
    
    # run agent with tracing
    result = graph.invoke(
        {"prompt": system_prompt, ...},
        config={"callbacks": ctx.callbacks}
    )
```

---

## API Reference

### Traces

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/traces` | List all traces |
| GET | `/api/traces/{trace_id}` | Get trace events |
| GET | `/api/traces/{trace_id}/summary` | Get computed summary |
| POST | `/api/traces/{trace_id}/events` | Append events |
| DELETE | `/api/traces/{trace_id}` | Delete trace |

### Prompts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/prompts` | List all prompts |
| POST | `/api/prompts` | Create prompt |
| GET | `/api/prompts/{id}` | Get prompt with versions |
| DELETE | `/api/prompts/{id}` | Delete prompt |
| POST | `/api/prompts/{id}/versions` | Create version |
| GET | `/api/prompts/{id}/versions/{vid}` | Get version |
| GET | `/api/prompts/{id}/versions/{vid}/resolve` | Get resolved text |
| GET | `/api/prompts/{id}/latest` | Get latest version |

### Playground

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/playground/run` | Execute prompt |
| GET | `/api/playground/runs` | List runs |
| GET | `/api/playground/runs/{id}` | Get run |

### Model Configs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/model-configs` | List configs |
| POST | `/api/model-configs` | Create config |
| GET | `/api/model-configs/{id}` | Get config |
| PATCH | `/api/model-configs/{id}` | Update config |
| DELETE | `/api/model-configs/{id}` | Delete config |
| POST | `/api/model-configs/{id}/set-default` | Set as default |

---

## UI Integration

### Setup

```bash
# start server
cd tracee
uvicorn server.app:app --reload --port 8000
```



### React Examples

#### Fetch Traces

```tsx
const API_BASE = "http://localhost:8000/api";

async function fetchTraces(): Promise<TraceMetadata[]> {
  const res = await fetch(`${API_BASE}/traces`);
  return res.json();
}

async function fetchTraceEvents(traceId: string): Promise<TraceEvent[]> {
  const res = await fetch(`${API_BASE}/traces/${traceId}`);
  return res.json();
}

// usage in component
function TraceList() {
  const [traces, setTraces] = useState<TraceMetadata[]>([]);

  useEffect(() => {
    fetchTraces().then(setTraces);
  }, []);

  return (
    <ul>
      {traces.map((t) => (
        <li key={t.trace_id}>
          {t.trace_id} ({t.event_count} events)
        </li>
      ))}
    </ul>
  );
}
```

#### Prompt Editor

```tsx
interface CreateVersionRequest {
  name: string;
  components: PromptComponent[];
  variables?: Record<string, string>;
}

async function createPromptVersion(
  promptId: string,
  data: CreateVersionRequest
): Promise<PromptVersion> {
  const res = await fetch(`${API_BASE}/prompts/${promptId}/versions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

function PromptEditor({ promptId }: { promptId: string }) {
  const [components, setComponents] = useState<PromptComponent[]>([
    { type: "role", content: "", enabled: true },
    { type: "goal", content: "", enabled: true },
  ]);

  const handleSave = async () => {
    const version = await createPromptVersion(promptId, {
      name: "Draft",
      components,
    });
    console.log("Created version:", version.version_id);
  };

  return (
    <div>
      {components.map((comp, i) => (
        <div key={i}>
          <label>{comp.type}</label>
          <textarea
            value={comp.content}
            onChange={(e) => {
              const updated = [...components];
              updated[i] = { ...comp, content: e.target.value };
              setComponents(updated);
            }}
          />
          <input
            type="checkbox"
            checked={comp.enabled}
            onChange={(e) => {
              const updated = [...components];
              updated[i] = { ...comp, enabled: e.target.checked };
              setComponents(updated);
            }}
          />
        </div>
      ))}
      <button onClick={handleSave}>Save Version</button>
    </div>
  );
}
```

#### Playground Runner

```tsx
interface RunRequest {
  prompt_id: string;
  version_id: string;
  input_variables?: Record<string, string>;
  model?: string;
  provider?: string;
  temperature?: number;
}

async function executeRun(data: RunRequest): Promise<PlaygroundRun> {
  const res = await fetch(`${API_BASE}/playground/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const json = await res.json();
  return json.run;
}

function PlaygroundRunner({ promptId }: { promptId: string }) {
  const [output, setOutput] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    const run = await executeRun({
      prompt_id: promptId,
      version_id: "latest",
      model: "gpt-4",
      provider: "openai",
      temperature: 0.7,
    });
    setOutput(run.output);
    setLoading(false);
  };

  return (
    <div>
      <button onClick={handleRun} disabled={loading}>
        {loading ? "Running..." : "Run"}
      </button>
      <pre>{output}</pre>
    </div>
  );
}
```

#### Trace Viewer

```tsx
function TraceViewer({ traceId }: { traceId: string }) {
  const [events, setEvents] = useState<TraceEvent[]>([]);

  useEffect(() => {
    fetchTraceEvents(traceId).then(setEvents);
  }, [traceId]);

  // group events by span for tree view
  const eventsBySpan = events.reduce((acc, event) => {
    const span = event.span_id || "root";
    if (!acc[span]) acc[span] = [];
    acc[span].push(event);
    return acc;
  }, {} as Record<string, TraceEvent[]>);

  return (
    <div>
      <h2>Trace: {traceId}</h2>
      {events.map((event) => (
        <div key={event.event_id} style={{ marginLeft: event.parent_span_id ? 20 : 0 }}>
          <strong>{event.event_type}</strong>
          <span> @ {event.timestamp}</span>
          <pre>{JSON.stringify(event.payload, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}
```

#### Model Config Selector

```tsx
interface SavedModelConfig {
  config_id: string;
  name: string;
  provider: string;
  model_name: string;
  temperature: number;
  is_default: boolean;
}

async function fetchModelConfigs(): Promise<SavedModelConfig[]> {
  const res = await fetch(`${API_BASE}/model-configs`);
  return res.json();
}

function ModelConfigSelector({
  onSelect,
}: {
  onSelect: (config: SavedModelConfig) => void;
}) {
  const [configs, setConfigs] = useState<SavedModelConfig[]>([]);

  useEffect(() => {
    fetchModelConfigs().then(setConfigs);
  }, []);

  return (
    <select onChange={(e) => {
      const config = configs.find((c) => c.config_id === e.target.value);
      if (config) onSelect(config);
    }}>
      <option value="">Select model config...</option>
      {configs.map((c) => (
        <option key={c.config_id} value={c.config_id}>
          {c.name} ({c.model_name})
        </option>
      ))}
    </select>
  );
}
```

---

## Running

```bash
# install dependencies
cd tracee
pip install -e .

# set API keys
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key  # optional

# start server
uvicorn server.app:app --reload --port 8000

# health check
curl http://localhost:8000/
```
