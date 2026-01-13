# MAS Backbone Architecture Guide

A developer-focused explanation of the codebase structure, core patterns, and how this fits into the larger vision.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Directory Structure](#directory-structure)
3. [Core Concepts (Backend Perspective)](#core-concepts-backend-perspective)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Core Functions Explained](#core-functions-explained)
6. [Design Patterns Used](#design-patterns-used)
7. [Fitting Into the Grand Plan](#fitting-into-the-grand-plan)
8. [Playground Integration Example](#playground-integration-example)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUTURE UI LAYER                                   │
│                   (Playground, Trace Viewer, Evaluations)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAS BACKBONE (THIS CODE)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Models    │  │  Adapters   │  │  Analysis   │  │       Utils         │ │
│  │ (Pydantic)  │  │ (Event API) │  │  (Summary)  │  │   (ID generation)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SYSTEMS                                   │
│              LangChain / LangGraph Agents, Storage (JSONL/SQLite)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

The backbone is a **headless data layer** — no UI, no server. It defines:
- What data structures exist (models)
- How events get captured (adapters)
- How to reconstruct what happened (analysis)

---

## Directory Structure

```
backbone/
├── __init__.py              # Package exports (public API)
├── pyproject.toml           # Python package configuration (uv/pip compatible)
├── models/                  # Pydantic data models
│   ├── __init__.py
│   ├── prompt_artifact.py   # Authoring-side: PromptVersion, PromptComponent
│   ├── execution_record.py  # Execution-side: ExecutionRecord, ModelConfig
│   └── trace_event.py       # Trace layer: TraceEvent, EventType enum
├── adapters/                # Integration points
│   ├── __init__.py
│   ├── event_api.py         # Manual event emission (EventEmitter, EventSink)
│   └── langchain_callback.py# Auto-capture from LangChain/LangGraph
├── analysis/                # Post-hoc analysis utilities
│   ├── __init__.py
│   └── trace_summary.py     # Reconstruct agent graph from events
├── utils/                   # Shared utilities
│   ├── __init__.py
│   └── identifiers.py       # UUID generation, timestamps
├── scripts/                 # Development/testing scripts
│   ├── __init__.py
│   └── generate_dummy_run.py# Creates example scenarios with events
└── tests/                   # pytest test suite
    ├── __init__.py
    ├── test_models.py       # Serialization round-trip tests
    ├── test_invariants.py   # Payload validation tests
    ├── test_langchain.py    # Callback handler tests
    ├── test_integration.py  # End-to-end workflow tests
    └── fixtures/
        └── sample_contract.json
```

---

## Core Concepts (Backend Perspective)

### 1. Pydantic Models as the Schema Layer

**Why Pydantic?**
- **Validation at construction time**: Invalid data fails immediately, not later in a pipeline.
- **Serialization for free**: `.model_dump_json()` and `.model_validate_json()` handle JSON round-trips.
- **Type hints are documentation**: The code is self-documenting for IDE autocompletion.

Every model has `model_config = {"extra": "forbid"}` — this means if you pass an unexpected field, it errors. This catches typos and schema drift early.

### 2. Protocol-Based Abstraction (EventSink)

```python
class EventSink(Protocol):
    def append(self, event: TraceEvent) -> None: ...
```

This is a **structural subtyping** pattern (duck typing with type hints). Any class with an `append(TraceEvent)` method satisfies this protocol. You don't need inheritance — just implement the method.

**Why this matters**:
- `ListSink` stores events in memory (for tests)
- `FileSink` appends to a JSONL file (for persistence)
- Future: `SQLiteSink`, `OpenTelemetrySink`, `KafkaSink` — all work without changing `EventEmitter`

### 3. Callback Handler Pattern (LangChain Integration)

LangChain uses the **Observer pattern** via callbacks. When an LLM runs, chain executes, or tool is called, LangChain invokes registered callbacks.

`MASCallbackHandler` subscribes to these callbacks and translates them into `TraceEvent` objects. The key insight: **callbacks capture low-level execution events, NOT semantic agent communication**. That's why `emit_message()` is a manual API call.

### 4. JSONL as Append-Only Log

Events are stored in JSONL (JSON Lines) format:
```
{"event_id": "...", "event_type": "agent_input", ...}
{"event_id": "...", "event_type": "tool_call", ...}
{"event_id": "...", "event_type": "agent_message", ...}
```

**Why JSONL over JSON array?**
- **Append-only**: Write events as they happen, no need to read-modify-write.
- **Streaming**: Parse one line at a time, don't load entire file.
- **Crash-safe**: If process dies mid-write, you lose at most one line.

---

## File-by-File Breakdown

### `models/prompt_artifact.py`

**Purpose**: Defines the **authoring-side** data structures — what gets created in a Playground before execution.

```python
class PromptComponentType(str, Enum):
    """Types of prompt sections that can be toggled on/off."""
    role = "role"           # "You are a..."
    goal = "goal"           # "Your objective is..."
    constraints = "constraints"  # "Do not..."
    io_rules = "io_rules"   # "Output format must be..."
    examples = "examples"   # Few-shot examples
    safety = "safety"       # Safety guardrails
    tool_instructions = "tool_instructions"  # How to use tools
```

```python
class PromptComponent(BaseModel):
    type: PromptComponentType
    content: str
    enabled: bool = True  # Toggle components without deleting them
```

```python
class PromptVersion(BaseModel):
    prompt_id: str          # e.g., "prompt-planner-001"
    version_id: str         # e.g., "v1.0.0" (immutable once created)
    name: str               # Human-readable name
    components: list[PromptComponent]
    variables: dict[str, str] | None  # Template variables like {{max_steps}}
    created_at: str         # ISO8601 timestamp
```

**Key invariant**: `PromptVersion` is **immutable** once created. Changes create new versions.

---

### `models/execution_record.py`

**Purpose**: Captures **what actually happened** during a single execution run.

```python
class ModelConfig(BaseModel):
    """LLM configuration used for this run."""
    provider: str       # "openai", "anthropic", etc.
    model_name: str     # "gpt-4", "claude-3", etc.
    temperature: float
    max_tokens: int
    seed: int | None    # For reproducibility
```

```python
class ExecutionRecord(BaseModel):
    execution_id: str   # Unique run identifier
    trace_id: str       # Links to TraceEvent stream
    origin: Literal["playground", "prod", "sdk", "batch_eval"]
    created_at: str
    
    llm_config: ModelConfig
    input_payload: dict              # What was passed in
    resolved_prompt_text: str        # REQUIRED: The actual text sent to LLM
    
    prompt_refs: list[PromptArtifactRef] | None  # Links to PromptVersion
    contract_refs: list[ContractRef] | None      # Links to validation schemas
    
    # Environment context
    git_commit: str | None
    app_version: str | None
    env: Literal["dev", "staging", "prod"] | None
    tags: list[str] | None
```

**Critical invariant** (enforced by `@model_validator`):
```python
if not self.resolved_prompt_text.strip():
    raise ValueError("resolved_prompt_text must not be empty")
```

This ensures every execution has a record of the exact prompt sent to the model — not a reference, the actual text.

---

### `models/trace_event.py`

**Purpose**: Defines the **semantic event vocabulary** — what types of things can happen during execution.

```python
class EventType(str, Enum):
    agent_input = "agent_input"           # Agent received input
    agent_output = "agent_output"         # Agent produced output
    agent_message = "agent_message"       # Agent-to-agent communication
    agent_decision = "agent_decision"     # Agent made a routing decision
    tool_call = "tool_call"               # Tool was invoked
    contract_validation = "contract_validation"  # Schema was validated
    error = "error"                        # Something went wrong
```

```python
class TraceEvent(BaseModel):
    event_id: str       # UUID for deduplication
    trace_id: str       # Groups events in one trace
    execution_id: str   # Links to ExecutionRecord
    timestamp: str      # ISO8601
    sequence: int | None  # Monotonic ordering within trace
    
    event_type: EventType
    agent_id: str       # Which agent this event belongs to
    
    span_id: str | None         # For correlating start/end pairs
    parent_span_id: str | None  # For nesting (e.g., tool within chain)
    
    refs: dict[str, Any]  # Namespaced metadata: refs["langchain"], refs["langgraph"]
    payload: dict[str, Any]  # Event-specific data
```

**Payload validation** — different event types have different required fields:

| Event Type | Required Payload Fields |
|------------|------------------------|
| `agent_message` | `to_agent_id`, (`message_summary` OR `payload_ref`) |
| `contract_validation` | `contract_id`, `contract_version`, `validation_result.is_valid`, `validation_result.errors` |
| `tool_call` | `tool_name`, `phase` (must be "start" or "end") |
| `error` | `error_type` (one of: schema, tool, model, infra, logic), `message` |

These are enforced in `_validate_*` methods called by `@model_validator`.

---

### `adapters/event_api.py`

**Purpose**: The **manual event emission API** — for events that can't be auto-captured from callbacks.

#### EventSink Protocol

```python
class EventSink(Protocol):
    def append(self, event: TraceEvent) -> None: ...
```

#### Concrete Implementations

```python
class ListSink:
    """In-memory storage for tests."""
    def __init__(self):
        self.events: list[TraceEvent] = []
    
    def append(self, event: TraceEvent) -> None:
        self.events.append(event)

class FileSink:
    """JSONL file storage for production."""
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, event: TraceEvent) -> None:
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")
```

#### EventEmitter (The Main API)

```python
class EventEmitter:
    def __init__(self, execution_id: str, trace_id: str, event_sink: EventSink):
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.event_sink = event_sink
        self._sequence = 0  # Internal counter for ordering
```

**Core methods** (all return the created `TraceEvent`):

| Method | Creates Event Type | Key Parameters |
|--------|-------------------|----------------|
| `emit_input()` | `agent_input` | `agent_id`, `input_data` |
| `emit_output()` | `agent_output` | `agent_id`, `output_data` |
| `emit_message()` | `agent_message` | `from_agent`, `to_agent`, `summary` |
| `emit_decision()` | `agent_decision` | `agent_id`, `decision`, `reasoning` |
| `emit_tool_call()` | `tool_call` | `agent_id`, `tool_name`, `phase` |
| `emit_validation()` | `contract_validation` | `agent_id`, `contract_id`, `is_valid`, `errors` |
| `emit_error()` | `error` | `agent_id`, `error_type`, `message` |

---

### `adapters/langchain_callback.py`

**Purpose**: **Automatic** event capture from LangChain/LangGraph execution.

```python
class MASCallbackHandler(BaseCallbackHandler):
    """
    Callback mapping:
    - on_chain_start  → agent_input
    - on_chain_end    → agent_output
    - on_llm_start    → tool_call (phase=start, tool_name="llm.generate")
    - on_llm_end      → tool_call (phase=end)
    - on_tool_start   → tool_call (phase=start)
    - on_tool_end     → tool_call (phase=end)
    - on_chain_error  → error (classified by error type)
    """
```

**Key design decision**: The callback handler **does NOT emit `agent_message`**. Why?

LangChain callbacks see execution events (chain started, LLM called, tool ran) but don't understand the **semantic relationship** between agents. The decision to hand off work from planner to executor is a domain concept — only the developer knows when that happens.

**Error classification** — `_classify_error()` maps exceptions to error types:

```python
def _classify_error(error: BaseException) -> str:
    error_name = type(error).__name__.lower()
    if "openai" in error_name or "api" in error_name:
        return "model"
    if "timeout" in error_name or "connection" in error_name:
        return "infra"
    if "validation" in error_name or "schema" in error_name:
        return "schema"
    if "tool" in error_name:
        return "tool"
    return "logic"  # default
```

---

### `analysis/trace_summary.py`

**Purpose**: **Post-hoc analysis** — reconstruct what happened from a list of events.

```python
@dataclass
class TraceSummary:
    execution_id: str
    trace_id: str
    agents: list[str]                    # All agents that participated
    edges: list[AgentEdge]               # Who talked to whom
    messages_by_edge: dict[tuple[str, str], int]  # Message counts
    failures: list[dict]                 # All failures (errors + validation)
    failed_contracts: list[FailedContract]  # Contracts that failed
    tool_usage: list[ToolUsage]          # Tool statistics
    event_count: int
```

The `trace_summary()` function iterates through events once, collecting:
1. Unique agents from `agent_id` fields
2. Edges from `agent_message` events
3. Failures from `error` and failed `contract_validation` events
4. Tool latencies by matching `tool_call` start/end pairs via `span_id`

---

### `utils/identifiers.py`

**Purpose**: Consistent ID generation across the codebase.

```python
def generate_execution_id() -> str:
    return str(uuid.uuid4())  # Full UUID for execution runs

def generate_trace_id() -> str:
    return str(uuid.uuid4())  # Full UUID for trace grouping

def generate_event_id() -> str:
    return str(uuid.uuid4())  # Full UUID for event deduplication

def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]  # 16-char hex for OpenTelemetry compatibility

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # ISO8601 with timezone
```

---

### `scripts/generate_dummy_run.py`

**Purpose**: Creates **realistic test scenarios** with all the right events and relationships.

Two scenarios are implemented:

#### Scenario A: Failure Case
```
planner receives input
planner calls LLM
planner makes decision
planner outputs plan
planner → executor (agent_message)
executor receives plan
executor validates contract → FAILS (missing 'tool' field)
executor emits error
```

#### Scenario B: Success Case
```
Same as A, but plan includes 'tool' field
executor validates contract → PASSES
executor outputs results
```

These scenarios demonstrate:
- Event sequencing with timestamps
- Span correlation (LLM start/end share span_id)
- PromptArtifactRef linking
- ContractRef linking
- Failure detection in trace_summary

---

### Test Files

| File | Tests |
|------|-------|
| `test_models.py` | JSON serialization round-trips for all models |
| `test_invariants.py` | Payload validation rules (what gets rejected) |
| `test_langchain.py` | Callback handler event emission |
| `test_integration.py` | End-to-end: generate scenarios → summarize traces |

---

## Core Functions Explained

### 1. `EventEmitter.emit()` — The Universal Emit

```python
def emit(
    self,
    event_type: EventType,
    agent_id: str,
    refs: dict | None = None,
    payload: dict | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> TraceEvent:
```

**What it does**:
1. Creates a `TraceEvent` with auto-generated `event_id` and `timestamp`
2. Assigns the next `sequence` number (monotonically increasing)
3. Generates a `span_id` if not provided
4. Validates the payload against `event_type` rules (via Pydantic)
5. Appends to the sink
6. Returns the created event (useful for chaining or getting the span_id)

**Why return the event?**
```python
# Correlate start/end of a tool call
start_event = emitter.emit_tool_call("agent", "search", "start")
# ... later ...
emitter.emit_tool_call("agent", "search", "end", span_id=start_event.span_id)
```

---

### 2. `trace_summary()` — Reconstruct the Graph

```python
def trace_summary(events: list[TraceEvent]) -> TraceSummary:
```

**What it does**:
1. **Collects agents**: Every unique `agent_id` seen in events
2. **Builds edges**: From `agent_message` events, extracts `from_agent` (the `agent_id`) and `to_agent` (from `payload.to_agent_id`)
3. **Detects failures**: Both `error` events and `contract_validation` where `is_valid=False`
4. **Computes tool latency**: Matches `tool_call` start/end pairs by `span_id`, calculates time difference

**Why this matters**: From a flat list of events, you can reconstruct:
- The **communication graph** (who talked to whom)
- The **failure locations** (which agent, which contract)
- **Performance bottlenecks** (tool latencies)

---

### 3. `MASCallbackHandler` — Automatic LangChain Integration

The callback handler maintains internal state:

```python
self._run_span_map: dict[str, str] = {}   # run_id → span_id
self._run_agent_map: dict[str, str] = {}  # run_id → agent_id
```

**Why track run_id → span_id?**

LangChain passes a `run_id` for each execution unit. To correlate start/end events (e.g., `on_llm_start` and `on_llm_end`), we need to use the **same span_id**. The map ensures this.

**Why track run_id → agent_id?**

Nested callbacks (e.g., `on_llm_end`) don't receive metadata. By storing the agent_id from `on_chain_start`, child callbacks can look it up via `parent_run_id`.

---

## Design Patterns Used

### 1. **Inversion of Control** (Dependency Injection)

`EventEmitter` doesn't know where events go — it just calls `event_sink.append()`. The caller provides the sink:

```python
# For tests
sink = ListSink()
emitter = EventEmitter(exec_id, trace_id, sink)

# For production
sink = FileSink("/path/to/trace.jsonl")
emitter = EventEmitter(exec_id, trace_id, sink)
```

### 2. **Validation at the Boundary**

All data validation happens in Pydantic models. By the time an object exists, it's guaranteed valid. No defensive checks scattered through the codebase.

### 3. **Event Sourcing Lite**

Events are the source of truth. The `TraceSummary` is a **derived view** computed from events. If you need a different view (e.g., timeline, agent-centric), write a new function that reads the same events.

### 4. **Namespace Isolation** (refs dict)

Instead of flat fields like `langchain_run_id`, `langgraph_node`, etc., the `refs` dict uses namespaces:

```python
refs = {
    "langchain": {"run_id": "...", "parent_run_id": "..."},
    "langgraph": {"node": "planner", "state_keys": ["messages"]},
    "prompt": {"prompt_id": "...", "version_id": "..."},
}
```

**Why?** New integrations don't change the schema. Just add a new namespace.

---

## Fitting Into the Grand Plan

The development plan describes a **two-phase architecture**:

### Current Code (Backbone — Days 1-2)

| Component | Status | What It Does |
|-----------|--------|--------------|
| Core Models | ✅ Done | Defines semantic types with validation |
| EventEmitter | ✅ Done | Manual event emission for semantic events |
| LangChain Callback | ✅ Done | Auto-capture of execution events |
| TraceSummary | ✅ Done | Reconstructs agent graph from events |
| Dummy Scenarios | ✅ Done | Proves the system works end-to-end |

### Future Work (Not Yet Implemented)

| Component | Purpose |
|-----------|---------|
| **Playground UI** | Visual prompt editing, component toggling |
| **Trace Viewer** | Visualize events as a timeline/graph |
| **Evaluation Framework** | Run prompts in batch, compare outputs |
| **SQLite/Postgres Storage** | Persistent storage with queries |
| **OpenTelemetry Export** | Send traces to external tools |

### How Current Code Enables Future UIs

The backbone provides the **data layer** that UIs read from and write to:

```
┌──────────────────────────────────────────────────────────┐
│                     PLAYGROUND UI                         │
│  - Edit PromptVersion.components                          │
│  - Toggle component.enabled                               │
│  - Set variables                                          │
│  - Click "Run" → creates ExecutionRecord                  │
└───────────────────────────┬──────────────────────────────┘
                            │ writes
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    MAS BACKBONE                           │
│  ExecutionRecord + TraceEvents stored to FileSink         │
└───────────────────────────┬──────────────────────────────┘
                            │ reads
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    TRACE VIEWER UI                        │
│  - Load TraceEvents from JSONL                            │
│  - Call trace_summary() to get agent graph                │
│  - Render as timeline / DAG                               │
└──────────────────────────────────────────────────────────┘
```

---

## Playground Integration Example

Here's how a hypothetical Playground would use this backbone:

### 1. User Creates a Prompt

```python
from backbone import PromptComponent, PromptComponentType, PromptVersion
from backbone.utils import utc_timestamp

# User builds a prompt in the UI
prompt_version = PromptVersion(
    prompt_id="my-planner-prompt",
    version_id="v1.0.0",  # Auto-generated or user-specified
    name="My Planner Agent",
    components=[
        PromptComponent(type=PromptComponentType.role, content="You are a planning agent."),
        PromptComponent(type=PromptComponentType.goal, content="Create a step-by-step plan."),
        PromptComponent(type=PromptComponentType.constraints, content="Max 5 steps.", enabled=True),
        PromptComponent(type=PromptComponentType.examples, content="Example: ...", enabled=False),  # Toggled off
    ],
    variables={"max_steps": "5"},
    created_at=utc_timestamp(),
)

# Save to storage (JSON file or database)
with open("prompts/my-planner-prompt-v1.json", "w") as f:
    f.write(prompt_version.model_dump_json(indent=2))
```

### 2. User Clicks "Run"

```python
from backbone import ExecutionRecord, ModelConfig, PromptArtifactRef
from backbone.adapters import EventEmitter, FileSink
from backbone.utils import generate_execution_id, generate_trace_id, utc_timestamp

# Generate IDs
execution_id = generate_execution_id()
trace_id = generate_trace_id()

# Resolve the prompt text (only enabled components)
resolved_prompt = "\n\n".join(
    c.content for c in prompt_version.components if c.enabled
)

# Create execution record
execution_record = ExecutionRecord(
    execution_id=execution_id,
    trace_id=trace_id,
    origin="playground",  # Mark as playground run
    created_at=utc_timestamp(),
    llm_config=ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=1000,
    ),
    input_payload={"user_query": "Analyze Q4 sales data"},
    resolved_prompt_text=resolved_prompt,  # THE ACTUAL PROMPT SENT
    prompt_refs=[
        PromptArtifactRef(
            prompt_id=prompt_version.prompt_id,
            version_id=prompt_version.version_id,
            agent_id="planner",
        )
    ],
    env="dev",
    tags=["playground", "experiment-1"],
)

# Set up event capture
output_dir = Path(f"outputs/{trace_id}")
output_dir.mkdir(parents=True, exist_ok=True)

sink = FileSink(output_dir / "trace_events.jsonl")
emitter = EventEmitter(execution_id, trace_id, sink)
```

### 3. Execute with LangChain + Manual Events

```python
from langchain_openai import ChatOpenAI
from backbone.adapters import MASCallbackHandler

# Create callback handler (shares the same sink)
callback = MASCallbackHandler(
    execution_id=execution_id,
    trace_id=trace_id,
    event_sink=sink,
    default_agent_id="planner",
)

# Run LangChain with callback
llm = ChatOpenAI(model="gpt-4", temperature=0)
response = llm.invoke(
    resolved_prompt,
    config={"callbacks": [callback], "metadata": {"agent_id": "planner"}},
)

# Emit semantic event manually (agent decided to hand off)
emitter.emit_message(
    from_agent="planner",
    to_agent="executor",
    summary=f"Sending plan with {len(response.content)} chars",
)

# Save execution record
with open(output_dir / "execution_record.json", "w") as f:
    f.write(execution_record.model_dump_json(indent=2))
```

### 4. Analyze the Run

```python
from backbone.analysis import trace_summary
from backbone.models import TraceEvent
import json

# Load events from JSONL
events = []
with open(output_dir / "trace_events.jsonl") as f:
    for line in f:
        events.append(TraceEvent.model_validate_json(line))

# Generate summary
summary = trace_summary(events)

print(f"Agents: {summary.agents}")
print(f"Edges: {[(e.from_agent, e.to_agent) for e in summary.edges]}")
print(f"Failures: {summary.failures}")
print(f"Tool usage: {[(t.tool_name, t.call_count) for t in summary.tool_usage]}")
```

---

## Summary

This codebase implements the **foundational data layer** for MAS observability:

1. **Models** define what data exists with strict validation
2. **Adapters** provide ways to capture data (manual + automatic)
3. **Analysis** reconstructs semantic relationships from flat events
4. **Utils** provide consistent ID/timestamp generation

The key insight is **separation of concerns**:
- Authoring (PromptVersion) vs Execution (ExecutionRecord)
- Automatic capture (callbacks) vs Semantic events (manual emit)
- Storage (sinks) vs Analysis (summary)

This foundation enables future UIs without changing core data models.

