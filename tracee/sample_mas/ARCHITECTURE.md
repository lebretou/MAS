# Data Analysis Multi-Agent System — Architecture Guide

This document provides an explanation of the MAS implementation in this folder, including a brief overview of LangChain and LangGraph, how to set up agents, explanations of each file, and how the callback/telemetry system works.

---

## Table of Contents

1. [Overview](#overview)
2. [LangChain and LangGraph Primer](#langchain-and-langgraph-primer)
   - [What is LangChain?](#what-is-langchain)
   - [What is LangGraph?](#what-is-langgraph)
   - [Key Concepts](#key-concepts)
3. [Folder Structure](#folder-structure)
4. [Architecture Diagram](#architecture-diagram)
5. [Setting Up a Multi-Agent System](#setting-up-a-multi-agent-system)
6. [File-by-File Explanation](#file-by-file-explanation)
   - [CLI Entry Point](#cli-entry-point-mainpy)
   - [State Schema](#state-schema-backendstateschemapybackendstate__init__py)
   - [Workflow Graph](#workflow-graph-backendgraphworkflowpy)
   - [Agents](#agents)
   - [Tools](#tools)
   - [Telemetry Configuration](#telemetry-configuration-backendtelemetryconfigpy)
7. [How the Callback System Works](#how-the-callback-system-works)
8. [Execution Flow](#execution-flow)
9. [Running the System](#running-the-system)

---

## Overview

This project is a **Data Analysis Multi-Agent System** that allows users to:
1. Load datasets (CSV, Excel, JSON, Parquet) from the command line
2. Ask natural language questions about their data
3. Get automated analysis, visualizations, and insights

The system uses **four specialized agents** orchestrated by **LangGraph**:

| Agent | Role |
|-------|------|
| **Interaction** | Validates queries, answers simple questions, routes requests |
| **Planner** | Creates analysis plans and coding instructions |
| **Coding** | Generates and executes Python code for analysis |
| **Summary** | Interprets results and creates summaries |

---

## LangChain and LangGraph Primer

### What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It provides:

- **Abstractions for LLMs**: Unified interfaces for different model providers (OpenAI, Anthropic, etc.)
- **Prompt Management**: Templates and composition for complex prompts
- **Tool Integration**: Ability to give LLMs access to external tools and APIs
- **Callbacks**: Hooks into the execution lifecycle for logging, tracing, and monitoring

Key LangChain components used in this project:

```python
from langchain_openai import ChatOpenAI           # LLM wrapper for OpenAI models
from langchain_core.messages import (
    SystemMessage,   # System instructions for the LLM
    HumanMessage,    # User input messages
    AIMessage,       # LLM responses
    ToolMessage      # Results from tool executions
)
from langchain_core.tools import tool, StructuredTool  # Tool decorators and classes
from langchain_core.callbacks import BaseCallbackHandler  # For custom callbacks
```

### What is LangGraph?

**LangGraph** is built on top of LangChain and provides a way to build **stateful, multi-agent workflows** as directed graphs. Think of it as a state machine where:

- **Nodes** are agents or functions that process and transform state
- **Edges** define the flow between nodes (can be conditional)
- **State** is a typed dictionary that flows through the graph. Think of this as the shared context that all agents have access to. 

Key LangGraph components:

```python
from langgraph.graph import StateGraph, END

# Create a graph with a typed state schema
workflow = StateGraph(AnalysisState)

# Add nodes (agents)
workflow.add_node("agent_name", agent_function)

# Define flow
workflow.set_entry_point("first_agent")
workflow.add_edge("agent_a", "agent_b")  # Always go from A to B
workflow.add_conditional_edges(           # Conditional routing
    "agent_a",
    routing_function,
    {"option1": "agent_b", "option2": END}
)

# Compile and run
app = workflow.compile()
result = app.invoke(initial_state)
```

### Key Concepts

#### 1. State Management
LangGraph uses a `TypedDict` to define the state schema. State flows through nodes and can be modified at each step:

```python
class AnalysisState(TypedDict):
    dataset: pd.DataFrame
    user_query: str
    messages: Annotated[list[BaseMessage], add]  # Messages are appended
    next_agent: str
```

The `Annotated[list, add]` pattern tells LangGraph to **append** to the list rather than replace it.

#### 2. Conditional Edges
Routing decisions are made by functions that examine the state:

```python
def should_continue(state: AnalysisState) -> str:
    if state["relevance_decision"] == "relevant":
        return "planner"
    return "end"
```

#### 3. Tool Binding
LLMs can be given tools that they can choose to call:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([tool1, tool2])
response = llm_with_tools.invoke(messages)

# Check if the LLM wants to call a tool
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
```

---

## Folder Structure

```
sample_mas/
├── backend/                    # Main application code
│   ├── __init__.py
│   ├── agents/                 # Agent implementations
│   │   ├── __init__.py
│   │   ├── interaction.py      # Query validation & routing
│   │   ├── planner.py          # Analysis planning
│   │   ├── coding.py           # Code generation & execution
│   │   └── summary.py          # Result summarization
│   ├── graph/                  # LangGraph workflow definition
│   │   ├── __init__.py
│   │   └── workflow.py         # Graph construction & execution
│   ├── state/                  # State schema definition
│   │   ├── __init__.py
│   │   └── schema.py           # TypedDict state schema
│   ├── telemetry/              # Tracing & callbacks configuration
│   │   ├── __init__.py
│   │   └── config.py           # LangSmith + MAS backbone setup
│   └── tools/                  # LangChain tools for agents
│       ├── __init__.py
│       ├── dataset_tools.py    # Tools for dataset inspection
│       └── execution_tools.py  # Safe code execution sandbox
├── outputs/                    # Generated plots & traces
├── main.py                     # CLI entry point
├── test_system.py              # Integration tests
├── sample_data.csv             # Demo dataset
└── README.md                   # Quick start guide
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLI Entry Point                                  │
│                               (main.py)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   Dataset   │                                                            │
│  │   (CSV/     │                                                            │
│  │   Excel/    │                                                            │
│  │   JSON)     │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                  LangGraph Workflow                              │        │
│  │                    (workflow.py)                                 │        │
│  │                                                                  │        │
│  │  ┌───────────┐   ┌─────────┐   ┌────────┐   ┌──────┐            │        │
│  │  │Interaction│──▶│ Planner │──▶│ Coding │──▶│Summary│            │        │
│  │  │   Agent   │   │  Agent  │   │ Agent  │   │Agent │            │        │
│  │  └─────┬─────┘   └─────────┘   └───┬────┘   └──────┘            │        │
│  │        │                           │                             │        │
│  │        │ (uses tools)              │ (executes)                  │        │
│  │        ▼                           ▼                             │        │
│  │  ┌───────────┐              ┌───────────┐                        │        │
│  │  │ Dataset   │              │ Execution │                        │        │
│  │  │  Tools    │              │ Sandbox   │                        │        │
│  │  └───────────┘              └───────────┘                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
│                        ┌─────────────────────────────────┐                  │
│                        │         Callback System         │                  │
│                        │  ┌───────────┐  ┌────────────┐  │                  │
│                        │  │ LangSmith │  │MAS Backbone│  │                  │
│                        │  └───────────┘  └────────────┘  │                  │
│                        └─────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Setting Up a Multi-Agent System

Here's a step-by-step guide to building a MAS with LangGraph:

### Step 1: Define Your State Schema

Create a `TypedDict` that holds all data shared between agents:

```python
from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage

class AnalysisState(TypedDict):
    # Shared data
    dataset: pd.DataFrame
    user_query: str
    
    # Message history (use Annotated[list, add] to append)
    messages: Annotated[list[BaseMessage], add]
    
    # Agent outputs
    analysis_plan: str
    generated_code: str
    final_summary: str
    
    # Control flow
    next_agent: str
```

### Step 2: Create Agent Functions

In LangChain/LangGraph, agents(nodes) are treated as functions in traditional programming. Each agent is a function that takes state and returns modified state:

```python
def create_my_agent(state: AnalysisState) -> AnalysisState:
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create messages
    system_msg = SystemMessage(content="Your system prompt here")
    user_msg = HumanMessage(content=state["user_query"])
    
    # Get LLM response
    response = llm.invoke([system_msg, user_msg])
    
    # Update state
    state["agent_output"] = response.content
    state["next_agent"] = "next_agent_name"
    state["messages"].append(AIMessage(content=response.content))
    
    return state
```

### Step 3: Build the Workflow Graph

```python
from langgraph.graph import StateGraph, END

def create_workflow() -> StateGraph:
    workflow = StateGraph(AnalysisState)
    
    # Add nodes (agents)
    workflow.add_node("agent_a", create_agent_a)
    workflow.add_node("agent_b", create_agent_b)
    workflow.add_node("agent_c", create_agent_c)
    
    # Set entry point
    workflow.set_entry_point("agent_a")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent_a",
        lambda state: state["next_agent"],
        {"agent_b": "agent_b", "end": END}
    )
    workflow.add_edge("agent_b", "agent_c")
    workflow.add_edge("agent_c", END)
    
    return workflow.compile()
```

### Step 4: Run the Workflow

```python
app = create_workflow()

initial_state = {
    "dataset": my_dataframe,
    "user_query": "Analyze this data",
    "messages": [],
    # ... other fields initialized
}

result = app.invoke(initial_state)
print(result["final_summary"])
```

---

## File-by-File Explanation

### CLI Entry Point: `main.py`

The command-line interface that allows users to interact with the system:

#### Key Functions

| Function | Description |
|----------|-------------|
| `load_dataset()` | Loads CSV, Excel, JSON, or Parquet files into pandas DataFrame |
| `interactive_mode()` | Runs a loop accepting multiple queries from the user |
| `single_query_mode()` | Executes a single query and exits |
| `main()` | Parses arguments and orchestrates the execution |

#### Usage Modes

**Interactive Mode** (default when no `--query` is provided):
```bash
python main.py --dataset data.csv
```

**Single Query Mode**:
```bash
python main.py --dataset data.csv --query "Plot histogram of age column"
```

**With Sample Data**:
```bash
python main.py --sample --query "Create a correlation heatmap"
```

---

### State Schema: `backend/state/schema.py`

Defines the `AnalysisState` TypedDict that flows through all agents:

```python
class AnalysisState(TypedDict):
    # Dataset information
    dataset: pd.DataFrame           # The actual data
    dataset_path: str               # Original filename
    dataset_info: dict              # Columns, dtypes, shape
    
    # Conversation history
    messages: Annotated[list[BaseMessage], add]  # Appended across agents
    user_query: str                 # Original user question
    
    # Agent outputs (populated as workflow progresses)
    relevance_decision: str         # "relevant" or "chat_only"
    analysis_plan: str              # From planner agent
    coding_prompt: str              # Instructions for coding agent
    generated_code: str             # Python code from coding agent
    execution_result: dict          # {success, stdout, plots, error}
    final_summary: str              # User-facing summary
    
    # Control flow
    next_agent: str                 # Routing decision
    
    # Session & telemetry
    session_id: str
    callbacks: list                 # Callback handlers for tracing
```

**Important**: The `Annotated[list[BaseMessage], add]` syntax tells LangGraph to **append** new messages to the existing list rather than replacing it. This preserves conversation history across all agents.

---

### Workflow Graph: `backend/graph/workflow.py`

Constructs and executes the LangGraph workflow.

#### `create_workflow() -> StateGraph`

Builds the graph structure:

```python
def create_workflow() -> StateGraph:
    workflow = StateGraph(AnalysisState)
    
    # Add four agent nodes
    workflow.add_node("interaction", create_interaction_agent)
    workflow.add_node("planner", create_planner_agent)
    workflow.add_node("coding", create_coding_agent)
    workflow.add_node("summary", create_summary_agent)
    
    # Entry point
    workflow.set_entry_point("interaction")
    
    # Conditional edge: interaction decides whether to continue
    workflow.add_conditional_edges(
        "interaction",
        should_continue_to_planner,
        {"planner": "planner", "end": END}
    )
    
    # Linear flow for remaining agents
    workflow.add_edge("planner", "coding")
    workflow.add_edge("coding", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()
```

#### `should_continue_to_planner(state) -> str`

Routing function that decides whether to proceed to planning or end early:

```python
def should_continue_to_planner(state: AnalysisState) -> str:
    if state.get("relevance_decision") == "relevant":
        return "planner"  # Complex query → continue to planner
    return "end"          # Simple query → interaction already answered
```

#### `run_analysis_workflow()`

Main entry point that:
1. Extracts dataset metadata
2. Creates callback handlers for tracing
3. Initializes the state dictionary
4. Invokes the compiled graph
5. Returns formatted results

---

### Agents

#### Interaction Agent: `backend/agents/interaction.py`

**Purpose**: First point of contact. Validates query relevance and answers simple questions.

##### Key Function: `create_interaction_agent(state) -> AnalysisState`

```python
def create_interaction_agent(state: AnalysisState) -> AnalysisState:
    # Create LLM with tool binding
    dataset = state["dataset"]
    tools = create_dataset_tools_for_agent(dataset)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=callbacks)
    llm_with_tools = llm.bind_tools(tools)
    
    # Agentic loop: keep calling tools until LLM gives final answer
    messages = [SystemMessage(...), HumanMessage(...)]
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if response.tool_calls:
            # Execute each tool and add results to messages
            for tool_call in response.tool_calls:
                tool = find_tool(tool_call["name"])
                result = tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=...))
        else:
            break  # No more tool calls, we have the final response
    
    # Route based on response content
    if "[EXECUTE_ANALYSIS]" in response.content:
        state["relevance_decision"] = "relevant"
        state["next_agent"] = "planner"
    else:
        state["relevance_decision"] = "chat_only"
        state["next_agent"] = "end"
        state["final_summary"] = response.content
    
    return state
```

##### System Prompt Logic

The interaction agent uses a decision rule:
- **Simple questions** (e.g., "What columns are in this dataset?") → Answer directly using tools
- **Analysis requests** (e.g., "Plot X vs Y") → Output `[EXECUTE_ANALYSIS]` token to trigger full workflow

---

#### Planner Agent: `backend/agents/planner.py`

**Purpose**: Creates a detailed analysis plan and instructions for the coding agent.

##### Key Function: `create_planner_agent(state) -> AnalysisState`

```python
def create_planner_agent(state: AnalysisState) -> AnalysisState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Slightly creative
    
    # Provide dataset context
    user_msg = HumanMessage(content=f"""
        User query: {state['user_query']}
        
        Dataset information:
        - Columns: {columns}
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols}
        
        Please create a detailed analysis plan and coding instructions.
    """)
    
    response = llm.invoke([system_msg, user_msg])
    
    # Store the plan (serves as prompt for coding agent)
    state["analysis_plan"] = response.content
    state["coding_prompt"] = response.content
    state["next_agent"] = "coding"
    
    return state
```

##### Output Format

The planner produces structured output:

```markdown
## Analysis Plan
1. Load and prepare data
2. Perform calculations/analysis
3. Create visualizations
4. Output results

## Coding Instructions
- Use column 'price' for the y-axis
- Calculate correlation using df.corr()
- Save plot as 'correlation_heatmap.png'
```

---

#### Coding Agent: `backend/agents/coding.py`

**Purpose**: Generates executable Python code and runs it in a sandbox.

##### Key Function: `create_coding_agent(state) -> AnalysisState`

```python
def create_coding_agent(state: AnalysisState) -> AnalysisState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Precise for code
    
    user_msg = HumanMessage(content=f"""
        Analysis Plan and Instructions:
        {state['coding_prompt']}
        
        Dataset columns available: {list(dataset.columns)}
        
        Please generate the Python code to accomplish this analysis.
        Return ONLY the code, no markdown formatting.
    """)
    
    response = llm.invoke([system_msg, user_msg])
    code = response.content
    
    # Clean up markdown formatting if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    
    state["generated_code"] = code
    
    # Execute in sandbox
    execution_result = execute_code_safely(code, dataset, output_dir)
    state["execution_result"] = execution_result
    state["next_agent"] = "summary"
    
    return state
```

##### Available Libraries

The coding agent can use:
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `seaborn` (as `sns`)
- `sklearn` (scikit-learn)
- `scipy`

---

#### Summary Agent: `backend/agents/summary.py`

**Purpose**: Interprets execution results and creates a user-friendly summary.

##### Key Function: `create_summary_agent(state) -> AnalysisState`

```python
def create_summary_agent(state: AnalysisState) -> AnalysisState:
    execution_result = state.get("execution_result", {})
    
    # Build context based on success/failure
    if execution_result.get("success"):
        context = f"""
            Code Execution: SUCCESS
            Standard Output: {execution_result.get('stdout')}
            Generated Visualizations: {execution_result.get('plots')}
        """
    else:
        context = f"""
            Code Execution: FAILED
            Error: {execution_result.get('error')}
        """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  # Natural language
    response = llm.invoke([system_msg, HumanMessage(content=context)])
    
    state["final_summary"] = response.content
    state["next_agent"] = "end"
    
    return state
```

---

### Tools

#### Dataset Tools: `backend/tools/dataset_tools.py`

Provides tools for the interaction agent to inspect datasets.

##### Available Tools

| Tool | Description |
|------|-------------|
| `get_dataset_info()` | Returns columns, dtypes, shape, missing values |
| `get_sample_rows(n)` | Returns first n rows as dictionaries |
| `search_dataset_columns(keyword)` | Finds columns containing keyword |
| `get_column_statistics(column)` | Returns detailed stats for a column |

##### Tool Binding Pattern

Tools need the dataset, but LangChain tools can't directly take DataFrame arguments. Solution: **closure-based factory**:

```python
def create_dataset_tools_for_agent(dataset: pd.DataFrame) -> list:
    # Create wrapper functions that capture dataset in closure
    def _get_dataset_info() -> dict:
        return get_dataset_info.func(dataset)  # Call original with captured dataset
    
    # Convert to LangChain tools
    tools = [
        StructuredTool.from_function(
            func=_get_dataset_info,
            name="get_dataset_info",
            description="Get comprehensive dataset information..."
        ),
        # ... other tools
    ]
    return tools
```

---

#### Execution Tools: `backend/tools/execution_tools.py`

##### `execute_code_safely(code, dataset, output_dir) -> dict`

Runs generated Python code in a restricted sandbox:

```python
def execute_code_safely(code: str, dataset: pd.DataFrame, output_dir: str) -> dict:
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "error": None,
        "plots": [],
        "variables": {}
    }
    
    # Restricted globals — only safe builtins + data science libraries
    safe_globals = {
        '__builtins__': {
            'print': print, 'len': len, 'range': range,
            'list': list, 'dict': dict, 'str': str,
            # ... other safe builtins
        },
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'df': dataset.copy(),      # Provide dataset as 'df'
        'dataset': dataset.copy(), # Also as 'dataset'
    }
    
    # Capture stdout/stderr
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        exec(code, safe_globals, local_vars)
        
        # Save any matplotlib figures
        for fig in plt.get_fignums():
            fig.savefig(os.path.join(output_dir, f"plot_{uuid}.png"))
        plt.close('all')
    
    result["success"] = True
    result["stdout"] = stdout_buffer.getvalue()
    result["plots"] = [list of saved filenames]
    
    return result
```

---

### Telemetry Configuration: `backend/telemetry/config.py`

Sets up tracing via LangSmith and the MAS backbone.

##### Key Functions

| Function | Description |
|----------|-------------|
| `get_langsmith_config()` | Returns LangSmith project name and API key from env |
| `get_mas_backbone_handler()` | Creates MASCallbackHandler and EventEmitter |
| `get_callbacks(session_id)` | Returns list of callback handlers for workflow |
| `get_emitter()` | Returns current EventEmitter for manual events |
| `setup_telemetry()` | Called at startup to initialize tracing |

##### Module-Level State

```python
_current_emitter: EventEmitter | None = None
_current_execution_id: str | None = None
_current_trace_id: str | None = None
_current_sink: FileSink | None = None
```

This allows agents to access the emitter via `get_emitter()` for manual event emission.

---

## How the Callback System Works

### What Are Callbacks?

LangChain/LangGraph callbacks are **hooks into the execution lifecycle**. They receive events like:
- Chain/agent started/ended
- LLM call started/ended
- Tool call started/ended
- Errors occurred

### Callback Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          LangChain Execution                             │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │ Chain Start │───▶│  LLM Call   │───▶│ Tool Call   │                   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                   │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Callback Handlers                             │    │
│  │  ┌────────────────────┐    ┌────────────────────────────────┐   │    │
│  │  │ LangSmith Handler  │    │   MASCallbackHandler            │   │    │
│  │  │ (built-in tracing) │    │   (custom backbone adapter)     │   │    │
│  │  └────────────────────┘    └────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│                     ┌──────────────────────────────┐                     │
│                     │  FileSink                     │                     │
│                     │  outputs/traces/trace_events │                     │
│                     │  .jsonl                       │                     │
│                     └──────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### MASCallbackHandler

Located in `backbone/adapters/langchain_callback.py`, this translates LangChain events into our trace event format:

```python
class MASCallbackHandler(BaseCallbackHandler):
    """
    Event Mapping:
    - on_chain_start  → agent_input event
    - on_chain_end    → agent_output event
    - on_llm_start    → tool_call (phase=start, tool_name="llm.generate")
    - on_llm_end      → tool_call (phase=end)
    - on_tool_start   → tool_call (phase=start)
    - on_tool_end     → tool_call (phase=end)
    - on_chain_error  → error event
    """
    
    def on_chain_start(self, serialized, inputs, *, run_id, metadata, **kwargs):
        # Extract agent_id from metadata
        agent_id = metadata.get("agent", "unknown")
        
        # Emit agent_input event
        self.emitter.emit(
            EventType.agent_input,
            agent_id,
            payload={"input": inputs, "chain_name": serialized.get("name")}
        )
    
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        # Emit tool_call start for LLM
        self.emitter.emit_tool_call(
            agent_id,
            tool_name="llm.generate",
            phase="start",
            tool_input={"prompts": prompts}
        )
```

### Manual Event Emission

For events that callbacks can't capture (like agent-to-agent messages), use the `EventEmitter`:

```python
from backend.telemetry.config import get_emitter

emitter = get_emitter()
if emitter:
    emitter.emit_message(
        from_agent="interaction",
        to_agent="planner",
        summary="Query requires analysis"
    )
```

### Passing Callbacks Through the Workflow

Callbacks are passed at multiple levels:

```python
# 1. Stored in state
initial_state = {
    "callbacks": get_callbacks(session_id),
    # ...
}

# 2. Passed to LLM initialization
llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=state.get("callbacks", []),
    metadata={"agent": "interaction"}  # Identifies agent in traces
)

# 3. Passed to invoke calls
response = llm.invoke(messages, config={"callbacks": callbacks})

# 4. Passed to workflow execution
final_state = app.invoke(initial_state, config={"callbacks": callbacks})
```

---

## Execution Flow

Here's what happens when a user runs an analysis:

### 1. Initialization Phase

```
User runs: python main.py --dataset data.csv --query "Plot correlation heatmap"
    │
    ▼
main.py
    │
    ├── Parse command-line arguments
    ├── Initialize telemetry (LangSmith + MAS backbone)
    ├── Load dataset into pandas DataFrame
    └── Call run_analysis_workflow()
```

### 2. Analysis Phase

```
run_analysis_workflow()
    │
    ├── Extract dataset metadata
    ├── Create callbacks (LangSmith + MAS backbone)
    ├── Initialize state dictionary
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Workflow                              │
│                                                                         │
│  ┌────────────────┐                                                     │
│  │  INTERACTION   │ ← Entry point                                       │
│  │    Agent       │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          │ Uses tools: get_dataset_info(), get_sample_rows()            │
│          │ Decides: Query requires analysis → output [EXECUTE_ANALYSIS] │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐                                                     │
│  │    PLANNER     │                                                     │
│  │    Agent       │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          │ Creates: Analysis plan with step-by-step instructions        │
│          │ Output: coding_prompt for next agent                         │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐                                                     │
│  │    CODING      │                                                     │
│  │    Agent       │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          │ Generates: Python code using pandas, matplotlib, seaborn     │
│          │ Executes: In sandbox via execute_code_safely()               │
│          │ Saves: Plots to outputs/ directory                           │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────┐                                                     │
│  │    SUMMARY     │                                                     │
│  │    Agent       │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          │ Interprets: stdout, generated plots, any errors              │
│          │ Produces: User-friendly summary of findings                  │
│          │                                                              │
│          ▼                                                              │
│        END                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Return result dictionary
    ├── success: true
    ├── final_summary: "The correlation analysis reveals..."
    ├── generated_code: "import pandas as pd..."
    └── execution_result: {plots: ["plot_abc123_0.png"]}
```

### 3. Trace Output

Throughout execution, events are written to `outputs/traces/<trace_id>/trace_events.jsonl`:

```json
{"event_id": "evt_...", "event_type": "agent_input", "agent_id": "interaction", "payload": {...}}
{"event_id": "evt_...", "event_type": "tool_call", "agent_id": "interaction", "payload": {"tool_name": "llm.generate", "phase": "start"}}
{"event_id": "evt_...", "event_type": "tool_call", "agent_id": "interaction", "payload": {"tool_name": "get_dataset_info", "phase": "start"}}
{"event_id": "evt_...", "event_type": "agent_message", "agent_id": "interaction", "payload": {"to_agent_id": "planner"}}
...
```

---

## Running the System

### Prerequisites

```bash
# Install dependencies
cd tracee/sample_mas
uv pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your_key

# Optional: Enable LangSmith tracing
export LANGSMITH_API_KEY=your_langsmith_key
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=data-analysis-agents
```

### Run with Sample Data

```bash
# Interactive mode
python main.py --sample

# Single query
python main.py --sample --query "Create a correlation heatmap of numeric columns"
```

### Run with Your Own Data

```bash
# Interactive mode
python main.py --dataset your_data.csv

# Single query
python main.py --dataset your_data.csv --query "Plot the distribution of the age column"
```

### Run Tests

```bash
python test_system.py
```

---

## Summary

This Multi-Agent System demonstrates:

1. **LangGraph for orchestration**: Stateful graph-based workflow with conditional routing
2. **Specialized agents**: Each agent has a focused responsibility (interaction, planning, coding, summarization)
3. **Tool integration**: Agents can use tools to inspect data and make informed decisions
4. **Safe code execution**: Generated code runs in a sandboxed environment
5. **Unified tracing**: Callback system provides observability via LangSmith and custom backbone
6. **Clean separation**: State schema, agents, tools, and workflow are in separate modules
7. **Simple CLI**: Easy-to-use command-line interface for terminal-based execution

This architecture is extensible — you can add more agents, tools, or integrate additional tracing backends by following the established patterns.
