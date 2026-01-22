# Data Analysis Multi-Agent System — Architecture Guide

This document provides an explanation of the MAS implementation in this folder, including a brief overview of LangChain and LangGraph, how to set up agents, explanations of each file, and how the callback/telemetry system works. It also covers the **MAS Backbone** — the underlying tracing infrastructure used across all agent systems.

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
10. [MAS Backbone](#mas-backbone)
    - [Backbone Overview](#backbone-overview)
    - [Backbone Directory Structure](#backbone-directory-structure)
    - [Core Concepts](#core-concepts)
    - [Data Models](#data-models)
    - [Event API and Sinks](#event-api-and-sinks)
    - [LangChain Callback Handler](#langchain-callback-handler)
    - [Trace Analysis](#trace-analysis)
    - [Utility Functions](#utility-functions)
    - [Design Patterns](#design-patterns)
    - [Integration Examples](#integration-examples)

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

## MAS Backbone

The **MAS Backbone** is the underlying tracing infrastructure that powers the observability in this multi-agent system. It provides data models, event emission APIs, and analysis utilities that are framework-agnostic and can be used with any LLM orchestration tool.

### Backbone Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUTURE UI LAYER                                   │
│                   (Playground, Trace Viewer, Evaluations)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAS BACKBONE                                        │
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
- **What data structures exist** (models)
- **How events get captured** (adapters)
- **How to reconstruct what happened** (analysis)

---

### Backbone Directory Structure

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
│   ├── event_api.py         # Manual event emission (EventEmitter)
│   ├── sinks.py             # Event sinks (EventSink, ListSink, FileSink)
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

### Core Concepts

#### 1. Pydantic Models as the Schema Layer

**Why Pydantic?**
- **Validation at construction time**: Invalid data fails immediately, not later in a pipeline
- **Serialization for free**: `.model_dump_json()` and `.model_validate_json()` handle JSON round-trips
- **Type hints are documentation**: The code is self-documenting for IDE autocompletion

Every model has `model_config = {"extra": "forbid"}` — this means if you pass an unexpected field, it errors. This catches typos and schema drift early.

#### 2. Protocol-Based Abstraction (EventSink)

```python
class EventSink(Protocol):
    def append(self, event: TraceEvent) -> None: ...
```

This is a **structural subtyping** pattern (duck typing with type hints). Any class with an `append(TraceEvent)` method satisfies this protocol. You don't need inheritance — just implement the method.

**Why this matters**:
- `ListSink` stores events in memory (for tests)
- `FileSink` appends to a JSONL file (for persistence)

#### 3. Callback Handler Pattern (LangChain Integration)

LangChain uses the **Observer pattern** via callbacks. When an LLM runs, chain executes, or tool is called, LangChain invokes registered callbacks.

`MASCallbackHandler` subscribes to these callbacks and translates them into `TraceEvent` objects. The key insight: **callbacks capture low-level execution events, NOT semantic agent communication**. That's why `emit_message()` is a manual API call.

#### 4. JSONL as Append-Only Log

Events are stored in JSONL (JSON Lines) format:
```
{"event_id": "...", "event_type": "agent_input", ...}
{"event_id": "...", "event_type": "tool_call", ...}
{"event_id": "...", "event_type": "agent_message", ...}
```

---

### Data Models

#### Prompt Artifact: `backbone/models/prompt_artifact.py`

**Purpose**: Defines the **authoring-side** data structures — what gets created in a Playground before execution.

##### PromptComponentType Enum

```python
class PromptComponentType(str, Enum):
    """Types of prompt sections that can be toggled on/off."""
    role = "role"                    # "You are a..."
    goal = "goal"                    # "Your objective is..."
    constraints = "constraints"      # "Do not..."
    io_rules = "io_rules"            # "Output format must be..."
    examples = "examples"            # Few-shot examples
    safety = "safety"                # Safety guardrails
    tool_instructions = "tool_instructions"  # How to use tools
```

##### PromptComponent Model

```python
class PromptComponent(BaseModel):
    type: PromptComponentType
    content: str
    enabled: bool = True  # Toggle components without deleting them
```

##### PromptVersion Model

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

#### Execution Record: `backbone/models/execution_record.py`

**Purpose**: Captures **what actually happened** during a single execution run.

##### ModelConfig Model

```python
class ModelConfig(BaseModel):
    """LLM configuration used for this run."""
    provider: str       # "openai", "anthropic", etc.
    model_name: str     # "gpt-4", "claude-3", etc.
    temperature: float
    max_tokens: int
    seed: int | None    # For reproducibility
```

##### Reference Models

```python
class PromptArtifactRef(BaseModel):
    """Reference to a versioned prompt artifact."""
    prompt_id: str
    version_id: str
    agent_id: str | None = None

class ContractRef(BaseModel):
    """Reference to a versioned contract."""
    contract_id: str
    contract_version: str
    agent_id: str | None = None
```

##### ExecutionRecord Model

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

#### Trace Event: `backbone/models/trace_event.py`

**Purpose**: Defines the **semantic event vocabulary** — what types of things can happen during execution.

##### EventType Enum

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

##### TraceEvent Model

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

##### Payload Validation Rules

Different event types have different required fields (enforced by `@model_validator`):

| Event Type | Required Payload Fields |
|------------|------------------------|
| `agent_message` | `to_agent_id`, (`message_summary` OR `payload_ref`) |
| `contract_validation` | `contract_id`, `contract_version`, `validation_result.is_valid`, `validation_result.errors` |
| `tool_call` | `tool_name`, `phase` (must be "start" or "end") |
| `error` | `error_type` (one of: schema, tool, model, infra, logic), `message` |

##### Valid Error Types

```python
ERROR_TYPES = {"schema", "tool", "model", "infra", "logic"}
```

---

### Event API and Sinks

The manual event emission API lives in `backbone/adapters/event_api.py`. Event sinks are defined in `backbone/adapters/sinks.py`.

#### EventSink Protocol

```python
class EventSink(Protocol):
    """Protocol for receiving trace events."""
    def append(self, event: TraceEvent) -> None:
        """Append an event to the sink."""
        ...
```

#### ListSink Implementation

```python
class ListSink:
    """Stores events in a list (for tests)."""
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
```

#### FileSink Implementation

```python
class FileSink:
    """Writes events to a JSONL file (for production)."""
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")
```

#### EventEmitter Class

The main API for emitting trace events:

```python
class EventEmitter:
    """Manual event emission API - shares sink with callback handler."""

    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        event_sink: EventSink,
    ) -> None:
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.event_sink = event_sink
        self._sequence = 0  # Internal counter for ordering
```

##### EventEmitter Methods

| Method | Creates Event Type | Key Parameters |
|--------|-------------------|----------------|
| `emit()` | Any | `event_type`, `agent_id`, `payload` |
| `emit_input()` | `agent_input` | `agent_id`, `input_data` |
| `emit_output()` | `agent_output` | `agent_id`, `output_data` |
| `emit_message()` | `agent_message` | `from_agent`, `to_agent`, `summary` |
| `emit_decision()` | `agent_decision` | `agent_id`, `decision`, `reasoning` |
| `emit_tool_call()` | `tool_call` | `agent_id`, `tool_name`, `phase` |
| `emit_validation()` | `contract_validation` | `agent_id`, `contract_id`, `is_valid`, `errors` |
| `emit_error()` | `error` | `agent_id`, `error_type`, `message` |

##### emit() Method Details

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
    """Emit a trace event with the given parameters."""
    event = TraceEvent(
        event_id=generate_event_id(),
        trace_id=self.trace_id,
        execution_id=self.execution_id,
        timestamp=utc_timestamp(),
        sequence=self._next_sequence(),
        event_type=event_type,
        agent_id=agent_id,
        span_id=span_id or generate_span_id(),
        parent_span_id=parent_span_id,
        refs=refs or {},
        payload=payload or {},
    )
    self.event_sink.append(event)
    return event
```

**Why return the event?** — To allow correlating start/end of a tool call:
```python
# Correlate start/end of a tool call
start_event = emitter.emit_tool_call("agent", "search", "start")
# ... later ...
emitter.emit_tool_call("agent", "search", "end", span_id=start_event.span_id)
```

---

### LangChain Callback Handler

Located in `backbone/adapters/langchain_callback.py`, this provides **automatic** event capture from LangChain/LangGraph execution.

#### MASCallbackHandler Class

```python
class MASCallbackHandler(BaseCallbackHandler):
    """Emits low-level execution events from LangChain/LangGraph.

    Does NOT infer agent-to-agent messages (use manual API for that).

    Callback Mapping:
    - on_chain_start  → agent_input (only if chain = agent node)
    - on_chain_end    → agent_output (only if chain = agent node)
    - on_llm_start    → tool_call (phase=start, tool_name="llm.generate")
    - on_llm_end      → tool_call (phase=end, tool_name="llm.generate")
    - on_tool_start   → tool_call (phase=start)
    - on_tool_end     → tool_call (phase=end)
    - on_chain_error  → error (classified)
    """
```

#### Internal State

The callback handler maintains internal state for correlating events:

```python
self._run_span_map: dict[str, str] = {}   # run_id → span_id
self._run_agent_map: dict[str, str] = {}  # run_id → agent_id
```

**Why track run_id → span_id?**
LangChain passes a `run_id` for each execution unit. To correlate start/end events (e.g., `on_llm_start` and `on_llm_end`), we need to use the **same span_id**. The map ensures this.

**Why track run_id → agent_id?**
Nested callbacks (e.g., `on_llm_end`) don't receive metadata. By storing the agent_id from `on_chain_start`, child callbacks can look it up via `parent_run_id`.

#### Key Callback Methods

##### on_chain_start

```python
def on_chain_start(
    self,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Handle chain start - emit agent_input if this is an agent node."""
    # Store agent ID for this run if provided
    if metadata and ("agent_id" in metadata or "agent" in metadata):
        agent_id = metadata.get("agent_id", metadata.get("agent", self.default_agent_id))
        self._run_agent_map[str(run_id)] = agent_id

    agent_id = self._get_agent_id(run_id, metadata)
    refs = self._make_refs(run_id, parent_run_id, metadata)
    # ... emit agent_input event
```

##### on_llm_start / on_llm_end

```python
def on_llm_start(self, serialized, prompts, *, run_id, ...):
    """Handle LLM start - emit tool_call with phase=start."""
    self.emitter.emit_tool_call(
        agent_id,
        tool_name="llm.generate",
        phase="start",
        tool_input={"prompts": prompts[:1]},  # truncate for brevity
        refs=refs,
        span_id=span_id,
        parent_span_id=parent_span,
    )

def on_llm_end(self, response, *, run_id, ...):
    """Handle LLM end - emit tool_call with phase=end."""
    output_text = response.generations[0][0].text[:200]  # truncated
    self.emitter.emit_tool_call(
        agent_id,
        tool_name="llm.generate",
        phase="end",
        tool_output={"text": output_text},
        span_id=span_id,  # Same span_id as start
    )
```

#### Error Classification

The `_classify_error()` function maps exceptions to error types:

```python
def _classify_error(error: BaseException) -> str:
    """Classify an error into one of the valid error types."""
    error_name = type(error).__name__.lower()

    if any(x in error_name for x in ["openai", "anthropic", "llm", "api", "rate"]):
        return "model"
    if any(x in error_name for x in ["tool", "function"]):
        return "tool"
    if any(x in error_name for x in ["timeout", "connection", "network", "http"]):
        return "infra"
    if any(x in error_name for x in ["validation", "schema", "parse", "json", "type"]):
        return "schema"
    return "logic"  # default
```

#### Key Design Decision

The callback handler **does NOT emit `agent_message`**. Why?

LangChain callbacks see execution events (chain started, LLM called, tool ran) but don't understand the **semantic relationship** between agents. The decision to hand off work from planner to executor is a domain concept — only the developer knows when that happens.

---

### Trace Analysis

Located in `backbone/analysis/trace_summary.py`, this provides **post-hoc analysis** — reconstructing what happened from a list of events.

#### Data Classes

```python
@dataclass
class AgentEdge:
    """Represents communication between two agents."""
    from_agent: str
    to_agent: str
    message_count: int

@dataclass
class FailedContract:
    """Represents a failed contract validation."""
    contract_id: str
    contract_version: str
    failure_count: int

@dataclass
class ToolUsage:
    """Represents tool usage statistics."""
    tool_name: str
    call_count: int
    avg_latency_ms: float | None = None
```

#### TraceSummary Class

```python
@dataclass
class TraceSummary:
    """Summary of a trace including agent communication and failures."""
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

#### trace_summary() Function

```python
def trace_summary(events: list[TraceEvent]) -> TraceSummary:
    """Reconstruct agent graph and detect failures from trace events.

    Args:
        events: List of TraceEvent objects from a single trace.

    Returns:
        TraceSummary with agent communication graph and failure information.
    """
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

### Utility Functions

Located in `backbone/utils/identifiers.py`, these provide consistent ID generation across the codebase.

```python
def generate_execution_id() -> str:
    """Generate a unique execution ID (UUID4)."""
    return str(uuid.uuid4())

def generate_trace_id() -> str:
    """Generate a unique trace ID (UUID4)."""
    return str(uuid.uuid4())

def generate_event_id() -> str:
    """Generate a unique event ID (UUID4)."""
    return str(uuid.uuid4())

def generate_span_id() -> str:
    """Generate a span ID (16-char hex string for OpenTelemetry compatibility)."""
    return uuid.uuid4().hex[:16]

def utc_timestamp() -> str:
    """Generate an ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
```

---

### Design Patterns

#### 1. Inversion of Control (Dependency Injection)

`EventEmitter` doesn't know where events go — it just calls `event_sink.append()`. The caller provides the sink:

```python
# For tests
sink = ListSink()
emitter = EventEmitter(exec_id, trace_id, sink)

# For production
sink = FileSink("/path/to/trace.jsonl")
emitter = EventEmitter(exec_id, trace_id, sink)
```

#### 2. Validation at the Boundary

All data validation happens in Pydantic models. By the time an object exists, it's guaranteed valid. No defensive checks scattered through the codebase.

#### 3. Event Sourcing Lite

Events are the source of truth. The `TraceSummary` is a **derived view** computed from events. If you need a different view (e.g., timeline, agent-centric), write a new function that reads the same events.

#### 4. Namespace Isolation (refs dict)

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

### Integration Examples

#### Creating a Prompt Version

```python
from backbone.models.prompt_artifact import (
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.utils.identifiers import utc_timestamp

# User builds a prompt in the UI
prompt_version = PromptVersion(
    prompt_id="my-planner-prompt",
    version_id="v1.0.0",
    name="My Planner Agent",
    components=[
        PromptComponent(type=PromptComponentType.role, content="You are a planning agent."),
        PromptComponent(type=PromptComponentType.goal, content="Create a step-by-step plan."),
        PromptComponent(type=PromptComponentType.constraints, content="Max 5 steps.", enabled=True),
        PromptComponent(type=PromptComponentType.examples, content="Example: ...", enabled=False),
    ],
    variables={"max_steps": "5"},
    created_at=utc_timestamp(),
)

# Save to storage
with open("prompts/my-planner-prompt-v1.json", "w") as f:
    f.write(prompt_version.model_dump_json(indent=2))
```

#### Setting Up Event Capture for a Run

```python
from backbone.adapters.event_api import EventEmitter
from backbone.adapters.sinks import FileSink
from backbone.adapters.langchain_callback import MASCallbackHandler
from backbone.utils.identifiers import generate_execution_id, generate_trace_id

# Generate IDs
execution_id = generate_execution_id()
trace_id = generate_trace_id()

# Create output directory
output_dir = Path(f"outputs/traces/{trace_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Set up event capture
sink = FileSink(output_dir / "trace_events.jsonl")
emitter = EventEmitter(execution_id, trace_id, sink)

# Create callback handler (shares the same sink)
callback = MASCallbackHandler(
    execution_id=execution_id,
    trace_id=trace_id,
    event_sink=sink,
    default_agent_id="planner",
)
```

#### Running LangChain with Tracing

```python
from langchain_openai import ChatOpenAI

# Resolve the prompt text (only enabled components)
resolved_prompt = "\n\n".join(
    c.content for c in prompt_version.components if c.enabled
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
```

#### Analyzing a Trace

```python
from backbone.analysis.trace_summary import trace_summary
from backbone.models.trace_event import TraceEvent

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

#### Dummy Scenario Generation

The `scripts/generate_dummy_run.py` script creates realistic test scenarios:

**Scenario A: Failure Case**
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

**Scenario B: Success Case**
```
Same as A, but plan includes 'tool' field
executor validates contract → PASSES
executor outputs results
```

---

## Summary

This architecture guide covers two interconnected systems:

### Sample MAS (Multi-Agent System)

A **Data Analysis Multi-Agent System** demonstrating:
1. **LangGraph for orchestration**: Stateful graph-based workflow with conditional routing
2. **Specialized agents**: Each agent has a focused responsibility (interaction, planning, coding, summarization)
3. **Tool integration**: Agents can use tools to inspect data and make informed decisions
4. **Safe code execution**: Generated code runs in a sandboxed environment
5. **Simple CLI**: Easy-to-use command-line interface for terminal-based execution

### MAS Backbone

The **underlying tracing infrastructure** providing:
1. **Pydantic models**: Strict validation for prompts, executions, and events
2. **Event emission API**: Manual (`EventEmitter`) and automatic (`MASCallbackHandler`)
3. **Pluggable sinks**: `ListSink` for tests, `FileSink` for production
4. **Trace analysis**: Reconstruct agent graphs and detect failures from events
5. **Framework-agnostic design**: Works with any LLM orchestration tool

### Key Insight: Separation of Concerns

- **Authoring** (PromptVersion) vs **Execution** (ExecutionRecord)
- **Automatic capture** (callbacks) vs **Semantic events** (manual emit)
- **Storage** (sinks) vs **Analysis** (summary)

This foundation enables future UIs (Playground, Trace Viewer, Evaluations) without changing core data models, and the architecture is extensible — you can add more agents, tools, or integrate additional tracing backends by following the established patterns.
