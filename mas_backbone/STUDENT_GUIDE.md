# MAS Backbone: A Beginner's Guide

Welcome! This guide will help you understand the MAS (Multi-Agent System) Backbone project. Don't worry if you don't have programming experience - we'll explain everything step by step using everyday analogies.

---

## Table of Contents

1. [What is this project about?](#what-is-this-project-about)
2. [The Big Picture: Why do we need this?](#the-big-picture-why-do-we-need-this)
3. [Core Concepts Explained Simply](#core-concepts-explained-simply)
4. [Folder Structure: What's in each folder?](#folder-structure-whats-in-each-folder)
5. [The Three Main Building Blocks](#the-three-main-building-blocks)
6. [How Agents Talk to Each Other](#how-agents-talk-to-each-other)
7. [A Real Example: The Planner and Executor Story](#a-real-example-the-planner-and-executor-story)
8. [Glossary: Terms You'll See](#glossary-terms-youll-see)

---

## What is this project about?

Imagine you're watching a group project where several students (let's call them "agents") work together to complete a task. Each student has a specific role:

- One student **plans** what needs to be done
- Another student **executes** the plan
- They **communicate** with each other by passing notes

This project is like a **recording system** that captures everything that happens during this group work:
- What instructions each student received
- What decisions they made
- What messages they sent to each other
- What went wrong (if anything)

**In the AI world**, these "students" are AI agents powered by Large Language Models (like ChatGPT). This backbone helps researchers understand how these AI agents work together, communicate, and sometimes fail.

---

## The Big Picture: Why do we need this?

### The Problem

When multiple AI agents work together:
- It's hard to see **who said what to whom**
- It's difficult to trace **where things went wrong**
- The instructions (prompts) given to each agent can get messy and hard to track

### The Solution

This backbone provides a structured way to:

1. **Write and organize prompts** (the instructions we give to AI agents)
2. **Record what actually happened** when the system ran
3. **Track communication** between agents
4. **Find failures** and understand why they occurred

Think of it like having a detailed transcript of a meeting, where every word, decision, and handoff between people is recorded.

---

## Core Concepts Explained Simply

### Concept 1: Prompts are like Recipe Cards

When you give an AI agent instructions, that's called a **prompt**. 

Think of a prompt like a recipe card for a chef:
- **Role**: "You are a pastry chef"
- **Goal**: "Bake a chocolate cake"
- **Constraints**: "Use only organic ingredients"
- **Examples**: "Here's what a good cake looks like..."

In this project, we break prompts into these components so they're organized and reusable.

### Concept 2: Executions are like Security Camera Footage

When the AI system actually runs, we call that an **execution**. 

Like security camera footage, an execution record captures:
- Exactly what instructions were sent
- What model (like GPT-4) was used
- What the input was
- What environment it ran in (testing vs. production)

The key rule: **Always save the actual text that was sent**, not just a reference to it. This way, you can always reproduce what happened.

### Concept 3: Events are like Play-by-Play Commentary

During an execution, many things happen. Each thing that happens is an **event**.

Think of a sports commentator describing a basketball game:
- "Player A receives the ball" → `agent_input`
- "Player A decides to pass" → `agent_decision`
- "Player A passes to Player B" → `agent_message`
- "Player B catches the ball" → `agent_input`
- "Referee checks for foul" → `contract_validation`
- "Foul called!" → `error`

Each event captures **who did what, when, and with what result**.

---

## Folder Structure: What's in each folder?

```
mas_backbone/
├── models/          # 📋 Definitions of our data structures
├── adapters/        # 🔌 Connectors to other systems (like LangChain)
├── utils/           # 🔧 Helper tools (generating IDs, timestamps)
├── analysis/        # 📊 Tools to analyze what happened
├── scripts/         # 🎬 Runnable examples and demos
├── tests/           # ✅ Automated tests to make sure everything works
└── pyproject.toml   # ⚙️ Project configuration
```

Let's explore each one:

### `models/` - The Data Blueprints

This folder contains **definitions** of our main concepts. Think of these like forms or templates.

| File | What it defines | Real-world analogy |
|------|----------------|-------------------|
| `prompt_artifact.py` | Structure for prompts | A recipe card template |
| `execution_record.py` | Structure for recording runs | A flight recorder |
| `trace_event.py` | Structure for individual events | A logbook entry form |

### `adapters/` - The Translators

This folder helps our system work with other tools.

| File | What it does | Real-world analogy |
|------|-------------|-------------------|
| `event_api.py` | Provides easy ways to record events | A microphone for the commentator |
| `langchain_callback.py` | Connects to LangChain framework | A translator between two languages |

### `utils/` - The Toolbox

Small helper functions that do one thing well.

| File | What it provides |
|------|-----------------|
| `identifiers.py` | Generates unique IDs and timestamps |

### `analysis/` - The Detective Tools

Tools to understand what happened after everything is recorded.

| File | What it does | Real-world analogy |
|------|-------------|-------------------|
| `trace_summary.py` | Summarizes a trace, finds patterns | A detective reviewing evidence |

### `scripts/` - The Demos

Runnable examples that show how everything works together.

| File | What it does |
|------|-------------|
| `generate_dummy_run.py` | Creates fake but realistic example data |

### `tests/` - The Quality Checkers

Automated tests that make sure the code works correctly.

---

## The Three Main Building Blocks

### 1. PromptVersion (The Recipe Card)

A `PromptVersion` is a saved, versioned prompt. Key parts:

```
PromptVersion
├── prompt_id: "prompt-planner-001"     # Unique identifier
├── version_id: "v1.0.0"                # Version number
├── name: "Planner Agent Prompt"        # Human-readable name
├── components:                         # The actual content, broken into parts
│   ├── role: "You are a planning agent..."
│   ├── goal: "Create step-by-step plans..."
│   ├── constraints: "Plans must have at most 5 steps..."
│   └── io_rules: "Output must be valid JSON..."
└── created_at: "2024-01-15T10:30:00Z"  # When it was created
```

**Why version prompts?** Just like software has versions (v1.0, v2.0), prompts evolve. By versioning, you can:
- Go back to an older version if a new one doesn't work
- Compare which version performs better
- Know exactly what prompt was used in any past run

### 2. ExecutionRecord (The Flight Recorder)

An `ExecutionRecord` captures everything about a single run:

```
ExecutionRecord
├── execution_id: "abc-123"           # Unique ID for this run
├── trace_id: "xyz-789"               # Groups related events together
├── origin: "playground"              # Where it ran (testing vs. production)
├── llm_config:                       # What AI model was used
│   ├── provider: "openai"
│   ├── model_name: "gpt-4"
│   └── temperature: 0.0
├── resolved_prompt_text: "..."       # THE ACTUAL TEXT SENT (very important!)
├── prompt_refs: [...]                # Links back to the PromptVersion used
└── contract_refs: [...]              # What validation rules were applied
```

**The Golden Rule**: Always save `resolved_prompt_text` - the actual text that was sent to the AI. Never just save a reference. This ensures you can always reproduce the exact run.

### 3. TraceEvent (The Logbook Entry)

A `TraceEvent` records one thing that happened:

```
TraceEvent
├── event_id: "evt-001"               # Unique ID for this event
├── trace_id: "xyz-789"               # Which trace this belongs to
├── execution_id: "abc-123"           # Which execution this is part of
├── timestamp: "2024-01-15T10:30:01Z" # When it happened
├── sequence: 0                       # Order within the trace (0, 1, 2, ...)
├── event_type: "agent_message"       # What kind of event
├── agent_id: "planner"               # Which agent did this
├── payload: {...}                    # The actual data/content
└── refs: {...}                       # Extra context/metadata
```

#### Event Types Explained

| Event Type | What it means | Example |
|------------|--------------|---------|
| `agent_input` | Agent receives something | Planner gets the user's request |
| `agent_output` | Agent produces something | Planner outputs a plan |
| `agent_message` | Agent sends to another agent | Planner sends plan to Executor |
| `agent_decision` | Agent makes a choice | Planner decides which approach to use |
| `tool_call` | Agent uses a tool | Agent calls a search function |
| `contract_validation` | Checking if output is valid | Verifying the plan has correct format |
| `error` | Something went wrong | Plan was missing required fields |

---

## How Agents Talk to Each Other

### The Communication Pattern

```
┌─────────────┐                      ┌─────────────┐
│   Planner   │                      │  Executor   │
│   Agent     │                      │   Agent     │
└─────────────┘                      └─────────────┘
       │                                    │
       │  1. Receives user request          │
       │◄───────────────────────────────────│
       │                                    │
       │  2. Creates a plan                 │
       │                                    │
       │  3. Sends plan to executor         │
       │────────────────────────────────────►│
       │                                    │
       │                      4. Validates plan
       │                                    │
       │                      5. If valid: executes
       │                         If invalid: reports error
       │                                    │
```

### Messages Have Rules (Contracts)

When Agent A sends something to Agent B, there might be **contracts** - rules about what the message should look like.

Example contract: "A plan must have a `steps` array, and each step must have an `action` and a `tool`"

**Valid plan:**
```json
{
  "steps": [
    {"action": "load_data", "tool": "csv_reader"},
    {"action": "analyze", "tool": "pandas"}
  ]
}
```

**Invalid plan (missing `tool`):**
```json
{
  "steps": [
    {"action": "load_data"},
    {"action": "analyze"}
  ]
}
```

The backbone records when these validations pass or fail.

---

## A Real Example: The Planner and Executor Story

Let's walk through the example that's included in `scripts/generate_dummy_run.py`.

### The Scenario

A user asks: **"Analyze sales data and create a report"**

### What Happens (Success Case)

| Step | Event Type | Agent | What Happens |
|------|-----------|-------|--------------|
| 1 | `agent_input` | Planner | Receives the user's request |
| 2 | `tool_call` (start) | Planner | Starts calling the LLM |
| 3 | `tool_call` (end) | Planner | LLM returns a plan |
| 4 | `agent_decision` | Planner | Decides the plan looks good |
| 5 | `agent_output` | Planner | Outputs the plan |
| 6 | `agent_message` | Planner | Sends plan to Executor |
| 7 | `agent_input` | Executor | Receives the plan |
| 8 | `contract_validation` | Executor | Checks plan format - **PASSES** |
| 9 | `agent_output` | Executor | Outputs success result |

### What Happens (Failure Case)

Same as above, but at step 8:

| Step | Event Type | Agent | What Happens |
|------|-----------|-------|--------------|
| 8 | `contract_validation` | Executor | Checks plan format - **FAILS** (missing `tool` field) |
| 9 | `error` | Executor | Reports schema error |

### The Summary

After all events are recorded, we can analyze them:

```
TraceSummary:
├── agents: ["planner", "executor"]
├── edges: [planner → executor (1 message)]
├── failures: 1 schema error
├── failed_contracts: plan-input-v1 failed 1 time
└── tool_usage: llm.generate called 1 time
```

This tells us:
- Two agents were involved
- Planner sent one message to Executor
- There was one failure
- The failure was a schema validation error

---

## Glossary: Terms You'll See

| Term | Simple Explanation |
|------|-------------------|
| **Agent** | An AI "worker" that does a specific task |
| **Prompt** | Instructions given to an AI agent |
| **Execution** | One complete run of the system |
| **Trace** | A collection of events from one execution |
| **Event** | One thing that happened (like a log entry) |
| **Contract** | Rules about what data should look like |
| **Validation** | Checking if data follows the rules |
| **LLM** | Large Language Model (like GPT-4, Claude) |
| **LangChain** | A popular framework for building AI applications |
| **Payload** | The actual data/content of an event |
| **Refs** | Extra references/metadata attached to an event |
| **Pydantic** | A Python library for defining data structures |
| **UUID** | A unique identifier (like a serial number) |
| **JSONL** | JSON Lines - a file format with one JSON object per line |

---

## Quick Start: Running the Example

If you want to try running the code yourself:

```bash
# 1. Navigate to the project
cd mas_backbone

# 2. Install dependencies
uv pip install -e "."

# 3. Run the example scenario generator
.venv/bin/python -c "import sys; sys.path.insert(0, '..'); from mas_backbone.scripts.generate_dummy_run import main; main()"

# 4. Look at the output
ls ../outputs/
```

This will create two folders (one for success, one for failure) containing:
- `execution_record.json` - The execution metadata
- `trace_events.jsonl` - All the events that happened
- `prompt_version.json` - The prompt that was used

---

## Questions?

If you have questions about any part of this guide or the codebase, feel free to ask! The best way to learn is to:

1. Read through this guide
2. Look at the generated output files
3. Try modifying `generate_dummy_run.py` to create your own scenarios
4. Run the tests to see how everything is verified

Happy exploring! 🎉
