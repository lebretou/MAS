# Data Analysis Multi-Agent System

A lightweight LangGraph workflow for automated data analysis with four focused agents: interaction, planner, coding, and summary. Designed for terminal-based execution.

## Features
- Load CSV, Excel, JSON, or Parquet files directly from the command line
- Interactive mode for exploratory analysis with multiple queries
- Single-query mode for scripting and automation
- Automatic query routing: simple dataset questions stay in chat, deeper requests trigger the full workflow
- Generated code runs in a sandboxed environment and saves plots to `outputs/`
- Full tracing via LangSmith and MAS backbone

## Setup

1. Install dependencies (uv recommended):
   ```bash
   cd tracee/sample_mas
   uv pip install -r requirements.txt
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI key (environment variable or `.env`):
   ```bash
   export OPENAI_API_KEY=your_key
   ```

3. (Optional) Enable LangSmith tracing:
   ```bash
   export LANGSMITH_API_KEY=your_langsmith_key
   export LANGSMITH_TRACING=true
   export LANGSMITH_PROJECT=data-analysis-agents
   ```

## Usage

### Interactive Mode
Start an interactive session to run multiple queries:
```bash
python main.py --dataset your_data.csv
```

### Single Query Mode
Run a single analysis and exit:
```bash
python main.py --dataset your_data.csv --query "Plot the distribution of the age column"
```

### Use Sample Data
Test with the included sample dataset:
```bash
python main.py --sample --query "Create a correlation heatmap"
```

### Full Options
```
usage: main.py [-h] [--dataset DATASET] [--query QUERY] [--sample] [--session-id SESSION_ID]

Options:
  --dataset, -d    Path to the dataset file (CSV, Excel, JSON, or Parquet)
  --query, -q      Analysis query (if not provided, runs in interactive mode)
  --sample, -s     Use sample_data.csv for testing
  --session-id     Session ID for tracing (default: cli)
```

## Run Tests
```bash
python test_system.py
```

## Project Layout
```
backend/        Agents, workflow, tools, state, telemetry
outputs/        Generated plots and trace artifacts
uploads/        (Legacy) Upload directory
sample_data.csv Demo dataset for testing
test_system.py  Basic workflow checks
main.py         CLI entry point
```

## Tracing & Telemetry
- **LangSmith**: Enabled when `LANGSMITH_API_KEY` is set. View traces at https://smith.langchain.com/
- **MAS Backbone**: Local traces are written to `outputs/traces/<trace_id>/trace_events.jsonl`

## Notes
- Code execution is sandboxed but allows standard imports needed for pandas/matplotlib workflows
- Plots are saved with unique filenames in `outputs/`
- Keep dataset sizes reasonable for in-memory processing
