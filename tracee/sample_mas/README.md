# Data Analysis Multi-Agent System

A lightweight LangGraph workflow for automated data analysis with four focused agents: interaction, planner, coding, and summary. Telemetry integrations were removed to keep the stack simple.

## Features
- Upload CSV, Excel, or JSON files through the FastAPI backend.
- Automatic query routing: simple dataset questions stay in chat, deeper requests trigger the full workflow.
- Generated code runs in a sandboxed environment and saves plots to `outputs/`.
- Minimal dependencies and no external tracing services.

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

## Run the API
```bash
python -m backend.main
```
Visit `http://localhost:8000/app/` for the minimal web UI.

## Tests
```bash
python test_system.py
```

## Project layout
```
backend/        FastAPI app, agents, workflow, tools, state
frontend/       Static UI served at /app
uploads/        Uploaded datasets
outputs/        Generated plots and artifacts
sample_data.py  Helper to create demo data
test_system.py  Basic workflow checks
```

## Notes
- Code execution is sandboxed but now allows standard imports needed for pandas/matplotlib workflows.
- Plots are saved with unique filenames in `outputs/`; they are served via `GET /outputs/{filename}`.
- Keep dataset sizes reasonable for in-memory processing.
- LangSmith tracing is enabled when `LANGSMITH_API_KEY` is set; MAS backbone emits local traces under `outputs/traces/<trace_id>/trace_events.jsonl`.

