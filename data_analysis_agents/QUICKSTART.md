# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd data_analysis_agents
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls_...
LANGSMITH_PROJECT=data-analysis-agents
LANGSMITH_TRACING=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Generate Sample Data

```bash
python sample_data.py
```

This creates three test CSV files.

### 4. Start the Server

```bash
./start_server.sh
```

Or directly:

```bash
python -m backend.main
```

### 5. Open the Web Interface

Navigate to: **http://localhost:8000/app/**

### 6. Test the System

1. **Upload a dataset** (use `sample_data_1.csv`)
2. **Try a simple query**: "What columns are in this dataset?"
3. **Try an analysis query**: "Plot correlation between all numeric variables"
4. **Check telemetry**:
   - LangSmith: https://smith.langchain.com/
   - Langfuse: https://cloud.langfuse.com/

## Test Queries

### Simple Queries (Interaction Agent Only)
- "What columns are in this dataset?"
- "How many rows are there?"
- "Show me the data types"

### Analysis Queries (Full Workflow)
- "Create a histogram of the age column"
- "Plot correlation heatmap for numeric variables"
- "Show scatter plot of income vs score"

### Complex Queries
- "Run linear regression and show results with plots"
- "Analyze distributions of all numeric columns"
- "Compare categories using box plots"

## Verify Telemetry

After running queries, check:

### LangSmith
1. Go to project "data-analysis-agents"
2. Click on latest trace
3. See: Agent sequence → Tool calls → LLM prompts

### Langfuse
1. Navigate to Traces
2. Click on latest trace
3. Explore: Generations → Spans → Metrics

## Troubleshooting

**Server won't start:**
- Check Python version (3.10+)
- Verify all dependencies installed
- Check port 8000 is available

**No telemetry showing:**
- Verify API keys in `.env`
- Check internet connection
- Wait a few seconds for traces to appear

**Code execution fails:**
- Check matplotlib backend is 'Agg'
- Verify pandas/numpy installed
- Check error in Summary section

## Next Steps

1. Try different datasets
2. Experiment with complex queries
3. Explore telemetry dashboards
4. Modify agents to test different behaviors
5. Add custom tools

## Support

See full documentation in `README.md`
