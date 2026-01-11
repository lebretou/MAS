# Data Analysis Multi-Agent System

A LangGraph-based multi-agent system for automated data analysis, designed to demonstrate LangSmith and Langfuse telemetry capabilities.

## 🎯 Purpose

This system is built specifically to test and explore telemetry tools (LangSmith and Langfuse) through a practical multi-agent application. It showcases:

- **Agent Execution Traces**: See how each agent processes information
- **Tool Usage Tracking**: Monitor when and how agents use tools
- **Performance Metrics**: Track latency, token usage, and execution flow
- **Decision Points**: Observe conditional routing between agents

## 🏗️ Architecture

The system consists of 4 specialized agents orchestrated by LangGraph:

```
User Query → Interaction Agent → Planner Agent → Coding Agent → Summary Agent
                    ↓                                                    ↓
              Chat Response                                         Final Results
```

### Agents

1. **Interaction Agent**
   - Validates query relevance to the dataset
   - Uses tools to inspect dataset structure
   - Decides: chat response OR proceed to analysis

2. **Planner Agent**
   - Creates detailed analysis plan
   - Suggests appropriate visualizations
   - Generates coding instructions

3. **Coding Agent**
   - Writes executable Python code
   - Validates code safety
   - Executes code in sandboxed environment

4. **Summary Agent**
   - Interprets execution results
   - Creates user-friendly summary
   - Highlights key findings

## 🚀 Setup

### 1. Install Dependencies

Using `uv` (recommended):

```bash
cd data_analysis_agents
uv pip install -r requirements.txt
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=data-analysis-agents
LANGSMITH_TRACING=true

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **LangSmith**: https://smith.langchain.com/
- **Langfuse**: https://cloud.langfuse.com/

## 📊 Running the Application

### Start the Server

From the `data_analysis_agents` directory:

```bash
python -m backend.main
```

Or with uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### Access the Web Interface

Open your browser and navigate to:

```
http://localhost:8000/app/
```

## 🧪 Testing the System

### Sample Queries

1. **Simple Query** (Interaction Agent only):
   - "What columns are in this dataset?"
   - "How many rows does this dataset have?"
   - "Show me the first few rows"

2. **Analysis Query** (Full workflow):
   - "Plot the correlation between all numeric variables"
   - "Create a histogram showing the distribution of [column_name]"
   - "Perform a scatter plot of X vs Y"

3. **Complex Query** (Full workflow with multiple outputs):
   - "Run a linear regression of Y on X1, X2, X3 and visualize the results"
   - "Analyze the relationship between all variables and create a correlation heatmap"
   - "Show the distribution of each numeric column with box plots"

### Create Sample Dataset

Create a CSV file with sample data:

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n = 100

df = pd.DataFrame({
    'age': np.random.randint(20, 70, n),
    'income': np.random.normal(50000, 15000, n),
    'score': np.random.uniform(0, 100, n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'satisfaction': np.random.randint(1, 6, n)
})

df.to_csv('sample_data.csv', index=False)
```

## 📈 Viewing Telemetry

### LangSmith

1. Go to https://smith.langchain.com/
2. Select your project: `data-analysis-agents`
3. View traces showing:
   - Agent execution order
   - Tool invocations
   - LLM prompts and responses
   - Execution times

### Langfuse

1. Go to https://cloud.langfuse.com/
2. Navigate to your project
3. Explore:
   - Traces tab: Complete workflow execution
   - Generations tab: LLM calls
   - Metrics: Token usage, latency

## 🔧 API Endpoints

### POST /upload
Upload a dataset (CSV, Excel, or JSON)

**Response:**
```json
{
  "session_id": "uuid",
  "filename": "data.csv",
  "shape": {"rows": 100, "columns": 5},
  "columns": ["col1", "col2", "..."]
}
```

### POST /analyze
Run analysis on uploaded dataset

**Request:**
```json
{
  "session_id": "uuid",
  "query": "Plot correlation between X and Y"
}
```

**Response:**
```json
{
  "success": true,
  "summary": "Analysis summary...",
  "plots": ["plot_123.png"],
  "code": "import pandas as pd..."
}
```

### GET /outputs/{filename}
Retrieve generated plots

### GET /results/{session_id}
Get session information

### DELETE /session/{session_id}
Delete session and associated files

## 🛠️ Development

### Project Structure

```
data_analysis_agents/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── agents/              # Agent implementations
│   │   ├── interaction.py
│   │   ├── planner.py
│   │   ├── coding.py
│   │   └── summary.py
│   ├── tools/               # Tool definitions
│   │   ├── dataset_tools.py
│   │   ├── validation_tools.py
│   │   └── execution_tools.py
│   ├── graph/               # LangGraph workflow
│   │   └── workflow.py
│   ├── state/               # State schema
│   │   └── schema.py
│   └── telemetry/           # Telemetry configuration
│       └── config.py
├── frontend/
│   └── index.html           # Web interface
├── uploads/                 # Uploaded datasets
├── outputs/                 # Generated plots
└── requirements.txt
```

### Adding New Agents

1. Create agent file in `backend/agents/`
2. Define agent function that takes and returns `AnalysisState`
3. Add agent node to workflow in `backend/graph/workflow.py`
4. Add routing logic if needed

### Adding New Tools

1. Create tool in appropriate file in `backend/tools/`
2. Use `@tool` decorator
3. Add to agent's tool list in agent file

## 🔍 Telemetry Features Demonstrated

### Traces
- Complete workflow execution from user query to final summary
- Agent-to-agent transitions
- Conditional routing decisions

### Tool Calls
- Dataset inspection (Interaction Agent)
- Plan validation (Planner Agent)
- Code validation (Coding Agent)
- Each tool call is tracked with inputs/outputs

### LLM Generations
- All prompts sent to the LLM
- Model responses
- Token usage per call
- Temperature and model settings

### Metadata
- Agent type (interaction, planner, coding, summary)
- Has tools flag
- Execution order
- Success/failure status

### Performance Metrics
- Latency per agent
- Total workflow execution time
- Token usage breakdown
- Tool execution time

## 🐛 Troubleshooting

### "Module not found" errors
- Ensure you're running from the `data_analysis_agents` directory
- Check that all dependencies are installed

### Telemetry not showing
- Verify API keys in `.env` file
- Check that `LANGSMITH_TRACING=true`
- Ensure internet connectivity for cloud services

### Code execution fails
- Check that matplotlib is using 'Agg' backend (non-interactive)
- Verify pandas and numpy are installed
- Check logs for specific error messages

### Port already in use
- Change port: `uvicorn backend.main:app --port 8001`
- Or kill process using port 8000

## 📝 Notes

- The system uses GPT-4o-mini for cost-effectiveness
- Code execution is sandboxed with restricted imports
- Plots are saved to `outputs/` directory
- Sessions are stored in memory (use database for production)

## 🎓 Learning Points

This project demonstrates:

1. **LangGraph workflows** with conditional routing
2. **Agent specialization** for different tasks
3. **Tool use** for enhanced agent capabilities
4. **State management** across multiple agents
5. **Telemetry integration** for observability
6. **Safe code execution** with sandboxing
7. **Web interface** for agent interaction

## 📄 License

This is a demonstration project for learning telemetry tools.

## 🙏 Acknowledgments

Built with:
- LangChain / LangGraph
- LangSmith
- Langfuse
- FastAPI
- OpenAI GPT-4
