from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from backend.state.schema import AnalysisState
from backend.telemetry.config import get_emitter


PLANNER_SYSTEM_PROMPT = """You are a planning agent in a data analysis system. Your role is to:

1. Create a detailed, step-by-step analysis plan based on the user's query
2. Suggest appropriate visualizations using available tools
3. Write a comprehensive prompt for the coding agent

You have access to tools that let you:
- Validate analysis plans
- Suggest appropriate visualizations based on the query and dataset
- List available libraries

**Your output should include:**

1. **Analysis Plan**: A clear, numbered list of steps to accomplish the user's request
   - Step 1: Load and prepare data
   - Step 2: Perform calculations/analysis
   - Step 3: Create visualizations
   - Step 4: Output results

2. **Coding Instructions**: Detailed instructions for the coding agent, including:
   - Which columns to use
   - What calculations to perform
   - What plots to create
   - How to save figures (use plt.savefig('filename.png'))

**Important Guidelines:**
- Use only allowed libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns), sklearn, scipy
- The dataset is available as variable 'df' or 'dataset'
- Always save plots using plt.savefig() before creating new figures
- Include print statements for key results
- Be specific about column names and operations

Format your response as:

## analysis plan
[Numbered steps]

## coding instructions
[Detailed prompt for the coding agent]
"""


def create_planner_agent(state: AnalysisState) -> AnalysisState:
    """Planner agent that creates analysis plan and coding instructions.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with planning results
    """
    # get dataset info
    dataset = state["dataset"]
    dataset_info = state.get("dataset_info", {})
    callbacks = state.get("callbacks", [])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        callbacks=callbacks,
        metadata={"agent": "planner", "has_tools": False}
    )
    
    # prepare dataset info
    columns = dataset_info.get("columns", list(dataset.columns))
    shape = dataset_info.get("shape", {"rows": dataset.shape[0], "columns": dataset.shape[1]})
    numeric_cols = dataset_info.get("numeric_columns", list(dataset.select_dtypes(include=['number']).columns))
    categorical_cols = dataset_info.get("categorical_columns", list(dataset.select_dtypes(include=['object', 'category']).columns))
    
    # execute agent
    try:
        system_message = SystemMessage(content=PLANNER_SYSTEM_PROMPT)
        user_message = HumanMessage(content=f"""User query: {state['user_query']}

                                                Dataset information:
                                                - Columns: {columns}
                                                - Shape: {shape}
                                                - Numeric columns: {numeric_cols}
                                                - Categorical columns: {categorical_cols}

                                                Please create a detailed analysis plan and coding instructions.""")
        
        messages = [system_message, user_message]
        response = llm.invoke(messages, config={"callbacks": callbacks})
        output = response.content
        
        # store the plan and instructions
        state["analysis_plan"] = output
        state["coding_prompt"] = output  # the entire output serves as coding prompt
        state["next_agent"] = "coding"
        state["messages"].append(AIMessage(content=f"Analysis plan created: {output[:200]}..."))
        
        emitter = get_emitter()
        if emitter:
            emitter.emit_message(
                from_agent="planner",
                to_agent="coding",
                summary=f"Handing off analysis plan ({len(output)} chars) for code generation",
            )
        
    except Exception as e:
        error_msg = f"Error in planner agent: {str(e)}"
        state["analysis_plan"] = error_msg
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=error_msg))
        
        emitter = get_emitter()
        if emitter:
            emitter.emit_error(
                agent_id="planner",
                error_type="logic",
                message=str(e),
            )
    
    return state
