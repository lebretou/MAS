from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from backend.state.schema import AnalysisState
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


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
        system_prompt = loader.get("planner-prompt", agent_id="planner")
        system_message = SystemMessage(content=system_prompt)
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
        
    except Exception as e:
        error_msg = f"Error in planner agent: {str(e)}"
        state["analysis_plan"] = error_msg
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
