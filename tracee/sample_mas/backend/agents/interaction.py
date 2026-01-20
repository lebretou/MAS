from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from backend.state.schema import AnalysisState
from backend.tools.dataset_tools import create_dataset_tools_for_agent


INTERACTION_SYSTEM_PROMPT = """You are an interaction agent in a data analysis system. Your role is to:

1. Understand the user's query in the context of the provided dataset
2. Use the available tools to inspect the dataset and understand its structure
3. Determine if the user's query is relevant to the dataset and requires analysis/coding
4. Answer simple questions about the dataset directly

You have access to tools that let you:
- Get dataset information (columns, types, shape, statistics)
- Get sample rows from the dataset
- Search for columns by keyword
- Get statistics for specific columns

**Decision Rules:**
- If the user asks a simple question that you can answer using the tools (e.g., "What columns are in this dataset?", "How many rows?"), answer it directly.
- If the user requests analysis, visualization, or ANY computation such as creating a plot, asking for correlations, etc. that requires code execution (e.g., "Plot X vs Y", "Run regression", "Calculate correlation"), respond with EXACTLY: [EXECUTE_ANALYSIS]
- If the user's query is completely unrelated to data analysis or the dataset, politely explain that you can only help with dataset-related queries.

**Important:**
- Always use tools to inspect the dataset before making decisions
- Be conversational and helpful
- When you decide analysis is needed, output ONLY the token: [EXECUTE_ANALYSIS]
"""


def create_interaction_agent(state: AnalysisState) -> AnalysisState:
    """Interaction agent that validates query relevance and provides initial responses.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with interaction results
    """
    # get dataset and create tools
    dataset = state["dataset"]
    tools = create_dataset_tools_for_agent(dataset)
    callbacks = state.get("callbacks", [])
    llm = ChatOpenAI(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        callbacks=callbacks,
        metadata={"agent": "interaction", "has_tools": True}
    )
    llm_with_tools = llm.bind_tools(tools)
    
    # create prompt
    system_message = SystemMessage(content=INTERACTION_SYSTEM_PROMPT)
    user_message = HumanMessage(content=f"Dataset path: {state.get('dataset_path', 'uploaded_dataset')}\n\nUser query: {state['user_query']}")
    
    # execute agent with tool calling
    try:
        messages = [system_message, user_message]
        max_iterations = 5
        
        for i in range(max_iterations):
            response = llm_with_tools.invoke(messages, config={"callbacks": callbacks})
            messages.append(response)
            
            # check if there are tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # find and execute the tool
                    tool = next((t for t in tools if t.name == tool_call["name"]), None)
                    if tool:
                        tool_result = tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        ))
            else:
                # no more tool calls, we have the final response
                break
        
        output = response.content
        
        # check if analysis should be executed
        if "[EXECUTE_ANALYSIS]" in output:
            state["relevance_decision"] = "relevant"
            state["next_agent"] = "planner"
            state["messages"].append(AIMessage(content="Query requires analysis. Proceeding to planning phase."))
        else:
            state["relevance_decision"] = "chat_only"
            state["next_agent"] = "end"
            state["final_summary"] = output
            state["messages"].append(AIMessage(content=output))
            
    except Exception as e:
        state["relevance_decision"] = "chat_only"
        state["next_agent"] = "end"
        error_msg = f"Error in interaction agent: {str(e)}"
        state["final_summary"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
