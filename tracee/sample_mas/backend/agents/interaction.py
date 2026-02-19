from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from backend.state.schema import AnalysisState
from backend.tools.dataset_tools import create_dataset_tools_for_agent
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


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
    
    # load prompt from server (cached after first call)
    system_prompt = loader.get("interaction-prompt", agent_id="interaction")
    system_message = SystemMessage(content=system_prompt)
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
