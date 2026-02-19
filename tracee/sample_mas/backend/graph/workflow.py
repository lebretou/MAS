from langgraph.graph import StateGraph, END
from backend.state.schema import AnalysisState
from backend.agents.interaction import create_interaction_agent
from backend.agents.planner import create_planner_agent
from backend.agents.coding import create_coding_agent
from backend.agents.summary import create_summary_agent
from langchain_core.messages import HumanMessage
import pandas as pd


def should_continue_to_planner(state: AnalysisState) -> str:
    if state.get("relevance_decision") == "relevant":
        return "planner"
    return "end"


def create_workflow() -> StateGraph:
    workflow = StateGraph(AnalysisState)
    
    workflow.add_node("interaction", create_interaction_agent, metadata={
        "prompt_id": "interaction-prompt",
        "model": "gpt-4.1-2025-04-14",
        "temperature": 0,
        "has_tools": True,
    })
    workflow.add_node("planner", create_planner_agent, metadata={
        "prompt_id": "planner-prompt",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "has_tools": False,
    })
    workflow.add_node("coding", create_coding_agent, metadata={
        "prompt_id": "coding-prompt",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "has_tools": False,
    })
    workflow.add_node("summary", create_summary_agent, metadata={
        "prompt_id": "summary-prompt",
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "has_tools": False,
    })
    
    # starting point
    workflow.set_entry_point("interaction")
    
    # langgraph allows conditional edges, in our sample project, the first (interaction)
    # agent can determine whether to proceed with other agents
    workflow.add_conditional_edges(
        "interaction",
        should_continue_to_planner,
        {
            "planner": "planner",
            "end": END
        }
    )
    
    # manually defines the rest of the paths
    workflow.add_edge("planner", "coding")
    workflow.add_edge("coding", "summary")
    workflow.add_edge("summary", END)
    
    # compile the graph
    app = workflow.compile()
    
    return app


def run_analysis_workflow(dataset: pd.DataFrame, query: str, dataset_path: str = "uploaded", session_id: str = "default") -> dict:
    """Run the complete analysis workflow.
    
    Args:
        dataset: The pandas DataFrame to analyze
        query: User's query/request
        dataset_path: Path or name of the dataset
        session_id: Session identifier for tracking
        
    Returns:
        Dictionary with workflow results
    """
    from backend.telemetry.config import tracing_session
    
    # get dataset info upfront and store it in the state
    dataset_info = {
        "columns": list(dataset.columns),
        "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "numeric_columns": list(dataset.select_dtypes(include=['number']).columns),
        "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
    }
    
    with tracing_session(session_id=session_id) as ctx:
        callbacks = ctx.callbacks
        
        # initialize state
        initial_state = {
            "dataset": dataset,
            "dataset_path": dataset_path,
            "dataset_info": dataset_info,
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "relevance_decision": "",
            "analysis_plan": "",
            "coding_prompt": "",
            "generated_code": "",
            "execution_result": {},
            "final_summary": "",
            "next_agent": "interaction",
            "session_id": session_id,
            "callbacks": callbacks,
        }
        
        # create and run workflow
        app = create_workflow()
        
        try:
            final_state = app.invoke(initial_state, config={"callbacks": callbacks})
            
            return {
                "success": True,
                "final_summary": final_state.get("final_summary", ""),
                "relevance_decision": final_state.get("relevance_decision", ""),
                "generated_code": final_state.get("generated_code", ""),
                "execution_result": final_state.get("execution_result", {}),
                "analysis_plan": final_state.get("analysis_plan", ""),
                "messages": [{"role": msg.type, "content": msg.content} for msg in final_state.get("messages", [])],
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "final_summary": f"Workflow execution failed: {str(e)}",
            }


async def run_analysis_workflow_async(dataset: pd.DataFrame, query: str, dataset_path: str = "uploaded", session_id: str = "default") -> dict:
    """Run the complete analysis workflow asynchronously."""

    from backend.telemetry.config import tracing_session
    
    # get dataset info upfront
    dataset_info = {
        "columns": list(dataset.columns),
        "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "numeric_columns": list(dataset.select_dtypes(include=['number']).columns),
        "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
    }
    
    with tracing_session(session_id=session_id) as ctx:
        callbacks = ctx.callbacks
        
        # initialize state
        initial_state = {
            "dataset": dataset,
            "dataset_path": dataset_path,
            "dataset_info": dataset_info,
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "relevance_decision": "",
            "analysis_plan": "",
            "coding_prompt": "",
            "generated_code": "",
            "execution_result": {},
            "final_summary": "",
            "next_agent": "interaction",
            "session_id": session_id,
            "callbacks": callbacks,
        }
        
        # create and run workflow
        app = create_workflow()
        
        try:
            final_state = await app.ainvoke(initial_state, config={"callbacks": callbacks})
            
            return {
                "success": True,
                "final_summary": final_state.get("final_summary", ""),
                "relevance_decision": final_state.get("relevance_decision", ""),
                "generated_code": final_state.get("generated_code", ""),
                "execution_result": final_state.get("execution_result", {}),
                "analysis_plan": final_state.get("analysis_plan", ""),
                "messages": [{"role": msg.type, "content": msg.content} for msg in final_state.get("messages", [])],
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "final_summary": f"Workflow execution failed: {str(e)}",
            }
