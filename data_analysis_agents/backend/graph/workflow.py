from langgraph.graph import StateGraph, END
from backend.state.schema import AnalysisState
from backend.agents.interaction import create_interaction_agent
from backend.agents.planner import create_planner_agent
from backend.agents.coding import create_coding_agent
from backend.agents.summary import create_summary_agent
from langchain_core.messages import HumanMessage
import pandas as pd


def should_continue_to_planner(state: AnalysisState) -> str:
    """Determine if the workflow should continue to planner or end.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name: "planner" or "end"
    """
    if state.get("relevance_decision") == "relevant":
        return "planner"
    return "end"


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow for the multi-agent system.
    
    Returns:
        Compiled StateGraph workflow
    """
    # Initialize the graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes for each agent
    workflow.add_node("interaction", create_interaction_agent)
    workflow.add_node("planner", create_planner_agent)
    workflow.add_node("coding", create_coding_agent)
    workflow.add_node("summary", create_summary_agent)
    
    # Set entry point
    workflow.set_entry_point("interaction")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "interaction",
        should_continue_to_planner,
        {
            "planner": "planner",
            "end": END
        }
    )
    
    # Add linear edges for the analysis pipeline
    workflow.add_edge("planner", "coding")
    workflow.add_edge("coding", "summary")
    workflow.add_edge("summary", END)
    
    # Compile the graph
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
    from backend.telemetry.config import get_callbacks
    
    # Get dataset info upfront
    dataset_info = {
        "columns": list(dataset.columns),
        "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "numeric_columns": list(dataset.select_dtypes(include=['number']).columns),
        "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
    }
    
    # Create callbacks once for the entire workflow
    callbacks = get_callbacks()
    
    # Set trace parameters on Langfuse handler if available
    for callback in callbacks:
        if hasattr(callback, 'set_trace_params'):
            callback.set_trace_params(
                name="Data Analysis Workflow",
                session_id=session_id,
                user_id="user",
                tags=["multi-agent", "data-analysis"],
                metadata={
                    "dataset_path": dataset_path,
                    "query": query,
                    "dataset_shape": dataset_info["shape"]
                }
            )
    
    # Initialize state
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
        "callbacks": callbacks,  # Share callbacks across all agents
    }
    
    # Create and run workflow
    app = create_workflow()
    
    try:
        # Execute workflow with callbacks in config to maintain trace context
        final_state = app.invoke(
            initial_state,
            config={"callbacks": callbacks}  # Pass callbacks to LangGraph
        )
        
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
    """Run the complete analysis workflow asynchronously.
    
    Args:
        dataset: The pandas DataFrame to analyze
        query: User's query/request
        dataset_path: Path or name of the dataset
        session_id: Session identifier for tracking
        
    Returns:
        Dictionary with workflow results
    """
    from backend.telemetry.config import get_callbacks
    
    # Get dataset info upfront
    dataset_info = {
        "columns": list(dataset.columns),
        "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
        "shape": {"rows": dataset.shape[0], "columns": dataset.shape[1]},
        "numeric_columns": list(dataset.select_dtypes(include=['number']).columns),
        "categorical_columns": list(dataset.select_dtypes(include=['object', 'category']).columns),
    }
    
    # Create callbacks once for the entire workflow
    callbacks = get_callbacks()
    
    # Set trace parameters on Langfuse handler if available
    for callback in callbacks:
        if hasattr(callback, 'set_trace_params'):
            callback.set_trace_params(
                name="Data Analysis Workflow",
                session_id=session_id,
                user_id="user",
                tags=["multi-agent", "data-analysis"],
                metadata={
                    "dataset_path": dataset_path,
                    "query": query,
                    "dataset_shape": dataset_info["shape"]
                }
            )
    
    # Initialize state
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
        "callbacks": callbacks,  # Share callbacks across all agents
    }
    
    # Create and run workflow
    app = create_workflow()
    
    try:
        # Execute workflow asynchronously with callbacks in config
        final_state = await app.ainvoke(
            initial_state,
            config={"callbacks": callbacks}  # Pass callbacks to LangGraph
        )
        
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
