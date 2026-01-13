from typing import TypedDict, Annotated, Any
from operator import add
from langchain_core.messages import BaseMessage
import pandas as pd


class AnalysisState(TypedDict):
    """State schema shared across all agents in the workflow."""
    
    # dataset
    dataset: pd.DataFrame
    dataset_path: str
    dataset_info: dict  # columns, dtypes, shape, sample rows
    
    # conversation
    messages: Annotated[list[BaseMessage], add]  # use add to append messages
    user_query: str
    
    # agent outputs
    relevance_decision: str  # "relevant" or "chat_only"
    analysis_plan: str
    coding_prompt: str
    generated_code: str
    execution_result: dict  # stdout, plots, errors
    final_summary: str
    
    # control flow
    next_agent: str  # which agent to route to next
    
    # session management
    session_id: str
    
    # telemetry - shared callbacks for unified tracing
    callbacks: list  # callback handlers (LangSmith, MAS backbone)
