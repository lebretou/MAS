from typing import TypedDict, Annotated, Any
from operator import add
from langchain_core.messages import BaseMessage
import pandas as pd


class AnalysisState(TypedDict):
    """State schema shared across all agents in the workflow."""
    
    # Dataset
    dataset: pd.DataFrame
    dataset_path: str
    dataset_info: dict  # columns, dtypes, shape, sample rows
    
    # Conversation
    messages: Annotated[list[BaseMessage], add]  # Use add to append messages
    user_query: str
    
    # Agent outputs
    relevance_decision: str  # "relevant" or "chat_only"
    analysis_plan: str
    coding_prompt: str
    generated_code: str
    execution_result: dict  # stdout, plots, errors
    final_summary: str
    
    # Control flow
    next_agent: str  # Which agent to route to next
    
    # Session management
    session_id: str
    
    # Telemetry - shared callbacks for unified tracing
    callbacks: list  # List of callback handlers (Langfuse, etc.)
