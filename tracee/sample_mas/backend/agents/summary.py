from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from backend.state.schema import AnalysisState


SUMMARY_SYSTEM_PROMPT = """You are a summary agent in a data analysis system. Your role is to:

1. Interpret the results from the code execution
2. Create a clear, user-friendly summary of the analysis
3. Highlight key findings and insights

**Your summary should:**
- Be concise but informative (2-4 paragraphs)
- Explain what analysis was performed
- Present key numerical results clearly
- Mention any visualizations that were created
- Note any issues or limitations if the code failed

**Tone:**
- Professional but accessible
- Avoid technical jargon where possible
- Focus on insights, not just mechanics

**Format:**
Start with what was done, then present findings, then conclude with visualizations or next steps.
"""


def create_summary_agent(state: AnalysisState) -> AnalysisState:
    """Summary agent that interprets results and creates user-friendly summary.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with final summary
    """
    # get execution results
    execution_result = state.get("execution_result", {})
    user_query = state["user_query"]
    generated_code = state.get("generated_code", "")
    
    callbacks = state.get("callbacks", [])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        callbacks=callbacks,
        metadata={"agent": "summary", "has_tools": False}
    )
    
    # prepare context for summary
    if execution_result.get("success"):
        context = f"""User Query: {user_query}

Code Execution: SUCCESS

Standard Output:
{execution_result.get('stdout', 'No output')}

Generated Visualizations:
{', '.join(execution_result.get('plots', [])) if execution_result.get('plots') else 'None'}

Variables Captured:
{execution_result.get('variables', {})}
"""
    else:
        error = execution_result.get('error', {})
        context = f"""User Query: {user_query}

Code Execution: FAILED

Error Type: {error.get('type', 'Unknown')}
Error Message: {error.get('message', 'Unknown error')}

Standard Error:
{execution_result.get('stderr', 'No error details')}
"""
    
    # create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", "{context}\n\nPlease provide a clear summary of the analysis results for the user.")
    ])
    
    # generate summary
    try:
        chain = prompt | llm
        response = chain.invoke({"context": context}, config={"callbacks": callbacks})
        
        summary = response.content
        state["final_summary"] = summary
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=summary))
        
    except Exception as e:
        error_msg = f"Error in summary agent: {str(e)}"
        state["final_summary"] = f"Analysis completed but summary generation failed. {error_msg}"
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
