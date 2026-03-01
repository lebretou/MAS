from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from backend.state.schema import AnalysisState
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


class SummaryResult(BaseModel):
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


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
        metadata={"agent": "summary", "has_tools": False},
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
    
    # generate summary
    try:
        system_prompt, output_schema = loader.get_with_schema("summary-prompt", agent_id="summary")
        schema = output_schema if isinstance(output_schema, dict) and output_schema.get("title") else None
        structured_llm = llm.with_structured_output(schema or SummaryResult)
        response = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"{context}\n\nPlease provide a clear summary in JSON format."),
            ],
            config={"callbacks": callbacks},
        )

        if isinstance(response, dict):
            summary = str(response.get("summary", ""))
            key_findings = response.get("key_findings", [])
            limitations = response.get("limitations", [])
            next_steps = response.get("next_steps", [])
        else:
            summary = response.summary
            key_findings = response.key_findings
            limitations = response.limitations
            next_steps = response.next_steps

        findings_text = "\n- ".join(key_findings) if key_findings else "None reported"
        limitations_text = "\n- ".join(limitations) if limitations else "None reported"
        next_steps_text = "\n- ".join(next_steps) if next_steps else "No immediate next steps"
        formatted_summary = (
            f"{summary}\n\n"
            f"Key findings:\n- {findings_text}\n\n"
            f"Limitations:\n- {limitations_text}\n\n"
            f"Next steps:\n- {next_steps_text}"
        )
        state["final_summary"] = formatted_summary
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=formatted_summary))
        
    except Exception as e:
        error_msg = f"Error in summary agent: {str(e)}"
        state["final_summary"] = f"Analysis completed but summary generation failed. {error_msg}"
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
