from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from backend.state.schema import AnalysisState
from backend.tools.execution_tools import execute_code_safely
import os
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


class CodingResult(BaseModel):
    code: str
    explanation: str = Field(default="")
    libraries_used: list[str] = Field(default_factory=list)


def create_coding_agent(state: AnalysisState) -> AnalysisState:
    """Coding agent that generates executable Python code.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with generated code and execution results
    """
    # get dataset
    dataset = state["dataset"]
    callbacks = state.get("callbacks", [])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=callbacks,
        metadata={"agent": "coding", "has_tools": False}
    )
    
    # execute agent to generate code
    try:
        system_prompt, output_schema = loader.get_with_schema("coding-prompt", agent_id="coding")
        schema = output_schema if isinstance(output_schema, dict) and output_schema.get("title") else None
        structured_llm = llm.with_structured_output(schema or CodingResult)
        retry_count = state.get("retry_count", 0)
        previous_error = state.get("execution_result", {}).get("error", {}).get("message", "")
        system_message = SystemMessage(content=system_prompt)
        user_message = HumanMessage(content=f"""Analysis Plan and Instructions:
{state['coding_prompt']}

Dataset columns available: {list(dataset.columns)}
Retry attempt: {retry_count}
Previous execution error: {previous_error}

Please generate the Python code to accomplish this analysis.
Return JSON with the code field.""")
        
        response = structured_llm.invoke([system_message, user_message], config={"callbacks": callbacks})
        code = response["code"] if isinstance(response, dict) else response.code
        
        # clean up code if it has markdown formatting
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        state["generated_code"] = code
        state["messages"].append(AIMessage(content=f"Code generated successfully. {len(code)} characters."))
        
        # execute the code
        output_dir = os.path.join(os.path.dirname(__file__), "../../outputs")
        execution_result = execute_code_safely(code, dataset, output_dir)
        
        state["execution_result"] = execution_result
        
        if execution_result["success"]:
            state["next_agent"] = "summary"
            state["messages"].append(AIMessage(content="Code executed successfully."))
        else:
            retry_count += 1
            state["retry_count"] = retry_count
            if retry_count < 3:
                state["next_agent"] = "coding"
                state["messages"].append(AIMessage(content=f"Code execution failed. Retrying (attempt {retry_count + 1}/3)."))
                return state

            error_info = execution_result.get("error", {})
            state["next_agent"] = "summary"
            state["messages"].append(AIMessage(content=f"Code execution failed: {error_info.get('message', 'Unknown error')}"))
        
    except Exception as e:
        error_msg = f"Error in coding agent: {str(e)}"
        state["generated_code"] = ""
        state["execution_result"] = {"success": False, "error": {"message": str(e)}}
        state["next_agent"] = "summary"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
