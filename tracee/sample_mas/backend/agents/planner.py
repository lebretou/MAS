from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from backend.state.schema import AnalysisState
from backend.tools.rag_tools import retrieve_analysis_context_tool
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


class PlannerResult(BaseModel):
    analysis_steps: list[str] = Field(default_factory=list)
    coding_instructions: str
    visualization_suggestions: list[str] = Field(default_factory=list)
    statistical_methods: list[str] = Field(default_factory=list)


def _extract_last_assistant_text(messages: list[BaseMessage | dict]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role")
            if role not in ["ai", "assistant"]:
                continue
            content = message.get("content", "")
        else:
            if message.type != "ai":
                continue
            content = message.content

        if content:
            if isinstance(content, str):
                return content
            return str(content)
    return ""


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
        model="o3-mini",
        reasoning_effort="medium",
        callbacks=callbacks,
        metadata={"agent": "planner", "has_tools": False},
    )
    
    # prepare dataset info
    columns = dataset_info.get("columns", list(dataset.columns))
    shape = dataset_info.get("shape", {"rows": dataset.shape[0], "columns": dataset.shape[1]})
    numeric_cols = dataset_info.get("numeric_columns", list(dataset.select_dtypes(include=['number']).columns))
    categorical_cols = dataset_info.get("categorical_columns", list(dataset.select_dtypes(include=['object', 'category']).columns))
    # execute agent
    try:
        system_prompt, output_schema = loader.get_with_schema("planner-prompt", agent_id="planner")
        planning_agent = create_agent(
            model=llm,
            tools=[retrieve_analysis_context_tool],
            system_prompt=system_prompt,
        )

        planner_input = f"""User query: {state['user_query']}

Dataset information:
- Columns: {columns}
- Shape: {shape}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}

Use the retrieve_analysis_context_tool when useful for selecting robust analysis methods.
Draft a complete plan and coding instructions."""
        planner_response = planning_agent.invoke(
            {"messages": [{"role": "user", "content": planner_input}]},
            config={"callbacks": callbacks},
        )
        planner_messages = planner_response.get("messages", [])
        planner_draft = _extract_last_assistant_text(planner_messages)

        schema = output_schema if isinstance(output_schema, dict) and output_schema.get("title") else None
        structured_llm = llm.with_structured_output(schema or PlannerResult)
        result = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Convert the planner draft into strict JSON with fields: "
                        "analysis_steps, coding_instructions, visualization_suggestions, statistical_methods."
                    )
                ),
                HumanMessage(content=f"Planner draft:\n{planner_draft}"),
            ],
            config={"callbacks": callbacks},
        )

        if isinstance(result, dict):
            analysis_steps = result.get("analysis_steps", [])
            coding_instructions = str(result.get("coding_instructions", ""))
            visualization_suggestions = result.get("visualization_suggestions", [])
            statistical_methods = result.get("statistical_methods", [])
        else:
            analysis_steps = result.analysis_steps
            coding_instructions = result.coding_instructions
            visualization_suggestions = result.visualization_suggestions
            statistical_methods = result.statistical_methods

        analysis_plan = "\n".join([f"{idx}. {step}" for idx, step in enumerate(analysis_steps, start=1)])
        visualization_text = "\n- ".join(visualization_suggestions) if visualization_suggestions else "None"
        statistical_text = "\n- ".join(statistical_methods) if statistical_methods else "None"
        state["analysis_plan"] = analysis_plan
        state["coding_prompt"] = (
            f"{coding_instructions}\n\n"
            f"Visualization suggestions:\n- {visualization_text}\n\n"
            f"Statistical methods:\n- {statistical_text}"
        )
        state["rag_context"] = planner_draft
        state["next_agent"] = "coding"
        state["messages"].append(AIMessage(content=f"Analysis plan created: {analysis_plan[:200]}..."))
        
    except Exception as e:
        error_msg = f"Error in planner agent: {str(e)}"
        state["analysis_plan"] = error_msg
        state["next_agent"] = "end"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
