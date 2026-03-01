from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from backend.state.schema import AnalysisState
from backend.tools.dataset_tools import create_dataset_tools_for_agent
from backbone.sdk.prompt_loader import PromptLoader

loader = PromptLoader(base_url="http://localhost:8000")


class InteractionDecision(BaseModel):
    decision: str = Field(description="must be either relevant or chat_only")
    response: str = Field(description="final response for the user")
    reasoning: str = Field(description="concise routing reason")
    dataset_observations: list[str] = Field(default_factory=list)


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
    system_prompt, output_schema = loader.get_with_schema("interaction-prompt", agent_id="interaction")
    interaction_agent = create_agent(
        model=ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            temperature=0,
            callbacks=callbacks,
            metadata={"agent": "interaction", "has_tools": True},
        ),
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="block",
                apply_to_input=True,
            ),
        ],
    )

    try:
        user_message = (
            f"Dataset path: {state.get('dataset_path', 'uploaded_dataset')}\n\n"
            f"User query: {state['user_query']}"
        )
        agent_result = interaction_agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"callbacks": callbacks},
        )
        agent_messages = agent_result.get("messages", [])
        interaction_output = _extract_last_assistant_text(agent_messages)

        decision_llm = ChatOpenAI(
            model="gpt-4.1-2025-04-14",
            temperature=0,
            callbacks=callbacks,
            metadata={"agent": "interaction_decision", "has_tools": False},
        )
        schema = output_schema if isinstance(output_schema, dict) and output_schema.get("title") else None
        structured_llm = decision_llm.with_structured_output(schema or InteractionDecision)
        decision = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the routing decision step for the interaction agent. "
                        "Return JSON with decision, response, reasoning, dataset_observations. "
                        "decision must be relevant or chat_only."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original user query:\n{state['user_query']}\n\n"
                        f"Interaction agent draft response:\n{interaction_output}\n\n"
                        "If analysis/code execution is needed, decision should be relevant. "
                        "If user can be answered directly, decision should be chat_only."
                    )
                ),
            ],
            config={"callbacks": callbacks},
        )

        if isinstance(decision, dict):
            decision_value = str(decision.get("decision", "chat_only")).lower()
            response = str(decision.get("response", interaction_output))
            reasoning = str(decision.get("reasoning", ""))
        else:
            decision_value = decision.decision.lower()
            response = decision.response
            reasoning = decision.reasoning

        if decision_value == "relevant":
            state["relevance_decision"] = "relevant"
            state["next_agent"] = "planner"
            state["messages"].append(
                AIMessage(content=f"Query requires analysis. Proceeding to planning phase. Reason: {reasoning}")
            )
        else:
            state["relevance_decision"] = "chat_only"
            state["next_agent"] = "end"
            state["final_summary"] = response
            state["messages"].append(AIMessage(content=response))

    except Exception as e:
        state["relevance_decision"] = "chat_only"
        state["next_agent"] = "end"
        error_msg = f"Error in interaction agent: {str(e)}"
        state["final_summary"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
    
    return state
