"""seed prompt versions for the advanced sample mas."""

import os
import httpx

BASE_URL = os.getenv("TRACEE_API_URL", "http://localhost:8000/api")


def ensure_prompt(client: httpx.Client, prompt_id: str, name: str, description: str) -> None:
    response = client.get(f"{BASE_URL}/prompts/{prompt_id}")
    if response.status_code == 404:
        create_response = client.post(
            f"{BASE_URL}/prompts",
            json={
                "prompt_id": prompt_id,
                "name": name,
                "description": description,
            },
        )
        create_response.raise_for_status()
        return
    response.raise_for_status()


def create_version(
    client: httpx.Client,
    prompt_id: str,
    version_name: str,
    components: list[dict],
    output_schema: dict | None,
) -> None:
    response = client.post(
        f"{BASE_URL}/prompts/{prompt_id}/versions",
        json={
            "name": version_name,
            "components": components,
            "variables": None,
            "output_schema": output_schema,
        },
    )
    response.raise_for_status()


def seed_interaction_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "interaction-prompt",
        "Interaction Agent Prompt",
        "interaction routing and dataset-aware response prompt",
    )
    components = [
        {
            "type": "role",
            "content": "You are an interaction agent in a data analysis system.",
            "enabled": True,
        },
        {
            "type": "task",
            "content": (
                "Use dataset tools to understand the dataset and draft an answer. "
                "If the user asks for analysis, modeling, plotting, or calculations, "
                "prepare to route to downstream analysis agents."
            ),
            "enabled": True,
        },
        {
            "type": "constraints",
            "content": (
                "PII middleware runs before your model call. Focus on relevance/routing. "
                "Be specific and concise."
            ),
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": (
                "Final output schema fields: decision, response, reasoning, dataset_observations. "
                "decision must be relevant or chat_only."
            ),
            "enabled": True,
        },
    ]
    output_schema = {
        "title": "InteractionDecision",
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["relevant", "chat_only"]},
            "response": {"type": "string"},
            "reasoning": {"type": "string"},
            "dataset_observations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "response", "reasoning", "dataset_observations"],
        "additionalProperties": False,
    }
    create_version(client, "interaction-prompt", "v2 structured output", components, output_schema)


def seed_planner_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "planner-prompt",
        "Planner Agent Prompt",
        "analysis planning and coding instruction generation",
    )
    components = [
        {
            "type": "role",
            "content": "You are a planning agent in a data analysis system.",
            "enabled": True,
        },
        {
            "type": "task",
            "content": (
                "Use user query and dataset metadata to produce a high-quality plan and coding guidance "
                "for the coding agent. call retrieve_analysis_context_tool when external analytical "
                "guidance would improve method selection."
            ),
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": (
                "Return JSON fields: analysis_steps, coding_instructions, "
                "visualization_suggestions, statistical_methods."
            ),
            "enabled": True,
        },
    ]
    output_schema = {
        "title": "PlannerResult",
        "type": "object",
        "properties": {
            "analysis_steps": {"type": "array", "items": {"type": "string"}},
            "coding_instructions": {"type": "string"},
            "visualization_suggestions": {"type": "array", "items": {"type": "string"}},
            "statistical_methods": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "analysis_steps",
            "coding_instructions",
            "visualization_suggestions",
            "statistical_methods",
        ],
        "additionalProperties": False,
    }
    create_version(client, "planner-prompt", "v2 structured output", components, output_schema)


def seed_coding_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "coding-prompt",
        "Coding Agent Prompt",
        "code generation prompt for analysis execution",
    )
    components = [
        {
            "type": "role",
            "content": "You are a coding agent in a data analysis system.",
            "enabled": True,
        },
        {
            "type": "constraints",
            "content": (
                "generate executable python using allowed libraries only. "
                "avoid file io and disallowed imports. validate columns before use."
            ),
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": "Return JSON fields: code, explanation, libraries_used.",
            "enabled": True,
        },
    ]
    output_schema = {
        "title": "CodingResult",
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "explanation": {"type": "string"},
            "libraries_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["code", "explanation", "libraries_used"],
        "additionalProperties": False,
    }
    create_version(client, "coding-prompt", "v2 structured output", components, output_schema)


def seed_summary_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "summary-prompt",
        "Summary Agent Prompt",
        "summary generation prompt for analysis results",
    )
    components = [
        {
            "type": "role",
            "content": "You are a summary agent in a data analysis system.",
            "enabled": True,
        },
        {
            "type": "task",
            "content": (
                "interpret execution outcomes and produce a user-facing summary with findings, "
                "limitations, and next steps."
            ),
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": "Return JSON fields: summary, key_findings, limitations, next_steps.",
            "enabled": True,
        },
    ]
    output_schema = {
        "title": "SummaryResult",
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "key_findings", "limitations", "next_steps"],
        "additionalProperties": False,
    }
    create_version(client, "summary-prompt", "v2 structured output", components, output_schema)


def main() -> None:
    with httpx.Client(timeout=30.0) as client:
        seed_interaction_prompt(client)
        seed_planner_prompt(client)
        seed_coding_prompt(client)
        seed_summary_prompt(client)
    print("prompt seeding completed")


if __name__ == "__main__":
    main()
