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


def prompt_has_versions(client: httpx.Client, prompt_id: str) -> bool:
    response = client.get(f"{BASE_URL}/prompts/{prompt_id}")
    response.raise_for_status()
    data = response.json()
    return len(data.get("versions", [])) > 0


def create_version(
    client: httpx.Client,
    prompt_id: str,
    version_name: str,
    components: list[dict],
    output_schema: dict | None,
    *,
    parent_version_id: str | None = None,
    branch_name: str | None = None,
    revision_note: str | None = None,
) -> str:
    response = client.post(
        f"{BASE_URL}/prompts/{prompt_id}/versions",
        json={
            "name": version_name,
            "components": components,
            "variables": None,
            "output_schema": output_schema,
            "parent_version_id": parent_version_id,
            "branch_name": branch_name,
            "revision_note": revision_note,
        },
    )
    response.raise_for_status()
    return response.json()["version_id"]


def seed_interaction_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "interaction-prompt",
        "Interaction Agent Prompt",
        "interaction routing and dataset-aware response prompt",
    )
    if prompt_has_versions(client, "interaction-prompt"):
        return
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
                "prepare to route to downstream analysis agents. The other agents are: planner, coding, and summary."
                "If you can answer the user's query directly, you should do so using your tools and return chat_only."
                "If you think that the answer requires computational analysis, you should route to the planner agent."
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
                "Final output schema fields: decision, response, reasoning, dataset_observations. You must use your internal knowledge to determine whether other agents are needed to answer the user's query. "
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
    create_version(
        client,
        "interaction-prompt",
        "v2 structured output",
        components,
        output_schema,
        revision_note="adds structured output for routing decisions",
    )


def seed_planner_prompt(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "planner-prompt",
        "Planner Agent Prompt",
        "analysis planning and coding instruction generation",
    )
    if prompt_has_versions(client, "planner-prompt"):
        return
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
    if prompt_has_versions(client, "coding-prompt"):
        return
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
    if prompt_has_versions(client, "summary-prompt"):
        return
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


def seed_version_tree_demo(client: httpx.Client) -> None:
    ensure_prompt(
        client,
        "prompt-evolution-demo",
        "Prompt Evolution Demo",
        "non-linear playground demo prompt with multiple branches",
    )
    if prompt_has_versions(client, "prompt-evolution-demo"):
        return

    base_components = [
        {
            "type": "role",
            "content": "You are a release review assistant for a software team.",
            "enabled": True,
        },
        {
            "type": "task",
            "content": "Read {{release_brief}} and summarize launch readiness.",
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": "Return plain text with a short recommendation.",
            "enabled": True,
        },
    ]
    base_version = create_version(
        client,
        "prompt-evolution-demo",
        "baseline",
        base_components,
        None,
        revision_note="initial baseline prompt",
    )

    structured_components = [
        *base_components[:-1],
        {
            "type": "constraints",
            "content": "Do not invent missing facts. Keep the reasoning short and concrete.",
            "enabled": True,
        },
        {
            "type": "outputs",
            "content": "Return JSON with recommendation, blockers, confidence, and next_steps.",
            "enabled": True,
        },
    ]
    structured_schema = {
        "title": "ReleaseReview",
        "type": "object",
        "properties": {
            "recommendation": {"type": "string"},
            "blockers": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["recommendation", "blockers", "confidence", "next_steps"],
        "additionalProperties": False,
    }
    structured_version = create_version(
        client,
        "prompt-evolution-demo",
        "structured output",
        structured_components,
        structured_schema,
        parent_version_id=base_version,
        branch_name="main",
        revision_note="adds schema and explicit constraints",
    )

    examples_branch_components = [
        *structured_components,
        {
            "type": "examples",
            "content": (
                "Example:\n"
                "Input: release has one blocker and no QA signoff.\n"
                "Output: recommendation should be delay, blockers should list the blocker and missing signoff."
            ),
            "enabled": True,
        },
    ]
    create_version(
        client,
        "prompt-evolution-demo",
        "examples branch",
        examples_branch_components,
        structured_schema,
        parent_version_id=structured_version,
        branch_name="examples",
        revision_note="branches to add few-shot guidance",
    )

    stricter_branch_components = [
        *structured_components,
        {
            "type": "io_rules",
            "content": "confidence must be one of low, medium, or high. blockers must be empty when none are found.",
            "enabled": True,
        },
    ]
    create_version(
        client,
        "prompt-evolution-demo",
        "strict schema branch",
        stricter_branch_components,
        structured_schema,
        parent_version_id=structured_version,
        branch_name="strict-schema",
        revision_note="branches to tighten output normalization",
    )


def main() -> None:
    with httpx.Client(timeout=30.0) as client:
        seed_interaction_prompt(client)
        seed_planner_prompt(client)
        seed_coding_prompt(client)
        seed_summary_prompt(client)
        seed_version_tree_demo(client)
    print("prompt seeding completed")


if __name__ == "__main__":
    main()
