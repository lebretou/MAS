"""API routes for playground LLM runs."""

import asyncio
import json
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.models.prompt_artifact import PromptTool, PromptVersion
from backbone.utils.identifiers import generate_run_id, utc_timestamp
from server.llm_clients import call_llm_messages
from server.playground_db import get_run, insert_run, list_runs
from server.prompt_db import get_latest_version as db_get_latest_version
from server.prompt_db import get_prompt as db_get_prompt
from server.prompt_db import get_version as db_get_version

router = APIRouter()

# Storage directories
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
MODEL_CONFIGS_DIR = Path(os.getenv("MODEL_CONFIGS_DIR", str(DEFAULT_DATA_DIR / "model_configs")))

def _validate_output_schema(output_schema: dict, version: PromptVersion) -> PromptVersion:
    """Validate and apply an output_schema to a PromptVersion.

    Re-runs Pydantic validation to prevent bypassing model validators
    via model_copy.
    """
    try:
        return PromptVersion.model_validate({
            **version.model_dump(),
            "output_schema": output_schema,
        })
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid output_schema: {e.errors()}"
        )


def _load_prompt_version(prompt_id: str, version_id: str) -> PromptVersion:
    """Load a prompt version from storage."""
    if not db_get_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    if version_id == "latest":
        version = db_get_latest_version(prompt_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"No versions for prompt: {prompt_id}")
        return version
    version = db_get_version(prompt_id, version_id)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Version not found: {prompt_id}/{version_id}",
        )
    return version


def _load_model_config(config_id: str) -> dict:
    """Load a saved model configuration."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config_id):
        raise HTTPException(status_code=400, detail=f"Invalid model config id: {config_id}")
    config_file = MODEL_CONFIGS_DIR / f"{config_id}.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {config_id}")
    return json.loads(config_file.read_text())


def _substitute_variables(template: str, variables: dict[str, str]) -> str:
    """Substitute {{variable}} placeholders in the template.

    Supports {{ variable }} syntax with optional whitespace.
    """
    return re.sub(
        r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
        lambda match: variables.get(match.group(1), match.group(0)),
        template,
    )


def _component_to_chat_role(component) -> str:
    role = component.resolved_message_role().value
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return "system"


def _component_to_message_content(component, variables: dict[str, str]) -> str:
    content = _substitute_variables(component.content, variables) if component.content else ""
    title = component.display_name()
    if not content:
        return f"{title}:"
    return f"{title}:\n{content}"


def _build_playground_messages(
    version: PromptVersion,
    variables: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    active_variables = variables or {}
    messages: list[dict[str, str]] = []

    for component in version.components:
        if not component.enabled:
            continue

        message = {
            "role": _component_to_chat_role(component),
            "content": _component_to_message_content(component, active_variables),
        }
        messages.append(message)

    return messages


def _serialize_playground_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role'].capitalize()}:\n{message['content']}"
        for message in messages
    )


# --- API Endpoints ---


class PlaygroundRunResponse(BaseModel):
    """Response model for playground run with additional metadata."""
    run: PlaygroundRun
    message: str = "Run completed successfully"


@router.post("/playground/run", response_model=PlaygroundRunResponse)
async def execute_playground_run(request: PlaygroundRunCreate) -> PlaygroundRunResponse:
    """Execute a prompt in the playground.

    Takes a prompt reference and model configuration, runs the prompt
    against the LLM, and stores the result.
    """
    version = await asyncio.to_thread(_load_prompt_version, request.prompt_id, request.version_id)

    if request.output_schema is not None:
        version = _validate_output_schema(request.output_schema, version)

    if request.model_config_id:
        config = _load_model_config(request.model_config_id)
        model = config["model_name"]
        provider = config["provider"]
        temperature = config["temperature"]
        max_tokens = config.get("max_tokens")
    else:
        model = request.model
        provider = request.provider
        temperature = request.temperature
        max_tokens = request.max_tokens

    if provider.lower() != "openai":
        raise HTTPException(
            status_code=400,
            detail="Only OpenAI is supported in the playground.",
        )

    messages = _build_playground_messages(version, request.input_variables)
    if not messages:
        raise HTTPException(
            status_code=422,
            detail="Playground run requires at least one enabled prompt component.",
        )
    resolved_prompt = _serialize_playground_messages(messages)

    start_time = time.time()
    response = await call_llm_messages(
        messages=messages,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        output_schema=version.output_schema,
        prompt_tools=version.tools or None,
    )
    latency_ms = (time.time() - start_time) * 1000

    run = PlaygroundRun(
        run_id=generate_run_id(),
        created_at=utc_timestamp(),
        prompt_id=request.prompt_id,
        version_id=version.version_id,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        input_variables=request.input_variables,
        resolved_prompt=resolved_prompt,
        output_schema=version.output_schema,
        tools=version.tools or None,
        tool_calls=response["tool_calls"],
        output=response["content"],
        output_schema_used=response["schema_enforced"],
        latency_ms=latency_ms,
        prompt_tokens=response["usage"]["prompt_tokens"],
        completion_tokens=response["usage"]["completion_tokens"],
        total_tokens=response["usage"]["total_tokens"],
        model_config_id=request.model_config_id,
        tags=request.tags,
        notes=request.notes,
    )

    await asyncio.to_thread(insert_run, run)

    return PlaygroundRunResponse(run=run)


@router.get("/playground/runs")
def list_playground_runs(
    prompt_id: str | None = None,
    limit: int = 50,
) -> list[PlaygroundRun]:
    """List playground runs, optionally filtered by prompt_id."""
    return list_runs(limit=limit, prompt_id=prompt_id)


@router.get("/playground/runs/{run_id}")
def get_playground_run(run_id: str) -> PlaygroundRun:
    """Get a specific playground run by ID."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@router.get("/playground/runs/prompt/{prompt_id}")
def list_runs_for_prompt(prompt_id: str, limit: int = 20) -> list[PlaygroundRun]:
    """List all playground runs for a specific prompt."""
    return list_runs(limit=limit, prompt_id=prompt_id)
