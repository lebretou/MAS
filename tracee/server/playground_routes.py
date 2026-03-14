"""API routes for playground LLM runs.

Provides endpoints to execute prompts against LLMs and track runs.
Supports OpenAI and Anthropic (Claude) models.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.models.prompt_artifact import PromptVersion, SchemaMode
from backbone.utils.identifiers import generate_run_id, utc_timestamp
from server.playground_db import get_run, insert_run, list_runs
from server.prompt_db import get_latest_version as db_get_latest_version
from server.prompt_db import get_prompt as db_get_prompt
from server.prompt_db import get_version as db_get_version

logger = logging.getLogger(__name__)

router = APIRouter()

# Storage directories
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
MODEL_CONFIGS_DIR = Path(os.getenv("MODEL_CONFIGS_DIR", str(DEFAULT_DATA_DIR / "model_configs")))

# LLM Client instances (lazily initialized)
_openai_client = None
_anthropic_client = None


def _get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        import openai
        api_key = os.getenv("OPENAI_API_KEY") # we will change this later
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        _openai_client = openai.AsyncOpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_client():
    """Get or create Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY environment variable not set"
            )
        _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


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
    config_file = MODEL_CONFIGS_DIR / f"{config_id}.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {config_id}")
    return json.loads(config_file.read_text())


def _substitute_variables(template: str, variables: dict[str, str]) -> str:
    """Substitute {{variable}} placeholders in the template.

    Supports both {{variable}} and {{ variable }} syntax.
    """
    result = template
    for key, value in variables.items():
        # Handle both {{key}} and {{ key }} (with optional spaces)
        result = result.replace(f"{{{{{key}}}}}", value)
        result = result.replace(f"{{{{ {key} }}}}", value)
    return result


def _build_schema_params(output_schema: dict, provider: str) -> dict:
    """Build provider-specific function/tool calling params for schema enforcement."""
    if provider == "openai":
        return {
            "functions": [{
                "name": "structured_output",
                "description": "Return structured output conforming to the schema.",
                "parameters": output_schema,
            }],
            "function_call": {"name": "structured_output"},
        }
    elif provider == "anthropic":
        return {
            "tools": [{
                "name": "structured_output",
                "description": "Return structured output conforming to the schema.",
                "input_schema": output_schema,
            }],
            "tool_choice": {"type": "tool", "name": "structured_output"},
        }
    return {}


async def _call_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    output_schema: dict | None = None,
) -> dict:
    """Call OpenAI API and return standardized response."""
    client = _get_openai_client()

    try:
        # Build request parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        if output_schema:
            params.update(_build_schema_params(output_schema, "openai"))

        response = await client.chat.completions.create(**params)

        # Extract response data
        message = response.choices[0].message
        if output_schema and message.function_call:
            content = message.function_call.arguments
        elif output_schema:
            # Schema was requested but LLM did not return function_call
            logger.warning(
                "OpenAI did not return function_call despite schema enforcement "
                "(model=%s). Falling back to message content.",
                model,
            )
            content = message.content or ""
        else:
            content = message.content or ""
        usage = response.usage

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "schema_enforced": bool(output_schema and message.function_call),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OpenAI API call failed (model=%s)", model)
        raise HTTPException(
            status_code=500,
            detail="OpenAI API error. See server logs for details."
        )


async def _call_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    output_schema: dict | None = None,
) -> dict:
    """Call Anthropic API and return standardized response."""
    client = _get_anthropic_client()

    try:
        # anthropic requires max_tokens
        effective_max_tokens = max_tokens or 4096

        api_params = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if output_schema:
            api_params.update(_build_schema_params(output_schema, "anthropic"))

        response = await client.messages.create(**api_params)

        # Extract response data
        content = ""
        schema_enforced = False
        if output_schema and response.content:
            for block in response.content:
                if block.type == "tool_use" and block.name == "structured_output":
                    try:
                        content = json.dumps(block.input)
                    except TypeError:
                        content = json.dumps(block.input, default=str)
                        logger.warning(
                            "Anthropic tool_use input contained non-serializable types; "
                            "coerced to string (model=%s)",
                            model,
                        )
                    schema_enforced = True
                    break
            if not schema_enforced:
                # Schema was requested but LLM did not return tool_use
                logger.warning(
                    "Anthropic did not return tool_use block despite schema enforcement "
                    "(model=%s). Falling back to text content.",
                    model,
                )
                content = "".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
        elif response.content:
            content = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )

        usage = response.usage

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
                "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
            },
            "schema_enforced": schema_enforced,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Anthropic API call failed (model=%s)", model)
        raise HTTPException(
            status_code=500,
            detail="Anthropic API error. See server logs for details."
        )


async def _call_llm(
    prompt: str,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int | None,
    output_schema: dict | None = None,
) -> dict:
    """Call the appropriate LLM based on provider.

    Supports:
    - OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet, etc.

    Returns:
        dict with 'content' (str), 'usage' (dict with token counts),
        and 'schema_enforced' (bool)
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        return await _call_openai(prompt, model, temperature, max_tokens, output_schema)
    elif provider_lower == "anthropic":
        return await _call_anthropic(prompt, model, temperature, max_tokens, output_schema)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported: openai, anthropic"
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

    if version.output_schema is not None:
        resolved_prompt = version.resolve(schema_mode=SchemaMode.hint)
    else:
        resolved_prompt = version.resolve()

    if request.input_variables:
        resolved_prompt = _substitute_variables(resolved_prompt, request.input_variables)

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

    start_time = time.time()
    response = await _call_llm(
        prompt=resolved_prompt,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        output_schema=version.output_schema,
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
        output_schema=request.output_schema,
        output=response["content"],
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
