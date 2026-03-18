"""API routes for playground LLM runs.

Provides endpoints to execute prompts against LLMs and track runs.
Supports OpenAI and Anthropic (Claude) models.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate, PlaygroundToolCall
from backbone.models.prompt_artifact import PromptTool, PromptVersion, SchemaMode
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
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config_id):
        raise HTTPException(status_code=400, detail=f"Invalid model config id: {config_id}")
    config_file = MODEL_CONFIGS_DIR / f"{config_id}.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {config_id}")
    return json.loads(config_file.read_text())


def _build_openai_response_format(schema: dict) -> dict:
    """Build OpenAI response_format from a JSON Schema dict."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("title", "output"),
            "schema": schema,
            "strict": True,
        },
    }


def _supports_openai_json_schema(model: str) -> bool:
    """Return whether the model supports OpenAI JSON Schema response_format."""
    return model.startswith("gpt-4o") or model.startswith("gpt-4.1")


def _build_anthropic_schema_tool(schema: dict) -> dict:
    """Build an Anthropic tool definition that forces structured output."""
    return {
        "name": "structured_output",
        "description": "Return the response in the required structured format.",
        "input_schema": schema,
    }


def _build_openai_tool(tool: PromptTool) -> dict:
    """Build an OpenAI tool definition from a prompt-authored tool."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema(),
        },
    }


def _build_anthropic_tool(tool: PromptTool) -> dict:
    """Build an Anthropic tool definition from a prompt-authored tool."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema(),
    }


def _extract_anthropic_structured_content(content_blocks: list) -> str:
    """Extract structured JSON from Anthropic tool_use response blocks."""
    for block in content_blocks:
        if getattr(block, "type", None) == "tool_use":
            return json.dumps(block.input)
    # fallback to text if no tool_use block found
    return "".join(
        getattr(block, "text", "") for block in content_blocks
        if getattr(block, "type", None) == "text"
    )


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
    prompt_tools: list[PromptTool] | None = None,
) -> dict:
    """Call OpenAI API and return standardized response."""
    client = _get_openai_client()

    try:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        if prompt_tools:
            params["tools"] = [_build_openai_tool(tool) for tool in prompt_tools]
        if output_schema:
            if not _supports_openai_json_schema(model):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Structured output requires an OpenAI model with json_schema "
                        f"support. Unsupported model: {model}"
                    ),
                )
            params["response_format"] = _build_openai_response_format(output_schema)
        response = await client.chat.completions.create(**params)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls: list[PlaygroundToolCall] = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                try:
                    parsed_arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    parsed_arguments = arguments
                tool_calls.append(
                    PlaygroundToolCall(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=parsed_arguments,
                    )
                )
        if not content and tool_calls:
            content = json.dumps([call.model_dump() for call in tool_calls], indent=2)
        usage = response.usage

        return {
            "content": content,
            "tool_calls": tool_calls or None,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "schema_enforced": bool(output_schema) and not tool_calls,
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
    prompt_tools: list[PromptTool] | None = None,
) -> dict:
    """Call Anthropic API and return standardized response."""
    client = _get_anthropic_client()

    try:
        effective_max_tokens = max_tokens or 4096
        api_params = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        tools: list[dict] = []
        if prompt_tools:
            tools.extend(_build_anthropic_tool(tool) for tool in prompt_tools)
        if output_schema:
            tools.append(_build_anthropic_schema_tool(output_schema))
        if tools:
            api_params["tools"] = tools
        if output_schema and not prompt_tools:
            api_params["tool_choice"] = {"type": "tool", "name": "structured_output"}

        response = await client.messages.create(**api_params)

        content = ""
        schema_enforced = False
        tool_calls: list[PlaygroundToolCall] = []
        text_parts: list[str] = []
        if response.content:
            for block in response.content:
                if block.type == "tool_use" and getattr(block, "name", None) == "structured_output":
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
                    continue
                if block.type == "tool_use":
                    tool_calls.append(
                        PlaygroundToolCall(
                            call_id=getattr(block, "id", None),
                            name=getattr(block, "name", "tool"),
                            arguments=getattr(block, "input", None),
                        )
                    )
                    continue
                if hasattr(block, "text"):
                    text_parts.append(block.text)
        if not schema_enforced:
            content = "".join(text_parts).strip()
        if not content and tool_calls:
            content = json.dumps([call.model_dump() for call in tool_calls], indent=2)
        usage = response.usage

        return {
            "content": content,
            "tool_calls": tool_calls or None,
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
    prompt_tools: list[PromptTool] | None = None,
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
        return await _call_openai(
            prompt,
            model,
            temperature,
            max_tokens,
            output_schema,
            prompt_tools,
        )
    elif provider_lower == "anthropic":
        return await _call_anthropic(
            prompt,
            model,
            temperature,
            max_tokens,
            output_schema,
            prompt_tools,
        )
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
        resolved_prompt = version.resolve(
            schema_mode=SchemaMode.full if version.tools else SchemaMode.hint
        )
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
        output_schema=version.output_schema if not version.tools else None,
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
