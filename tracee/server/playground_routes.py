"""API routes for playground LLM runs.

Provides endpoints to execute prompts against LLMs and track runs.
Supports OpenAI and Anthropic (Claude) models.
"""

import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.models.prompt_artifact import PromptVersion
from backbone.utils.identifiers import generate_run_id, utc_timestamp
from server.playground_db import get_run, insert_run, list_runs
from server.prompt_db import get_latest_version as db_get_latest_version
from server.prompt_db import get_prompt as db_get_prompt
from server.prompt_db import get_version as db_get_version

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


async def _call_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
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
        
        response = await client.chat.completions.create(**params)
        
        # Extract response data
        content = response.choices[0].message.content or ""
        usage = response.usage
        
        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )


async def _call_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> dict:
    """Call Anthropic API and return standardized response."""
    client = _get_anthropic_client()
    
    try:
        # anthropic requires max_tokens
        effective_max_tokens = max_tokens or 4096
        
        response = await client.messages.create(
            model=model,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Extract response data
        content = ""
        if response.content:
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
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anthropic API error: {str(e)}"
        )


async def _call_llm(
    prompt: str,
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int | None,
) -> dict:
    """Call the appropriate LLM based on provider.
    
    Supports:
    - OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet, etc.
    
    Returns:
        dict with 'content' (str) and 'usage' (dict with token counts)
    """
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return await _call_openai(prompt, model, temperature, max_tokens)
    elif provider_lower == "anthropic":
        return await _call_anthropic(prompt, model, temperature, max_tokens)
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
    version = _load_prompt_version(request.prompt_id, request.version_id)
    
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
        output=response["content"],
        latency_ms=latency_ms,
        prompt_tokens=response["usage"]["prompt_tokens"],
        completion_tokens=response["usage"]["completion_tokens"],
        total_tokens=response["usage"]["total_tokens"],
        model_config_id=request.model_config_id,
        tags=request.tags,
        notes=request.notes,
    )
    
    insert_run(run)
    
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
