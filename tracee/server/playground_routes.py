"""API routes for playground LLM runs."""

import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from backbone.models.playground_run import PlaygroundRun, PlaygroundRunCreate
from backbone.models.prompt_artifact import PromptTool, PromptVersion
from backbone.utils.identifiers import generate_run_id, utc_timestamp
from server.llm_clients import call_llm_messages, embed_openai_texts
from server.playground_db import get_run, insert_run, list_runs
from server.prompt_db import get_latest_version as db_get_latest_version
from server.prompt_db import get_prompt as db_get_prompt
from server.prompt_db import get_version as db_get_version

router = APIRouter()

# Storage directories
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
MODEL_CONFIGS_DIR = Path(os.getenv("MODEL_CONFIGS_DIR", str(DEFAULT_DATA_DIR / "model_configs")))
ALLOWED_EMBEDDING_MODELS = {"text-embedding-3-small"}
MAX_ANALYSIS_ITEMS = 60
MAX_ANALYSIS_CHARS = 120_000

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


def _build_inline_prompt_version(request: PlaygroundRunCreate) -> PromptVersion:
    """build a transient prompt version for unsaved draft runs."""
    try:
        return PromptVersion.model_validate({
            "prompt_id": request.prompt_id,
            "version_id": request.version_id,
            "name": request.version_id,
            "components": request.components or [],
            "variables": request.input_variables,
            "output_schema": request.output_schema,
            "tools": request.tools or [],
            "created_at": utc_timestamp(),
        })
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid inline prompt payload: {e.errors()}"
        )


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


class PlaygroundAnalysisItem(BaseModel):
    """Text item to project into a shared scatterplot."""

    id: str = Field(min_length=1, max_length=128)
    group_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=160)
    output: str = Field(min_length=1, max_length=20_000)


class PlaygroundAnalysisRequest(BaseModel):
    """Analysis request for embedding-backed projection."""

    items: list[PlaygroundAnalysisItem] = Field(min_length=1, max_length=MAX_ANALYSIS_ITEMS)
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1, max_length=64)


class PlaygroundAnalysisPoint(BaseModel):
    """Projected point for a single analysis item."""

    id: str
    group_id: str
    label: str
    x: float
    y: float
    average_similarity: float


class PlaygroundAnalysisResponse(BaseModel):
    """Projection response for playground analysis."""

    points: list[PlaygroundAnalysisPoint]


def _project_embeddings_2d(embeddings: list[list[float]]) -> tuple[list[tuple[float, float]], list[float]]:
    """project embeddings with pca and compute average cosine similarity."""
    if not embeddings:
        return [], []

    if not all(isinstance(vector, list) for vector in embeddings):
        raise HTTPException(status_code=422, detail="Embedding payload must be two-dimensional.")

    def dot(left: list[float], right: list[float]) -> float:
        return sum(left[index] * right[index] for index in range(len(left)))

    def norm(vector: list[float]) -> float:
        return math.sqrt(dot(vector, vector))

    def normalize(vector: list[float]) -> list[float]:
        magnitude = norm(vector)
        if magnitude == 0:
            return [0.0 for _ in vector]
        return [value / magnitude for value in vector]

    def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
        return [dot(row, vector) for row in matrix]

    def outer(vector: list[float], scale: float) -> list[list[float]]:
        return [[scale * left * right for right in vector] for left in vector]

    def power_iteration(matrix: list[list[float]], max_iterations: int = 80) -> tuple[float, list[float]]:
        size = len(matrix)
        vector = normalize([float(index + 1) for index in range(size)])
        for _ in range(max_iterations):
            next_vector = matrix_vector_multiply(matrix, vector)
            next_norm = norm(next_vector)
            if next_norm <= 1e-12:
                return 0.0, [0.0 for _ in range(size)]
            vector = [value / next_norm for value in next_vector]
        eigenvalue = dot(vector, matrix_vector_multiply(matrix, vector))
        return eigenvalue, vector

    normalized = [normalize(vector) for vector in embeddings]
    average_similarity = []
    for index, vector in enumerate(normalized):
        row = [dot(vector, other) for other in normalized]
        others = [value for other_index, value in enumerate(row) if other_index != index]
        average_similarity.append(sum(others) / len(others) if others else 1.0)

    if len(embeddings) == 1:
        return [(0.5, 0.5)], average_similarity

    dimension = len(normalized[0])
    means = [
        sum(vector[dim_index] for vector in normalized) / len(normalized)
        for dim_index in range(dimension)
    ]
    centered = [
        [value - means[dim_index] for dim_index, value in enumerate(vector)]
        for vector in normalized
    ]
    gram = [
        [dot(centered[row_index], centered[col_index]) for col_index in range(len(centered))]
        for row_index in range(len(centered))
    ]
    working = [row[:] for row in gram]
    axes: list[list[float]] = []
    for _ in range(2):
        eigenvalue, eigenvector = power_iteration(working)
        if eigenvalue <= 1e-9:
            break
        scale = math.sqrt(eigenvalue)
        axes.append([value * scale for value in eigenvector])
        deflation = outer(eigenvector, eigenvalue)
        for row_index in range(len(working)):
            for col_index in range(len(working)):
                working[row_index][col_index] -= deflation[row_index][col_index]

    if not axes:
        axes = [[0.0 for _ in centered], [0.0 for _ in centered]]
    elif len(axes) == 1:
        axes.append([0.0 for _ in centered])

    scaled_axes: list[list[float]] = []
    for axis in range(2):
        values = axes[axis]
        minimum = min(values)
        maximum = max(values)
        if maximum - minimum < 1e-9:
            scaled_axes.append([0.5 for _ in values])
            continue
        padding = (maximum - minimum) * 0.12
        scaled_axes.append([
            (value - (minimum - padding)) / ((maximum + padding) - (minimum - padding))
            for value in values
        ])

    points = [
        (
            max(0.0, min(1.0, scaled_axes[0][index])),
            max(0.0, min(1.0, 1.0 - scaled_axes[1][index])),
        )
        for index in range(len(embeddings))
    ]
    return points, average_similarity


@router.post("/playground/run", response_model=PlaygroundRunResponse)
async def execute_playground_run(request: PlaygroundRunCreate) -> PlaygroundRunResponse:
    """Execute a prompt in the playground.

    Takes a prompt reference and model configuration, runs the prompt
    against the LLM, and stores the result.
    """
    if request.components is not None:
        version = _build_inline_prompt_version(request)
    else:
        version = await asyncio.to_thread(_load_prompt_version, request.prompt_id, request.version_id)
        if request.output_schema is not None:
            version = _validate_output_schema(request.output_schema, version)
    if request.disable_output_schema:
        version = version.model_copy(update={"output_schema": None})

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


@router.post("/playground/analyze", response_model=PlaygroundAnalysisResponse)
async def analyze_playground_outputs(request: PlaygroundAnalysisRequest) -> PlaygroundAnalysisResponse:
    """project outputs into a 2d embedding space."""
    if request.embedding_model not in ALLOWED_EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported embedding model.")

    items = [item for item in request.items if item.output.strip()]
    if len(items) < 2:
        return PlaygroundAnalysisResponse(points=[])

    total_chars = sum(len(item.output) for item in items)
    if total_chars > MAX_ANALYSIS_CHARS:
        raise HTTPException(status_code=413, detail="Analysis payload is too large.")

    embeddings = await embed_openai_texts(
        texts=[item.output for item in items],
        model=request.embedding_model,
    )
    projected_points, average_similarity = _project_embeddings_2d(embeddings)
    return PlaygroundAnalysisResponse(
        points=[
            PlaygroundAnalysisPoint(
                id=item.id,
                group_id=item.group_id,
                label=item.label,
                x=projected_points[index][0],
                y=projected_points[index][1],
                average_similarity=average_similarity[index],
            )
            for index, item in enumerate(items)
        ]
    )


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
