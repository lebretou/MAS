"""API routes for prompt management."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.models.prompt_artifact import (
    Prompt,
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.utils.identifiers import utc_timestamp

router = APIRouter()

# configurable prompts directory
DEFAULT_PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", str(DEFAULT_PROMPTS_DIR)))


# --- Request/Response Models ---


class CreatePromptRequest(BaseModel):
    """Request body for creating a new prompt."""

    prompt_id: str  # unique identifier
    agent_name: str  # which agent uses this
    name: str  # display name
    description: str | None = None


class CreateVersionRequest(BaseModel):
    """Request body for creating a new prompt version."""

    name: str  # version label/name
    components: list[PromptComponent]
    variables: dict[str, str] | None = None


class PromptWithVersions(BaseModel):
    """Prompt metadata along with all its versions."""

    prompt: Prompt
    versions: list[PromptVersion]


class PromptListItem(BaseModel):
    """Summary item for listing prompts."""

    prompt_id: str
    agent_name: str
    name: str
    description: str | None
    latest_version_id: str | None
    version_count: int
    created_at: str
    updated_at: str


# --- Helper Functions ---


def _get_prompt_dir(prompt_id: str) -> Path:
    """Get path to prompt directory."""
    return PROMPTS_DIR / prompt_id


def _get_metadata_file(prompt_id: str) -> Path:
    """Get path to prompt metadata file."""
    return _get_prompt_dir(prompt_id) / "metadata.json"


def _get_versions_dir(prompt_id: str) -> Path:
    """Get path to versions directory."""
    return _get_prompt_dir(prompt_id) / "versions"


def _load_prompt(prompt_id: str) -> Prompt:
    """Load prompt metadata from disk."""
    metadata_file = _get_metadata_file(prompt_id)
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    return Prompt.model_validate_json(metadata_file.read_text())


def _save_prompt(prompt: Prompt) -> None:
    """Save prompt metadata to disk."""
    prompt_dir = _get_prompt_dir(prompt.prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = _get_metadata_file(prompt.prompt_id)
    metadata_file.write_text(prompt.model_dump_json(indent=2))


def _load_version(prompt_id: str, version_id: str) -> PromptVersion:
    """Load a specific prompt version from disk."""
    version_file = _get_versions_dir(prompt_id) / f"{version_id}.json"
    if not version_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Version not found: {prompt_id}/{version_id}",
        )
    return PromptVersion.model_validate_json(version_file.read_text())


def _save_version(version: PromptVersion) -> None:
    """Save a prompt version to disk."""
    versions_dir = _get_versions_dir(version.prompt_id)
    versions_dir.mkdir(parents=True, exist_ok=True)
    
    version_file = versions_dir / f"{version.version_id}.json"
    version_file.write_text(version.model_dump_json(indent=2))


def _list_versions(prompt_id: str) -> list[PromptVersion]:
    """List all versions for a prompt."""
    versions_dir = _get_versions_dir(prompt_id)
    if not versions_dir.exists():
        return []
    
    versions = []
    for version_file in versions_dir.glob("*.json"):
        versions.append(PromptVersion.model_validate_json(version_file.read_text()))
    
    # Sort by created_at descending (newest first)
    versions.sort(key=lambda v: v.created_at, reverse=True)
    return versions


def _generate_version_id(prompt_id: str) -> str:
    """Generate the next version ID for a prompt."""
    versions = _list_versions(prompt_id)
    if not versions:
        return "v1"
    
    # Find highest version number
    max_num = 0
    for v in versions:
        if v.version_id.startswith("v") and v.version_id[1:].isdigit():
            max_num = max(max_num, int(v.version_id[1:]))
    
    return f"v{max_num + 1}"


# --- API Endpoints ---


@router.get("/prompts")
def list_prompts() -> list[PromptListItem]:
    """List all prompts with metadata."""
    prompts = []
    
    if not PROMPTS_DIR.exists():
        return prompts
    
    for prompt_dir in PROMPTS_DIR.iterdir():
        if not prompt_dir.is_dir():
            continue
        
        metadata_file = prompt_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        prompt = Prompt.model_validate_json(metadata_file.read_text())
        versions = _list_versions(prompt.prompt_id)
        
        prompts.append(
            PromptListItem(
                prompt_id=prompt.prompt_id,
                agent_name=prompt.agent_name,
                name=prompt.name,
                description=prompt.description,
                latest_version_id=prompt.latest_version_id,
                version_count=len(versions),
                created_at=prompt.created_at,
                updated_at=prompt.updated_at,
            )
        )
    
    # Sort by updated_at descending
    prompts.sort(key=lambda p: p.updated_at, reverse=True)
    return prompts


@router.post("/prompts")
def create_prompt(request: CreatePromptRequest) -> Prompt:
    """Create a new prompt."""
    # Check if prompt already exists
    if _get_metadata_file(request.prompt_id).exists():
        raise HTTPException(
            status_code=409,
            detail=f"Prompt already exists: {request.prompt_id}",
        )
    
    now = utc_timestamp()
    prompt = Prompt(
        prompt_id=request.prompt_id,
        agent_name=request.agent_name,
        name=request.name,
        description=request.description,
        created_at=now,
        updated_at=now,
        latest_version_id=None,
    )
    
    _save_prompt(prompt)
    return prompt


@router.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: str) -> PromptWithVersions:
    """Get prompt details with all versions."""
    prompt = _load_prompt(prompt_id)
    versions = _list_versions(prompt_id)
    
    return PromptWithVersions(prompt=prompt, versions=versions)


@router.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str) -> dict:
    """Delete a prompt and all its versions."""
    prompt_dir = _get_prompt_dir(prompt_id)
    
    if not prompt_dir.exists():
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    
    # Delete all files in the prompt directory
    import shutil
    shutil.rmtree(prompt_dir)
    
    return {"deleted": prompt_id}


@router.post("/prompts/{prompt_id}/versions")
def create_version(prompt_id: str, request: CreateVersionRequest) -> PromptVersion:
    """Create a new version of a prompt (immutable once created)."""
    # Verify prompt exists
    prompt = _load_prompt(prompt_id)
    
    # Generate version ID
    version_id = _generate_version_id(prompt_id)
    now = utc_timestamp()
    
    version = PromptVersion(
        prompt_id=prompt_id,
        version_id=version_id,
        name=request.name,
        components=request.components,
        variables=request.variables,
        created_at=now,
    )
    
    _save_version(version)
    
    # Update prompt metadata with latest version
    prompt.latest_version_id = version_id
    prompt.updated_at = now
    _save_prompt(prompt)
    
    return version


@router.get("/prompts/{prompt_id}/versions/{version_id}")
def get_version(prompt_id: str, version_id: str) -> PromptVersion:
    """Get a specific prompt version."""
    # Verify prompt exists
    _load_prompt(prompt_id)
    return _load_version(prompt_id, version_id)


@router.get("/prompts/{prompt_id}/versions/{version_id}/resolve")
def resolve_version(prompt_id: str, version_id: str) -> dict:
    """Get the resolved prompt text for a version.
    
    Returns the concatenated text of all enabled components.
    """
    # Verify prompt exists
    _load_prompt(prompt_id)
    version = _load_version(prompt_id, version_id)
    
    resolved_text = version.resolve()
    
    return {
        "prompt_id": prompt_id,
        "version_id": version_id,
        "resolved_text": resolved_text,
        "component_count": len(version.components),
        "enabled_count": sum(1 for c in version.components if c.enabled),
    }


@router.get("/prompts/{prompt_id}/latest")
def get_latest_version(prompt_id: str) -> PromptVersion:
    """Get the latest version of a prompt."""
    prompt = _load_prompt(prompt_id)
    
    if not prompt.latest_version_id:
        raise HTTPException(
            status_code=404,
            detail=f"No versions exist for prompt: {prompt_id}",
        )
    
    return _load_version(prompt_id, prompt.latest_version_id)
