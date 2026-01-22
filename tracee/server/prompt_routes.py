"""API routes for prompt management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backbone.models.prompt_artifact import (
    Prompt,
    PromptComponent,
    PromptComponentType,
    PromptVersion,
)
from backbone.utils.identifiers import utc_timestamp
from server.prompt_db import (
    create_prompt as db_create_prompt,
    delete_prompt as db_delete_prompt,
    get_latest_version as db_get_latest_version,
    get_prompt as db_get_prompt,
    list_prompts as db_list_prompts,
    list_versions as db_list_versions,
    insert_version as db_insert_version,
    get_version as db_get_version,
    update_prompt as db_update_prompt,
)

router = APIRouter()


# --- Request/Response Models ---


class CreatePromptRequest(BaseModel):
    """Request body for creating a new prompt."""

    prompt_id: str  # unique identifier
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
    name: str
    description: str | None
    latest_version_id: str | None
    version_count: int
    created_at: str
    updated_at: str


def _generate_version_id(prompt_id: str) -> str:
    """Generate the next version ID for a prompt."""
    versions = db_list_versions(prompt_id)
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
    rows = db_list_prompts()
    items = []
    for row in rows:
        versions = db_list_versions(row.prompt_id)
        items.append(
            PromptListItem(
                prompt_id=row.prompt_id,
                name=row.name,
                description=row.description,
                latest_version_id=row.latest_version_id,
                version_count=len(versions),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )
    return items


@router.post("/prompts")
def create_prompt(request: CreatePromptRequest) -> Prompt:
    """Create a new prompt."""
    # Check if prompt already exists
    if db_get_prompt(request.prompt_id):
        raise HTTPException(
            status_code=409,
            detail=f"Prompt already exists: {request.prompt_id}",
        )
    
    now = utc_timestamp()
    prompt = Prompt(
        prompt_id=request.prompt_id,
        name=request.name,
        description=request.description,
        created_at=now,
        updated_at=now,
        latest_version_id=None,
    )
    
    db_create_prompt(prompt)
    return prompt


@router.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: str) -> PromptWithVersions:
    """Get prompt details with all versions."""
    prompt = db_get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    versions = db_list_versions(prompt_id)
    
    return PromptWithVersions(prompt=prompt, versions=versions)


@router.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str) -> dict:
    """Delete a prompt and all its versions."""
    if not db_get_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    db_delete_prompt(prompt_id)
    
    return {"deleted": prompt_id}


@router.post("/prompts/{prompt_id}/versions")
def create_version(prompt_id: str, request: CreateVersionRequest) -> PromptVersion:
    """Create a new version of a prompt (immutable once created)."""
    # Verify prompt exists
    prompt = db_get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    
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
    
    db_insert_version(version)
    
    # Update prompt metadata with latest version
    prompt.latest_version_id = version_id
    prompt.updated_at = now
    db_update_prompt(prompt)
    
    return version


@router.get("/prompts/{prompt_id}/versions/{version_id}")
def get_version(prompt_id: str, version_id: str) -> PromptVersion:
    """Get a specific prompt version."""
    # Verify prompt exists
    if not db_get_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    version = db_get_version(prompt_id, version_id)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Version not found: {prompt_id}/{version_id}",
        )
    return version


@router.get("/prompts/{prompt_id}/versions/{version_id}/resolve")
def resolve_version(prompt_id: str, version_id: str) -> dict:
    """Get the resolved prompt text for a version.
    
    Returns the concatenated text of all enabled components.
    """
    # Verify prompt exists
    if not db_get_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    version = db_get_version(prompt_id, version_id)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Version not found: {prompt_id}/{version_id}",
        )
    
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
    prompt = db_get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    version = db_get_latest_version(prompt_id)
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"No versions exist for prompt: {prompt_id}",
        )
    return version
