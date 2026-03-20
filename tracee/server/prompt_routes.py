"""API routes for prompt management."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backbone.models.prompt_artifact import (
    Prompt,
    PromptComponent,
    PromptComponentType,
    PromptTemplate,
    PromptTemplateField,
    PromptTool,
    PromptVersion,
)
from backbone.utils.identifiers import utc_timestamp
from server.prompt_db import (
    create_prompt as db_create_prompt,
    delete_prompt as db_delete_prompt,
    get_latest_version as db_get_latest_version,
    get_prompt as db_get_prompt,
    list_prompt_version_counts as db_list_prompt_version_counts,
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


class UpdatePromptRequest(BaseModel):
    """Request body for updating prompt metadata."""

    name: str
    description: str | None = None


class CreateVersionRequest(BaseModel):
    """Request body for creating a new prompt version."""

    name: str  # version label/name
    components: list[PromptComponent]
    variables: dict[str, str] | None = None
    output_schema: dict | None = None
    tools: list[PromptTool] = Field(default_factory=list)
    parent_version_id: str | None = None
    branch_name: str | None = None
    revision_note: str | None = None
    source_template_id: str | None = None


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


def _default_prompt_templates() -> list[PromptTemplate]:
    return [
        PromptTemplate(
            template_id="planner-archetype",
            name="Planner agent",
            description="Break work into steps, plan execution, and keep the output bounded.",
            archetype="planner",
            fields=[
                PromptTemplateField(
                    field_id="system_role",
                    label="System role",
                    placeholder="You are a planning agent for a multi-agent workflow.",
                    default_value="You are a planning agent for a multi-agent workflow.",
                ),
                PromptTemplateField(
                    field_id="goal",
                    label="Primary goal",
                    placeholder="Describe what the planner should accomplish.",
                ),
                PromptTemplateField(
                    field_id="inputs",
                    label="Inputs",
                    placeholder="What context or variables does the planner receive?",
                ),
                PromptTemplateField(
                    field_id="outputs",
                    label="Outputs",
                    placeholder="What should the planner return?",
                ),
                PromptTemplateField(
                    field_id="constraints",
                    label="Constraints",
                    placeholder="Any boundaries or non-goals for the planner.",
                    required=False,
                ),
            ],
            components=[
                PromptComponent(
                    type=PromptComponentType.role,
                    content="{{system_role}}",
                ),
                PromptComponent(
                    type=PromptComponentType.goal,
                    content="Goal:\n{{goal}}",
                ),
                PromptComponent(
                    type=PromptComponentType.inputs,
                    content="Available inputs:\n{{inputs}}",
                ),
                PromptComponent(
                    type=PromptComponentType.outputs,
                    content="Return:\n{{outputs}}",
                ),
                PromptComponent(
                    type=PromptComponentType.constraints,
                    content="Constraints:\n{{constraints}}",
                ),
            ],
        ),
        PromptTemplate(
            template_id="tool-user-archetype",
            name="Tool-using agent",
            description="Use tools deliberately and explain when tool usage is appropriate.",
            archetype="tool_user",
            fields=[
                PromptTemplateField(
                    field_id="system_role",
                    label="System role",
                    default_value="You are an execution agent that may use tools when needed.",
                ),
                PromptTemplateField(
                    field_id="goal",
                    label="Primary goal",
                    placeholder="Describe the task the agent should solve.",
                ),
                PromptTemplateField(
                    field_id="tool_policy",
                    label="Tool policy",
                    placeholder="When should the agent use tools versus reason directly?",
                ),
                PromptTemplateField(
                    field_id="output_contract",
                    label="Output contract",
                    placeholder="How should the answer be formatted?",
                ),
            ],
            components=[
                PromptComponent(type=PromptComponentType.role, content="{{system_role}}"),
                PromptComponent(type=PromptComponentType.task, content="Task:\n{{goal}}"),
                PromptComponent(
                    type=PromptComponentType.tool_instructions,
                    content="Tool usage policy:\n{{tool_policy}}",
                ),
                PromptComponent(
                    type=PromptComponentType.outputs,
                    content="Output contract:\n{{output_contract}}",
                ),
            ],
        ),
        PromptTemplate(
            template_id="critic-archetype",
            name="Critic agent",
            description="Review work, identify issues, and explain the reasoning behind the critique.",
            archetype="critic",
            fields=[
                PromptTemplateField(
                    field_id="system_role",
                    label="System role",
                    default_value="You are a critical reviewer in a multi-agent workflow.",
                ),
                PromptTemplateField(
                    field_id="review_target",
                    label="Review target",
                    placeholder="What artifact or output is being reviewed?",
                ),
                PromptTemplateField(
                    field_id="evaluation_criteria",
                    label="Evaluation criteria",
                    placeholder="What should the critic optimize for?",
                ),
                PromptTemplateField(
                    field_id="response_format",
                    label="Response format",
                    placeholder="How should the review be structured?",
                ),
            ],
            components=[
                PromptComponent(type=PromptComponentType.role, content="{{system_role}}"),
                PromptComponent(
                    type=PromptComponentType.task,
                    content="Review this target:\n{{review_target}}",
                ),
                PromptComponent(
                    type=PromptComponentType.constraints,
                    content="Evaluate using these criteria:\n{{evaluation_criteria}}",
                ),
                PromptComponent(
                    type=PromptComponentType.outputs,
                    content="Respond in this format:\n{{response_format}}",
                ),
            ],
        ),
    ]


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


@router.get("/prompt-templates")
def list_prompt_templates() -> list[PromptTemplate]:
    """List built-in prompt templates and agent archetypes."""
    return _default_prompt_templates()


@router.get("/prompts")
def list_prompts() -> list[PromptListItem]:
    """List all prompts with metadata."""
    rows = db_list_prompts()
    version_counts = db_list_prompt_version_counts()
    items = []
    for row in rows:
        items.append(
            PromptListItem(
                prompt_id=row.prompt_id,
                name=row.name,
                description=row.description,
                latest_version_id=row.latest_version_id,
                version_count=version_counts.get(row.prompt_id, 0),
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


@router.patch("/prompts/{prompt_id}")
def update_prompt(prompt_id: str, request: UpdatePromptRequest) -> Prompt:
    """Update prompt metadata without changing versions."""
    prompt = db_get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")

    prompt.name = request.name
    prompt.description = request.description
    prompt.updated_at = utc_timestamp()
    db_update_prompt(prompt)
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
    
    parent_version = None
    if request.parent_version_id:
        parent_version = db_get_version(prompt_id, request.parent_version_id)
        if not parent_version:
            raise HTTPException(
                status_code=404,
                detail=f"Parent version not found: {prompt_id}/{request.parent_version_id}",
            )

    # Generate version ID
    version_id = _generate_version_id(prompt_id)
    now = utc_timestamp()
    branch_name = request.branch_name or (parent_version.branch_name if parent_version else "main")
    
    version = PromptVersion(
        prompt_id=prompt_id,
        version_id=version_id,
        name=request.name,
        parent_version_id=request.parent_version_id,
        root_version_id=parent_version.root_version_id if parent_version else version_id,
        branch_name=branch_name,
        components=request.components,
        variables=request.variables,
        output_schema=request.output_schema,
        tools=request.tools,
        revision_note=request.revision_note,
        source_template_id=request.source_template_id,
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
        "tool_count": len(version.tools),
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
