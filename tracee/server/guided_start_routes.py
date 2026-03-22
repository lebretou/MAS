"""api routes for guided-start catalog and llm refinement."""

import json

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from backbone.models.prompt_artifact import (
    GuidedStartArchetype,
    GuidedStartCatalog,
    GuidedStartLlmRequest,
    GuidedStartLlmResponse,
    GuidedStartStage,
)
from server.guided_start_service import (
    build_guided_start_draft,
    get_guided_start_archetype,
    infer_guided_start_archetype,
    load_guided_start_catalog,
    merge_guided_draft,
)
from server.llm_clients import call_llm_messages

router = APIRouter()


def _build_stage_instruction(stage: GuidedStartStage) -> str:
    if stage == GuidedStartStage.questions:
        return (
            "stay in the prompt-question stage. ask only for missing information needed to shape "
            "the prompt components. if the current draft is specific enough, mark the stage complete."
        )
    if stage == GuidedStartStage.review:
        return (
            "stay in the review stage. refine wording, tighten structure, and preserve confirmed "
            "intent. do not invent new scope unless the user explicitly asks for it."
        )
    return "stay within the current stage and keep the response bounded to that work."


def _build_guided_start_system_prompt() -> str:
    return (
        "You are a prompt-design coach for MAS developers using Tracee's guided start.\n"
        "You are not the runtime agent being authored.\n"
        "Follow the product rules exactly:\n"
        "- keep the conversation inside the fixed guided-start workflow\n"
        "- work over structured prompt components, never a hidden freeform prompt blob\n"
        "- preserve confirmed user intent and avoid rewriting settled sections unless asked\n"
        "- ask focused follow-up questions only for the current stage when information is missing\n"
        "- keep assistant copy concise, practical, and evidence-oriented\n"
        "- support custom roles by mapping to the closest scaffold when possible\n"
        "- return only structured JSON matching the required schema\n"
        "- when updating the draft, prefer these component types: role, task, inputs, outputs, constraints, tool_instructions, examples, external_information, safety\n"
    )


def _resolve_archetype(archetype_id: str | None) -> GuidedStartArchetype | None:
    if not archetype_id:
        return None
    return get_guided_start_archetype(archetype_id)


@router.get("/guided-start/catalog", response_model=GuidedStartCatalog)
def get_guided_start_catalog() -> GuidedStartCatalog:
    """Return the guided-start reference catalog."""
    return load_guided_start_catalog()


@router.get("/guided-start/archetypes/{archetype_id}", response_model=GuidedStartArchetype)
def get_guided_start_archetype_route(archetype_id: str) -> GuidedStartArchetype:
    """Return a single guided-start archetype."""
    return get_guided_start_archetype(archetype_id)


@router.post("/guided-start/llm/respond", response_model=GuidedStartLlmResponse)
async def respond_guided_start(request: GuidedStartLlmRequest) -> GuidedStartLlmResponse:
    """Return a structured guided-start refinement response."""
    selected_archetype_id = (
        request.selected_archetype
        or infer_guided_start_archetype(request.custom_role)
        or infer_guided_start_archetype(request.latest_user_turn)
    )
    archetype = _resolve_archetype(selected_archetype_id)
    base_draft = build_guided_start_draft(
        archetype_id=selected_archetype_id,
        answers=request.answers,
        custom_role=request.custom_role,
    )
    working_draft = merge_guided_draft(base_draft, request.current_draft)
    archetype_slice = archetype.model_dump() if archetype else None
    catalog = load_guided_start_catalog()

    response = await call_llm_messages(
        messages=[
            {
                "role": "system",
                "content": _build_guided_start_system_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage_instruction": _build_stage_instruction(request.stage),
                        "current_stage": request.stage.value,
                        "selected_archetype": selected_archetype_id,
                        "custom_role": request.custom_role,
                        "fallback_questions": [item.model_dump() for item in catalog.fallback_questions],
                        "archetype_reference": archetype_slice,
                        "answers": request.answers,
                        "current_draft": [component.model_dump() for component in working_draft],
                        "conversation_history": [turn.model_dump() for turn in request.conversation_history],
                        "latest_user_turn": request.latest_user_turn,
                    },
                    indent=2,
                ),
            },
        ],
        model=request.model,
        provider=request.provider,
        temperature=request.temperature,
        max_tokens=2200,
        output_schema=GuidedStartLlmResponse.model_json_schema(),
        json_schema_strict=False,
    )

    try:
        parsed = GuidedStartLlmResponse.model_validate_json(response["content"])
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"guided start llm response was invalid: {exc.errors()}",
        )

    if not parsed.component_draft:
        parsed.component_draft = working_draft
    if parsed.selected_archetype is None:
        parsed.selected_archetype = selected_archetype_id
    if parsed.selected_archetype_title is None and archetype:
        parsed.selected_archetype_title = archetype.title
    return parsed
