"""helpers for guided-start catalog loading and scaffold generation."""

import json
import re
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

from fastapi import HTTPException

from backbone.models.prompt_artifact import (
    GuidedStartArchetype,
    GuidedStartCatalog,
    GuidedStartSuggestedComponent,
    PromptComponent,
    PromptTemplate,
    PromptTemplateField,
)

GUIDED_START_DATA_DIR = Path(__file__).parent / "data" / "guided_start"
GUIDED_START_CATALOG_PATH = GUIDED_START_DATA_DIR / "catalog.json"
LEGACY_TEMPLATE_IDS = {
    "planner": "planner-archetype",
    "tool_user": "tool-user-archetype",
    "evaluator": "critic-archetype",
}

_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "planner": ("planner", "plan", "coordinator", "orchestrator", "manager"),
    "evaluator": ("evaluator", "evaluate", "grader", "grade", "reviewer", "review", "critic", "judge", "validator", "verif"),
    "tool_user": ("tool", "search", "retriev", "executor", "action", "browser", "scraper"),
    "researcher": ("researcher", "research", "analyst", "analyzer"),
    "writer": ("writer", "generator", "creator", "composer", "author"),
    "coder": ("coder", "developer", "programmer", "code"),
}


@lru_cache(maxsize=1)
def load_guided_start_catalog() -> GuidedStartCatalog:
    """Load and validate the guided-start catalog."""
    if not GUIDED_START_CATALOG_PATH.exists():
        raise HTTPException(status_code=500, detail="Guided start catalog is missing.")
    try:
        return GuidedStartCatalog.model_validate(json.loads(GUIDED_START_CATALOG_PATH.read_text()))
    except JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Guided start catalog is invalid: {exc.msg}")


def list_guided_start_archetypes() -> list[GuidedStartArchetype]:
    """Return supported guided-start archetypes."""
    return load_guided_start_catalog().archetypes


def get_guided_start_archetype(archetype_id: str) -> GuidedStartArchetype:
    """Return a single guided-start archetype."""
    for archetype in load_guided_start_catalog().archetypes:
        if archetype.archetype_id == archetype_id:
            return archetype
    raise HTTPException(status_code=404, detail=f"Guided start archetype not found: {archetype_id}")


def infer_guided_start_archetype(role_text: str | None) -> str | None:
    """Infer the closest archetype from free-form role text."""
    if not role_text:
        return None
    lowered = role_text.lower()
    for archetype_id, keywords in _ROLE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return archetype_id
    return None


def _fill_template_value(template: str, values: dict[str, str]) -> str:
    return re.sub(
        r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
        lambda match: values.get(match.group(1), ""),
        template,
    )


def render_guided_components(
    suggestions: list[GuidedStartSuggestedComponent],
    values: dict[str, str],
) -> list[PromptComponent]:
    """Render ordered prompt components from guided suggestions."""
    rendered: list[PromptComponent] = []
    for suggestion in sorted(suggestions, key=lambda item: item.order_rank):
        content = _fill_template_value(suggestion.content_template, values).strip()
        if content:
            rendered.append(
                PromptComponent(
                    type=suggestion.component_type,
                    content=content,
                    enabled=True,
                )
            )
    return rendered


def build_guided_start_draft(
    *,
    archetype_id: str | None,
    answers: dict[str, str],
    custom_role: str | None = None,
) -> list[PromptComponent]:
    """Build a scaffold from the catalog and current answers."""
    catalog = load_guided_start_catalog()
    values = {key: value.strip() for key, value in answers.items() if value is not None}
    if custom_role and custom_role.strip():
        values.setdefault("custom_role", custom_role.strip())
    if archetype_id:
        archetype = get_guided_start_archetype(archetype_id)
        return render_guided_components(archetype.suggested_components, values)
    return render_guided_components(catalog.fallback_components, values)


def merge_guided_draft(
    base_draft: list[PromptComponent],
    current_draft: list[PromptComponent],
) -> list[PromptComponent]:
    """Overlay current edits onto a generated scaffold."""
    current_by_type = {component.type: component for component in current_draft}
    merged: list[PromptComponent] = []
    seen_types = set()
    for component in base_draft:
        override = current_by_type.get(component.type)
        merged.append(override or component)
        seen_types.add(component.type)
    for component in current_draft:
        if component.type not in seen_types:
            merged.append(component)
    return merged


def build_prompt_templates_from_catalog() -> list[PromptTemplate]:
    """Expose archetypes through the legacy prompt template endpoint."""
    templates: list[PromptTemplate] = []
    for archetype in list_guided_start_archetypes():
        template = PromptTemplate(
            template_id=f"guided-{archetype.archetype_id}",
            name=archetype.title,
            description=archetype.summary,
            archetype=archetype.archetype_id,
            fields=[
                PromptTemplateField(
                    field_id=question.question_id,
                    label=question.label,
                    description=question.description,
                    input_type=question.input_type,
                    required=question.required,
                    placeholder=question.placeholder,
                    default_value=question.default_value,
                )
                for question in archetype.starter_questions
            ],
            components=render_guided_components(archetype.suggested_components, {}),
            suggested_tools=archetype.suggested_tools,
            suggested_output_schema=archetype.suggested_output_schema,
        )
        templates.append(template)
        legacy_template_id = LEGACY_TEMPLATE_IDS.get(archetype.archetype_id)
        if legacy_template_id:
            templates.append(template.model_copy(update={"template_id": legacy_template_id}))
    return templates
