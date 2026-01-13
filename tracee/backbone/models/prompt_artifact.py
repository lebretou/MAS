"""Prompt artifact models for the authoring layer."""

from enum import Enum

from pydantic import BaseModel


class PromptComponentType(str, Enum):
    """Types of prompt components that can be authored."""

    role = "role"
    goal = "goal"
    constraints = "constraints"
    io_rules = "io_rules"
    examples = "examples"
    safety = "safety"
    tool_instructions = "tool_instructions"


class PromptComponent(BaseModel):
    """A single component of a structured prompt."""

    type: PromptComponentType
    content: str
    enabled: bool = True


class PromptVersion(BaseModel):
    """A versioned, immutable prompt artifact created in the Playground."""

    prompt_id: str
    version_id: str
    name: str
    components: list[PromptComponent]
    variables: dict[str, str] | None = None
    created_at: str
