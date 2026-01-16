"""Prompt artifact models for the authoring layer."""

from enum import Enum

from pydantic import BaseModel


class PromptComponentType(str, Enum):
    """Types of prompt components that can be authored."""

    # tentative
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


class Prompt(BaseModel):
    """A prompt artifact that can have multiple immutable versions.
    
    This is the parent container for prompt versions. Each prompt is
    associated with an agent and can have multiple versions over time.
    """

    prompt_id: str  # unique identifier, e.g., "planner-prompt"
    agent_name: str  # which agent uses this, e.g., "planner"
    name: str  # human-readable display name
    description: str | None = None
    created_at: str
    updated_at: str
    latest_version_id: str | None = None  # track the latest version


class PromptVersion(BaseModel):
    """A versioned, immutable prompt artifact created in the Playground.
    
    Once created, a PromptVersion should not be modified. Instead, create
    a new version with changes.
    """

    prompt_id: str  # links to parent Prompt
    version_id: str  # e.g., "v1", "v2", or a UUID
    name: str  # version-specific name/label
    components: list[PromptComponent]
    variables: dict[str, str] | None = None
    created_at: str

    def resolve(self) -> str:
        """Resolve the prompt to a single text string.
        
        Concatenates all enabled components into a single prompt text.
        """
        return "\n\n".join(
            component.content
            for component in self.components
            if component.enabled
        )
