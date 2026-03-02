"""Data model for prompts created in the playground"""

import json
from enum import Enum

from pydantic import BaseModel, model_validator


class PromptComponentType(str, Enum):
    """
    Currently the exact components here are tentative, but we aer likely going 
    to support multi-block editing so that users don't have to deal with a monotonic long text string
    Zhongzheng will confirm on the exact components soon 
    """

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
    """A prompt artifact that can have multiple versions.
    
    This is the parent container for prompt versions. 
    Users retrieve prompts by prompt_id and version_id in their codebase.
    """

    prompt_id: str  # unique identifier, e.g., "planner-prompt"
    name: str  # human-readable display name

    description: str | None = None # optional

    created_at: str
    updated_at: str

    latest_version_id: str | None = None  # track the latest version


class PromptVersion(BaseModel):
    """A versioned of the prompt created in the Playground.
    
    Once created, a PromptVersion should not be modified. Instead, create
    a new version with changes.
    """

    prompt_id: str  # links to parent Prompt

    version_id: str  # e.g., "v1", "v2", or a UUID
    name: str  # version-specific name/label

    components: list[PromptComponent]
    variables: dict[str, str] | None = None # we will support placeholders/variables
    output_schema: dict | None = None  # standard JSON Schema for structured LLM output
    created_at: str

    @model_validator(mode="after")
    def validate_output_schema(self) -> "PromptVersion":
        if self.output_schema is not None:
            if "type" not in self.output_schema or "properties" not in self.output_schema:
                raise ValueError(
                    "output_schema must have top-level 'type' and 'properties' keys"
                )
        return self

    def resolve(self) -> str:
        """Resolve the prompt to a single text string.

        If output_schema is set, appends a JSON Schema block instructing the LLM
        to return a conforming JSON object.
        """
        text = "\n\n".join(
            component.content
            for component in self.components
            if component.enabled
        )
        if self.output_schema is not None:
            schema_block = (
                "Respond with a JSON object that conforms to the following JSON Schema:\n"
                "```json\n"
                + json.dumps(self.output_schema, indent=2)
                + "\n```"
            )
            text = text + "\n\n" + schema_block
        return text
