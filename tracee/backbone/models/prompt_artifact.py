"""Data model for prompts created in the playground"""

import json
from enum import Enum

from pydantic import BaseModel, model_validator


class SchemaMode(str, Enum):
    """Controls how output_schema is included in the resolved prompt text."""

    full = "full"    # embed full JSON Schema block in prompt (default)
    hint = "hint"    # append a short generic hint only
    none = "none"    # no schema mention at all


class PromptComponentType(str, Enum):
    """
    Currently the exact components here are tentative, but we aer likely going
    to support multi-block editing so that users don't have to deal with a monotonic long text string
    Zhongzheng will confirm on the exact components soon
    """

    role = "role"
    goal = "goal"
    constraints = "constraints"
    task = "task"
    io_rules = "io_rules"
    inputs = "inputs"
    outputs = "outputs"
    examples = "examples"
    safety = "safety"
    tool_instructions = "tool_instructions"
    external_information = "external_information"


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
    output_schema: dict | None = None  # JSON Schema for structured LLM output
    created_at: str

    @model_validator(mode="after")
    def validate_output_schema(self) -> "PromptVersion":
        if self.output_schema is not None:
            if "type" not in self.output_schema or "properties" not in self.output_schema:
                raise ValueError(
                    "output_schema must have top-level 'type' and 'properties' keys"
                )
        return self

    def resolve(self, *, schema_mode: SchemaMode = SchemaMode.full) -> str:
        """Resolve the prompt to a single text string.

        Args:
            schema_mode: Controls how output_schema appears in the prompt.
                - full: embeds the full JSON Schema block (default, backward compatible)
                - hint: appends a short generic hint
                - none: no schema mention at all
        """
        text = "\n\n".join(
            component.content
            for component in self.components
            if component.enabled
        )
        if self.output_schema is not None:
            if schema_mode == SchemaMode.full:
                schema_block = (
                    "Respond with a JSON object that conforms to the following JSON Schema:\n"
                    "```json\n"
                    + json.dumps(self.output_schema, indent=2)
                    + "\n```"
                )
                text = text + "\n\n" + schema_block
            elif schema_mode == SchemaMode.hint:
                text = text + "\n\nYour response should be structured JSON matching the requested schema."
        return text
