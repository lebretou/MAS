"""Data model for prompts created in the playground"""

import json
import re
from enum import Enum

from pydantic import BaseModel, Field, model_validator


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


class ToolArgumentType(str, Enum):
    """Supported primitive argument types for prompt-authored tools."""

    string = "string"
    number = "number"
    integer = "integer"
    boolean = "boolean"
    array = "array"
    object = "object"


class PromptToolArgument(BaseModel):
    """A single tool argument exposed to the model."""

    name: str
    description: str | None = None
    type: ToolArgumentType = ToolArgumentType.string
    required: bool = False
    allowed_values: list[str] | None = None

    @model_validator(mode="after")
    def validate_argument(self) -> "PromptToolArgument":
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(
                "tool argument name must start with a letter or underscore and contain only letters, numbers, and underscores"
            )
        if self.allowed_values and self.type != ToolArgumentType.string:
            raise ValueError("allowed_values are only supported for string tool arguments")
        return self


class PromptTool(BaseModel):
    """A custom tool definition authored alongside a prompt version."""

    name: str
    description: str
    arguments: list[PromptToolArgument] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool(self) -> "PromptTool":
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError(
                "tool name must be 1-64 chars and contain only letters, numbers, underscores, or hyphens"
            )
        if self.name == "structured_output":
            raise ValueError("tool name 'structured_output' is reserved")

        argument_names = [argument.name for argument in self.arguments]
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("tool argument names must be unique within a tool")
        return self

    def input_schema(self) -> dict:
        """Convert the tool definition to a JSON Schema object."""
        properties: dict[str, dict] = {}
        required: list[str] = []

        for argument in self.arguments:
            schema: dict = {"type": argument.type.value}
            if argument.description:
                schema["description"] = argument.description
            if argument.type == ToolArgumentType.array:
                schema["items"] = {"type": "string"}
            if argument.type == ToolArgumentType.object:
                schema["additionalProperties"] = True
            if argument.allowed_values:
                schema["enum"] = argument.allowed_values
            properties[argument.name] = schema
            if argument.required:
                required.append(argument.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


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
    tools: list[PromptTool] = Field(default_factory=list)
    created_at: str

    @model_validator(mode="after")
    def validate_output_schema(self) -> "PromptVersion":
        if self.output_schema is not None:
            if "type" not in self.output_schema or "properties" not in self.output_schema:
                raise ValueError(
                    "output_schema must have top-level 'type' and 'properties' keys"
                )
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique within a prompt version")
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
