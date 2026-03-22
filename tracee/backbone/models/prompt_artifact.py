"""Data model for prompts created in the playground"""

import json
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from backbone.utils.identifiers import generate_config_id


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
    custom = "custom"


DEFAULT_COMPONENT_NAMES: dict[PromptComponentType, str] = {
    PromptComponentType.role: "Role",
    PromptComponentType.goal: "Goal",
    PromptComponentType.constraints: "Constraints",
    PromptComponentType.task: "Task",
    PromptComponentType.io_rules: "I/O Rules",
    PromptComponentType.inputs: "Inputs",
    PromptComponentType.outputs: "Outputs",
    PromptComponentType.examples: "Examples",
    PromptComponentType.safety: "Safety",
    PromptComponentType.tool_instructions: "Tool Instructions",
    PromptComponentType.external_information: "External Information",
    PromptComponentType.custom: "Custom Section",
}

DEFAULT_MESSAGE_ROLES: dict[PromptComponentType, "PromptMessageRole"] = {
    PromptComponentType.role: "system",
    PromptComponentType.goal: "system",
    PromptComponentType.constraints: "system",
    PromptComponentType.io_rules: "system",
    PromptComponentType.outputs: "system",
    PromptComponentType.safety: "system",
    PromptComponentType.tool_instructions: "system",
    PromptComponentType.custom: "system",
    PromptComponentType.task: "human",
    PromptComponentType.inputs: "human",
    PromptComponentType.external_information: "human",
    PromptComponentType.examples: "ai",
}


class PromptMessageRole(str, Enum):
    """Execution-layer role for chat-style prompts."""

    system = "system"
    human = "human"
    ai = "ai"


class PromptComponent(BaseModel):
    """A single component of a structured prompt."""

    component_id: str | None = None
    type: PromptComponentType
    name: str | None = None
    message_role: PromptMessageRole | None = None
    content: str
    enabled: bool = True

    @model_validator(mode="after")
    def normalize_component_name(self) -> "PromptComponent":
        if self.name is not None:
            normalized_name = self.name.strip()
            self.name = normalized_name or None
        return self

    def display_name(self) -> str:
        return self.name or DEFAULT_COMPONENT_NAMES[self.type]

    def resolved_message_role(self) -> PromptMessageRole:
        return self.message_role or PromptMessageRole(DEFAULT_MESSAGE_ROLES[self.type])


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

    argument_id: str | None = None
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

    tool_id: str | None = None
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


class PromptTemplateField(BaseModel):
    """A structured decision field used to scaffold prompt creation."""

    field_id: str
    label: str
    description: str | None = None
    input_type: str = "textarea"
    required: bool = True
    placeholder: str | None = None
    default_value: str = ""


class PromptTemplate(BaseModel):
    """A reusable prompt-start scaffold for common MAS roles."""

    template_id: str
    name: str
    description: str | None = None
    archetype: str | None = None
    fields: list[PromptTemplateField] = Field(default_factory=list)
    components: list[PromptComponent] = Field(default_factory=list)
    suggested_tools: list[PromptTool] = Field(default_factory=list)
    suggested_output_schema: dict | None = None


class GuidedStartStage(str, Enum):
    """Ordered stages used by guided prompt authoring."""

    role = "role"
    questions = "questions"
    tools = "tools"
    schema = "schema"
    review = "review"


class GuidedStartQuestion(BaseModel):
    """A starter question shown during guided prompt setup."""

    question_id: str
    label: str
    description: str | None = None
    input_type: str = "textarea"
    required: bool = True
    placeholder: str | None = None
    default_value: str = ""


class GuidedStartSuggestedComponent(BaseModel):
    """A suggested prompt component with evidence metadata."""

    component_type: PromptComponentType
    title: str
    prevalence: float
    order_rank: int
    content_template: str


class GuidedStartArchetype(BaseModel):
    """Curated guided-start metadata for a supported archetype."""

    archetype_id: str
    title: str
    summary: str
    example_jobs: list[str] = Field(default_factory=list)
    sample_size: int
    starter_questions: list[GuidedStartQuestion] = Field(default_factory=list)
    suggested_components: list[GuidedStartSuggestedComponent] = Field(default_factory=list)
    suggested_tools: list[PromptTool] = Field(default_factory=list)
    suggested_output_schema: dict | None = None


class GuidedStartCatalog(BaseModel):
    """Source-of-truth data for guided-start archetypes."""

    version: str
    generated_at: str
    fallback_questions: list[GuidedStartQuestion] = Field(default_factory=list)
    fallback_components: list[GuidedStartSuggestedComponent] = Field(default_factory=list)
    archetypes: list[GuidedStartArchetype] = Field(default_factory=list)


class GuidedStartConversationTurn(BaseModel):
    """A user-visible turn in the guided-start conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class GuidedStartLlmRequest(BaseModel):
    """Request payload for guided-start LLM refinement."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    temperature: float = 0
    stage: GuidedStartStage
    selected_archetype: str | None = Field(default=None, max_length=64)
    custom_role: str | None = Field(default=None, max_length=1000)
    answers: dict[str, str] = Field(default_factory=dict, max_length=16)
    current_draft: list[PromptComponent] = Field(default_factory=list, max_length=12)
    conversation_history: list[GuidedStartConversationTurn] = Field(default_factory=list, max_length=24)
    latest_user_turn: str = Field(min_length=1, max_length=4000)


class GuidedStartLlmResponse(BaseModel):
    """Structured response contract for guided-start assistance."""

    assistant_message: str
    selected_archetype: str | None = None
    selected_archetype_title: str | None = None
    component_draft: list[PromptComponent] = Field(default_factory=list)
    current_stage: GuidedStartStage
    follow_up_questions: list[str] = Field(default_factory=list)
    stage_complete: bool = False
    status: Literal["needs_input", "ready_for_next_stage", "ready_to_apply"]
    updated_component_types: list[PromptComponentType] = Field(default_factory=list)


class PromptVersion(BaseModel):
    """A versioned of the prompt created in the Playground.
    
    Once created, a PromptVersion should not be modified. Instead, create
    a new version with changes.
    """

    prompt_id: str  # links to parent Prompt

    version_id: str  # e.g., "v1", "v2", or a UUID
    name: str  # version-specific name/label
    parent_version_id: str | None = None
    root_version_id: str | None = None
    branch_id: str | None = None
    branch_name: str | None = None
    revision_note: str | None = None
    source_template_id: str | None = None

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
        for component in self.components:
            if not component.component_id:
                component.component_id = generate_config_id()
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique within a prompt version")
        for tool in self.tools:
            if not tool.tool_id:
                tool.tool_id = generate_config_id()
            for argument in tool.arguments:
                if not argument.argument_id:
                    argument.argument_id = generate_config_id()
        if self.parent_version_id and self.root_version_id is None:
            raise ValueError("root_version_id is required when parent_version_id is set")
        if self.root_version_id is None:
            self.root_version_id = self.version_id
        if self.branch_id is None:
            branch_name = self.branch_name or "main"
            self.branch_id = f"{self.prompt_id}:{branch_name}"
        if self.branch_name is None:
            self.branch_name = "main"
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
            f"{component.display_name()}:\n{component.content}"
            if component.content
            else f"{component.display_name()}:"
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
