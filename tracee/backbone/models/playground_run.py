"""Playground run model for single LLM invocations.

A playground run is a simple: prompt + model + variables → output.
For MAS traces, see the trace events stored in JSONL files.
"""

from pydantic import BaseModel


class PlaygroundRun(BaseModel):
    """Record of a single LLM invocation in the playground.

    Used for testing and iterating on prompts.
    Each run we store the exact prompt, model configuration, and output.
    """

    # protected_namespaces() removes field naming pretections from Pydantic ("model_")
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # identifiers
    run_id: str
    created_at: str

    # prompt reference (which prompt version was used)
    prompt_id: str
    version_id: str

    model: str  # e.g., "gpt-4", "claude-3-sonnet"
    provider: str  # e.g., "openai", "anthropic"
    temperature: float = 0.7
    max_tokens: int | None = None

    # IO
    input_variables: dict[str, str]  # Template variable substitutions
    resolved_prompt: str  # The final prompt text sent to LLM
    output: str  # LLM response

    # metadata
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # optional: link to model config if using a saved configuration
    model_config_id: str | None = None

    # user info
    created_by: str | None = None
    tags: list[str] | None = None
    notes: str | None = None  # User notes about this run


class PlaygroundRunCreate(BaseModel):
    """This is the request data model (created by an user so input only)
    fields are pretty similar as above"""

    model_config = {"protected_namespaces": ()}

    # prompt reference
    prompt_id: str
    version_id: str = "latest"
    input_variables: dict[str, str] = {}

    # either reference a saved config or provide inline
    model_config_id: str | None = None  # Use saved config

    # Inline model config (used if model_config_id is None)
    model: str = "gpt-4"
    provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int | None = None

    # optional metadata
    tags: list[str] | None = None
    notes: str | None = None
