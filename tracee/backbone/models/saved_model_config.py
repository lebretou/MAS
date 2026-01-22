"""Saved model configuration for reusable LLM settings.

Users can choose to create configurations in the playground and reuse them
"""

from pydantic import BaseModel


class SavedModelConfig(BaseModel):
    """
    model config data model
    """

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # id
    config_id: str
    name: str  # e.g., "Creative GPT-4", "Fast Claude"
    description: str | None = None

    provider: str  # "openai", "anthropic", "google", etc.
    model_name: str  # "gpt-4", "gpt-4-turbo", "claude-3-sonnet", etc.
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None

    # provider-specific extras (e.g., frequency_penalty, presence_penalty)
    extra_params: dict | None = None

    created_at: str
    updated_at: str

    is_default: bool = False  # Mark one config as the default
    tags: list[str] | None = None


class SavedModelConfigCreate(BaseModel):
    """Request model for creating a saved model configuration."""

    model_config = {"protected_namespaces": ()}

    name: str
    description: str | None = None

    provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None

    extra_params: dict | None = None
    is_default: bool = False
    tags: list[str] | None = None


class SavedModelConfigUpdate(BaseModel):
    """Request model for updating a saved model configuration."""

    model_config = {"protected_namespaces": ()}

    name: str | None = None
    description: str | None = None

    provider: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    extra_params: dict | None = None
    is_default: bool | None = None
    tags: list[str] | None = None
