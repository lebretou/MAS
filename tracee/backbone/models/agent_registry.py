"""Data model for the agent registry.

Tracks which prompt each agent uses, along with optional metadata
like model config. Auto-populated by PromptLoader.get() when agent_id is provided.
"""

from pydantic import BaseModel


class AgentRegistryEntry(BaseModel):
    """a registered agent and its current configuration."""

    agent_id: str
    prompt_id: str | None = None
    prompt_version_id: str | None = None  # last resolved version
    model: str | None = None
    temperature: float | None = None
    has_tools: bool = False
    metadata: dict | None = None
    updated_at: str
