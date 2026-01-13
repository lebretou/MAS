"""Execution record models for the execution layer."""

from typing import Literal, Self

from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    """Configuration for the model used in an execution."""

    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    seed: int | None = None


class PromptArtifactRef(BaseModel):
    """Reference to a versioned prompt artifact."""

    prompt_id: str
    version_id: str
    agent_id: str | None = None


class ContractRef(BaseModel):
    """Reference to a versioned contract."""

    contract_id: str
    contract_version: str
    agent_id: str | None = None


class ExecutionRecord(BaseModel):
    """A factual snapshot of one execution, regardless of where it ran."""

    model_config = {"extra": "forbid"}

    execution_id: str
    trace_id: str  # required, never None
    origin: Literal["playground", "prod", "sdk", "batch_eval"]
    created_at: str

    llm_config: ModelConfig  # renamed to avoid conflict with pydantic's model_config
    input_payload: dict
    context_payload: dict | None = None

    resolved_prompt_text: str  # required, non-empty

    prompt_refs: list[PromptArtifactRef] | None = None
    contract_refs: list[ContractRef] | None = None

    # environment context
    git_commit: str | None = None
    app_version: str | None = None
    env: Literal["dev", "staging", "prod"] | None = None

    tags: list[str] | None = None

    @model_validator(mode="after")
    def validate_resolved_prompt(self) -> Self:
        if not self.resolved_prompt_text.strip():
            raise ValueError("resolved_prompt_text must not be empty")
        return self
