"""Data models for the cognition layer.

The cognition layer uses LLM to produce objective, descriptive summaries
of multi-agent trace executions — per-node descriptions with handoff context
and a trace-level narrative.
"""

from typing import Any

from pydantic import BaseModel


class NodeCognition(BaseModel):
    """objective description of a single agent node's execution."""

    agent_id: str
    description: str  # objective description, may contain {tool:name} {state:key} {agent:id} tags
    handoff_description: str = ""  # brief factual description of what was received from upstream


class TraceCognition(BaseModel):
    """full cognition result for an entire trace."""

    trace_id: str
    graph_id: str | None = None
    node_cognitions: dict[str, NodeCognition]  # keyed by agent_id
    narrative: str  # descriptive summary with inline chip tags, no verdicts
    created_at: str


class CognitionLog(BaseModel):
    """raw LLM input/output for auditing."""

    trace_id: str
    agent_id: str | None = None  # None for trace-level call
    llm_input: str
    llm_output: str
    model: str | None = None
    tokens_used: int | None = None
    created_at: str


class NodeSegment(BaseModel):
    """extracted execution segment for a single agent node, used as LLM input."""

    agent_id: str
    upstream_agents: list[str] = []
    input_state: dict[str, Any] | None = None
    output_state: dict[str, Any] | None = None
    changed_keys: list[str] = []
    operations: list[dict[str, Any]] = []  # [{type, id, label, input, output, ...}]
