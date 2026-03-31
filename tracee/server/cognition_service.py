"""LLM service for the cognition layer.

Makes per-node LLM calls (in parallel) and a trace-level narrative call,
using function calling for structured output. All output is purely
descriptive and objective — no judgments or correctness assessments.
"""

import asyncio
import json
import logging
import os
from typing import Any

from backbone.models.cognition import (
    CognitionLog,
    NodeCognition,
    NodeSegment,
    TraceCognition,
)
from backbone.models.graph_topology import GraphTopology
from backbone.utils.identifiers import utc_timestamp
from server.llm_clients import get_openai_client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

# ── function schemas for structured output ──────────────────────

NODE_COGNITION_FUNCTION = {
    "type": "function",
    "function": {
        "name": "report_node_description",
        "description": "report an objective description of what this agent did",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "2-4 sentence objective description. MUST wrap every tool in {tool:name}, state key in {state:key}, agent in {agent:name}, and runtime error in {error:desc} tags. No plain-text references.",
                },
                "handoff_description": {
                    "type": "string",
                    "description": "one brief sentence describing what data this agent received from upstream. use {state:key} and {agent:name} tags. empty string if first node.",
                },
            },
            "required": ["description", "handoff_description"],
            "additionalProperties": False,
        },
    },
}

TRACE_NARRATIVE_FUNCTION = {
    "type": "function",
    "function": {
        "name": "report_trace_narrative",
        "description": "report a descriptive narrative of the entire workflow execution",
        "parameters": {
            "type": "object",
            "properties": {
                "narrative": {
                    "type": "string",
                    "description": "2-4 sentence descriptive narrative. MUST wrap every tool in {tool:name}, state key in {state:key}, agent in {agent:name}, and runtime error in {error:desc} tags. No plain-text references.",
                },
            },
            "required": ["narrative"],
            "additionalProperties": False,
        },
    },
}

# ── system prompts ──────────────────────────────────────────────

NODE_SYSTEM_PROMPT = """\
You are describing one step of a multi-agent workflow execution.
You will receive the workflow structure, this agent's role, what it
received, what operations it performed, and what it produced.

Describe objectively what this agent did. State what input it received,
what operations it performed, and what output it produced.

Do not judge correctness. Do not speculate about errors or what the
agent should have done differently. Do not use words like "failed",
"incorrect", "unnecessary", "overlooked", "missed", "should have",
"problematic", or "but". Simply describe what happened.

TAG FORMATTING RULES — you MUST follow these precisely:

1. {agent:name} — use for every reference to an agent in the workflow.
   Agents are listed in the "Workflow context" section.

2. {tool:name} — use ONLY for actual tool calls listed under "Operations
   performed" with type [tool_call], [rag_retrieve], or [code_exec].
   Do NOT tag LLM calls as tools. An LLM call is just "called the model"
   or "invoked the LLM", never {tool:gpt-4o}.

3. {state:key} — use for every state field key that appears in "Input
   state received", "Output state produced", or "State changes".
   These are the JSON keys like query, plan, code_result, etc.

4. {error:description} — use for any runtime error or exception that
   appeared in the operations (e.g. NameError, TypeError, timeout).

GOOD example:
"{agent:interaction} received {state:query} from the user. It called
{tool:search_columns} and {tool:get_dataset_info} to inspect the data,
then wrote {state:messages} with the findings. The query was routed to
{agent:planner} for further processing."

BAD example (DO NOT do this):
"The interaction agent received the query. It used gpt-4o tool to
generate a response and updated the messages state."
(Problems: agent not tagged, LLM call tagged as tool, state not tagged)"""

TRACE_SYSTEM_PROMPT = """\
You are describing the execution of a multi-agent workflow.
You will receive per-agent descriptions.

Write a concise narrative (2-4 sentences) describing what happened
across the workflow from start to finish. Be descriptive and objective.
Do not judge correctness or success. Do not speculate about what
should have happened differently.

TAG FORMATTING RULES — you MUST follow these precisely:
- {agent:name} for every agent reference
- {tool:name} for every tool call (NOT LLM/model calls)
- {state:key} for every state field key
- {error:description} for any runtime error or exception

Preserve the tags from the per-agent descriptions you receive. Do not
convert tagged references back to plain text.

GOOD: "{agent:planner} processed {state:query} and wrote {state:plan}.
{agent:coder} called {tool:python_repl} which raised {error:NameError}."

BAD: "The planner processed the query and wrote a plan." (no tags)"""


# ── prompt builders ─────────────────────────────────────────────

def _build_graph_context(
    graph: GraphTopology,
    agent_prompts: dict[str, str],
) -> str:
    """build the workflow context block shared across per-node calls."""
    agent_nodes = [n for n in graph.nodes if n.node_type == "agent"]
    node_ids = [n.node_id for n in agent_nodes]

    flow_parts = []
    for e in graph.edges:
        if e.source in node_ids or e.target in node_ids:
            flow_parts.append(f"{e.source} -> {e.target}")
    flow_str = ", ".join(flow_parts) if flow_parts else " -> ".join(node_ids)

    lines = [
        "## Workflow context",
        f"Agents: {', '.join(node_ids)}",
        f"Flow: {flow_str}",
        "",
        "Agent roles:",
    ]

    for node in agent_nodes:
        meta = node.metadata or {}
        model = meta.get("model", "unknown")
        tools = meta.get("tools", [])
        prompt_text = agent_prompts.get(node.node_id, "")
        if len(prompt_text) > 800:
            prompt_text = prompt_text[:800] + "..."
        prompt_display = f'"{prompt_text}"' if prompt_text else "(no prompt available)"
        tools_str = f", tools: [{', '.join(tools)}]" if tools else ""
        lines.append(f"- {node.node_id}: {prompt_display}")
        lines.append(f"  model: {model}{tools_str}")

    return "\n".join(lines)


def _build_node_user_message(
    segment: NodeSegment,
    graph_context: str,
    node_metadata: dict | None = None,
) -> str:
    """build the user message for a per-node LLM call."""
    meta = node_metadata or {}
    model = meta.get("model", "unknown")
    upstream_str = ", ".join(segment.upstream_agents) if segment.upstream_agents else "(none — first node)"

    # collect tool names from operations for the tag reference
    tool_names = sorted({
        op.get("label", "")
        for op in segment.operations
        if op.get("type") in ("tool_call", "rag_retrieve", "code_exec") and op.get("label")
    })
    # collect state keys from input/output
    state_keys = sorted({
        *(segment.input_state.keys() if segment.input_state else []),
        *(segment.output_state.keys() if segment.output_state else []),
    })

    lines = [
        graph_context,
        "",
        f"## Describing agent: {segment.agent_id}",
        f"## Model: {model}",
        f"## Upstream: {upstream_str}",
    ]

    if tool_names:
        lines.append(f"## Available tools (ONLY these can use {{tool:name}} tag): {', '.join(tool_names)}")
    if state_keys:
        lines.append(f"## State keys (use {{state:key}} tag for these): {', '.join(state_keys)}")
    lines.append("")

    if segment.input_state is not None:
        lines.append("## Input state received:")
        lines.append(json.dumps(segment.input_state, indent=2, default=str))
        lines.append("")

    if segment.operations:
        lines.append("## Operations performed:")
        for i, op in enumerate(segment.operations, 1):
            op_type = op.get("type", "unknown")
            op_id = op.get("id", "")
            label = op.get("label", op_type)
            lines.append(f"{i}. [{op_type}] (id: {op_id}) {label}")
            if op.get("input") is not None:
                input_str = json.dumps(op["input"], indent=2, default=str) if not isinstance(op["input"], str) else op["input"]
                if len(input_str) > 2000:
                    input_str = input_str[:2000] + "... (truncated)"
                lines.append(f"   Input: {input_str}")
            if op.get("output") is not None:
                output_str = json.dumps(op["output"], indent=2, default=str) if not isinstance(op["output"], str) else op["output"]
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "... (truncated)"
                lines.append(f"   Output: {output_str}")
            if op.get("error_message"):
                lines.append(f"   Error: {op['error_message']}")
            lines.append("")

    if segment.output_state is not None:
        lines.append("## Output state produced:")
        lines.append(json.dumps(segment.output_state, indent=2, default=str))
        lines.append("")

    if segment.changed_keys:
        lines.append(f"## State changes: {json.dumps(segment.changed_keys)}")

    return "\n".join(lines)


def _build_trace_user_message(
    node_cognitions: dict[str, NodeCognition],
    graph: GraphTopology,
) -> str:
    """build the user message for the trace-level narrative call."""
    agent_nodes = [n for n in graph.nodes if n.node_type == "agent"]
    node_ids = [n.node_id for n in agent_nodes]
    flow_str = " -> ".join(node_ids) + " -> END"

    lines = [
        f"## Workflow: {flow_str}",
        "",
        "## Per-agent descriptions:",
    ]

    for agent_id in node_ids:
        cog = node_cognitions.get(agent_id)
        if not cog:
            lines.append(f"### {agent_id}: (no data)")
            continue
        lines.append(f"### {agent_id}")
        lines.append(f"Description: {cog.description}")
        lines.append("")

    return "\n".join(lines)


# ── LLM call helpers ────────────────────────────────────────────

def _get_cognition_model() -> str:
    return os.getenv("TRACEE_COGNITION_MODEL", DEFAULT_MODEL)


async def _call_with_function(
    messages: list[dict[str, str]],
    function_def: dict,
    model: str | None = None,
) -> tuple[dict[str, Any], int]:
    """call OpenAI with a single function tool and return parsed args + total tokens."""
    client = get_openai_client()
    use_model = model or _get_cognition_model()

    response = await client.chat.completions.create(
        model=use_model,
        messages=messages,
        tools=[function_def],
        tool_choice={"type": "function", "function": {"name": function_def["function"]["name"]}},
        temperature=0.5,
    )

    message = response.choices[0].message
    total_tokens = response.usage.total_tokens if response.usage else 0

    if message.tool_calls and message.tool_calls[0].function.arguments:
        args = json.loads(message.tool_calls[0].function.arguments)
        return args, total_tokens

    if message.content:
        return json.loads(message.content), total_tokens

    return {}, total_tokens


# ── per-node analysis ───────────────────────────────────────────

async def analyze_node(
    segment: NodeSegment,
    graph_context: str,
    node_metadata: dict | None = None,
    model: str | None = None,
) -> tuple[NodeCognition, CognitionLog]:
    """run the LLM on a single agent node and return cognition + log."""
    user_message = _build_node_user_message(segment, graph_context, node_metadata)
    messages = [
        {"role": "system", "content": NODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    use_model = model or _get_cognition_model()
    args, total_tokens = await _call_with_function(messages, NODE_COGNITION_FUNCTION, use_model)

    cognition = NodeCognition(
        agent_id=segment.agent_id,
        description=args.get("description", ""),
        handoff_description=args.get("handoff_description", ""),
    )

    log = CognitionLog(
        trace_id="",
        agent_id=segment.agent_id,
        llm_input=json.dumps(messages, default=str),
        llm_output=json.dumps(args, default=str),
        model=use_model,
        tokens_used=total_tokens,
        created_at=utc_timestamp(),
    )

    return cognition, log


# ── trace-level narrative ───────────────────────────────────────

async def generate_trace_narrative(
    node_cognitions: dict[str, NodeCognition],
    graph: GraphTopology,
    model: str | None = None,
) -> tuple[str, CognitionLog]:
    """generate a descriptive narrative for the full trace."""
    user_message = _build_trace_user_message(node_cognitions, graph)
    messages = [
        {"role": "system", "content": TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    use_model = model or _get_cognition_model()
    args, total_tokens = await _call_with_function(messages, TRACE_NARRATIVE_FUNCTION, use_model)

    narrative = args.get("narrative", "")

    log = CognitionLog(
        trace_id="",
        agent_id=None,
        llm_input=json.dumps(messages, default=str),
        llm_output=json.dumps(args, default=str),
        model=use_model,
        tokens_used=total_tokens,
        created_at=utc_timestamp(),
    )

    return narrative, log


# ── orchestrator ────────────────────────────────────────────────

async def run_cognition_analysis(
    trace_id: str,
    events: list,
    graph: GraphTopology,
    agent_prompts: dict[str, str] | None = None,
) -> tuple[TraceCognition, list[CognitionLog]]:
    """run full cognition analysis on a trace: per-node + narrative."""
    from backbone.analysis.agent_analyzer import extract_all_segments

    agent_nodes = [n for n in graph.nodes if n.node_type == "agent"]
    agent_ids = [n.node_id for n in agent_nodes]
    edge_dicts = [{"source": e.source, "target": e.target} for e in graph.edges]

    node_meta: dict[str, dict] = {}
    for n in agent_nodes:
        node_meta[n.node_id] = n.metadata or {}

    segments = extract_all_segments(events, agent_ids, edge_dicts)
    graph_context = _build_graph_context(graph, agent_prompts or {})

    model = _get_cognition_model()
    tasks = []
    for agent_id in agent_ids:
        seg = segments.get(agent_id)
        if seg and seg.operations:
            tasks.append((agent_id, analyze_node(seg, graph_context, node_meta.get(agent_id), model)))
        else:
            tasks.append((agent_id, None))

    node_cognitions: dict[str, NodeCognition] = {}
    logs: list[CognitionLog] = []

    coros = [t[1] for t in tasks if t[1] is not None]
    ids_with_coros = [t[0] for t in tasks if t[1] is not None]

    if coros:
        results = await asyncio.gather(*coros, return_exceptions=True)
        for agent_id, result in zip(ids_with_coros, results):
            if isinstance(result, Exception):
                logger.error("cognition analysis failed for node %s: %s", agent_id, result)
                node_cognitions[agent_id] = NodeCognition(
                    agent_id=agent_id,
                    description=f"analysis could not be completed for {{agent:{agent_id}}}",
                )
                continue
            cog, log = result
            log.trace_id = trace_id
            node_cognitions[agent_id] = cog
            logs.append(log)

    for agent_id in agent_ids:
        if agent_id not in node_cognitions:
            node_cognitions[agent_id] = NodeCognition(
                agent_id=agent_id,
                description=f"{{agent:{agent_id}}} was not invoked in this trace",
            )

    narrative = ""
    narrative_result = await generate_trace_narrative(node_cognitions, graph, model)
    if isinstance(narrative_result, tuple):
        narrative, narrative_log = narrative_result
        narrative_log.trace_id = trace_id
        logs.append(narrative_log)

    trace_cognition = TraceCognition(
        trace_id=trace_id,
        graph_id=graph.graph_id,
        node_cognitions=node_cognitions,
        narrative=narrative,
        created_at=utc_timestamp(),
    )

    return trace_cognition, logs
