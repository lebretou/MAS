"""Analysis utilities for trace events."""

from backbone.analysis.trace_summary import (
    AgentEdge,
    ToolUsage,
    TraceSummary,
    trace_summary,
)
from backbone.analysis.analyze_trace import (
    load_trace_events,
    format_summary,
    summary_to_dict,
)
from backbone.analysis.agent_analyzer import (
    AgentAnalysis,
    AgentDecision,
    AgentMessage,
    TraceAnalysis,
    analyze_agent,
    analyze_trace,
    group_events_by_agent,
    infer_agent_messages,
)

__all__ = [
    # trace_summary exports
    "AgentEdge",
    "ToolUsage",
    "TraceSummary",
    "trace_summary",
    # analyze_trace exports
    "load_trace_events",
    "format_summary",
    "summary_to_dict",
    # agent_analyzer exports (LLM-powered, placeholder)
    "AgentAnalysis",
    "AgentDecision",
    "AgentMessage",
    "TraceAnalysis",
    "analyze_agent",
    "analyze_trace",
    "group_events_by_agent",
    "infer_agent_messages",
]
