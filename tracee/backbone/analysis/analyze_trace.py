#!/usr/bin/env python3
"""CLI script to analyze trace files.

Usage:
    uv run python backbone/analysis/analyze_trace.py <trace_file.jsonl>
    
    # or with JSON output
    uv run python backbone/analysis/analyze_trace.py <trace_file.jsonl> --json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# ensure backbone package is importable when running as script
_script_dir = Path(__file__).resolve().parent
_backbone_root = _script_dir.parent.parent
if str(_backbone_root) not in sys.path:
    sys.path.insert(0, str(_backbone_root))

from backbone.analysis.trace_summary import TraceSummary, trace_summary
from backbone.models.trace_event import TraceEvent


def load_trace_events(trace_file: Path) -> list[TraceEvent]:
    """Load trace events from a JSONL file.
    
    Args:
        trace_file: path to the JSONL trace file
        
    Returns:
        list of TraceEvent objects
    """
    events = []
    with open(trace_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            event = TraceEvent.model_validate(data)
            events.append(event)
    return events


def format_summary(summary: TraceSummary) -> str:
    """Format trace summary for human-readable output."""
    lines = []
    lines.append("=" * 60)
    lines.append("TRACE SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    
    # basic info
    lines.append(f"Execution ID: {summary.execution_id}")
    lines.append(f"Trace ID:     {summary.trace_id}")
    lines.append(f"Event Count:  {summary.event_count}")
    lines.append("")
    
    # agents
    lines.append("-" * 40)
    lines.append("AGENTS")
    lines.append("-" * 40)
    for agent in summary.agents:
        lines.append(f"  • {agent}")
    if not summary.agents:
        lines.append("  (no agents identified)")
    lines.append("")
    
    # agent transitions (inferred)
    if summary.edges:
        lines.append("-" * 40)
        lines.append("AGENT TRANSITIONS (inferred)")
        lines.append("-" * 40)
        for edge in summary.edges:
            lines.append(f"  {edge.from_agent} → {edge.to_agent} ({edge.message_count}x)")
        lines.append("")
    
    # LLM usage
    if summary.llm_usage:
        lines.append("-" * 40)
        lines.append("LLM USAGE")
        lines.append("-" * 40)
        for llm in summary.llm_usage:
            latency_str = f" (avg: {llm.avg_latency_ms:.2f}ms)" if llm.avg_latency_ms else ""
            lines.append(f"  • {llm.tool_name}: {llm.call_count} calls{latency_str}")
        lines.append("")
    
    # tool usage
    if summary.tool_usage:
        lines.append("-" * 40)
        lines.append("TOOL USAGE")
        lines.append("-" * 40)
        for tool in summary.tool_usage:
            latency_str = f" (avg: {tool.avg_latency_ms:.2f}ms)" if tool.avg_latency_ms else ""
            lines.append(f"  • {tool.tool_name}: {tool.call_count} calls{latency_str}")
        lines.append("")
    
    # failures
    if summary.failures:
        lines.append("-" * 40)
        lines.append("FAILURES")
        lines.append("-" * 40)
        for failure in summary.failures:
            lines.append(f"  [{failure['type']}] {failure.get('agent_id', 'unknown')}")
            lines.append(f"    Error: {failure.get('error_type', 'unknown')}")
            lines.append(f"    Message: {failure.get('message', '')}")
            lines.append(f"    Time: {failure.get('timestamp', '')}")
        lines.append("")
    
    if not summary.failures:
        lines.append("-" * 40)
        lines.append("✓ No failures detected")
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def summary_to_dict(summary: TraceSummary) -> dict:
    """Convert TraceSummary to a JSON-serializable dict."""
    d = asdict(summary)
    # convert tuple keys in messages_by_edge to string keys
    d["messages_by_edge"] = {
        f"{k[0]}->{k[1]}": v for k, v in summary.messages_by_edge.items()
    }
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a trace file and output summary statistics."
    )
    parser.add_argument(
        "trace_file",
        type=Path,
        help="path to the JSONL trace file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output summary as JSON instead of human-readable format",
    )
    
    args = parser.parse_args()
    
    if not args.trace_file.exists():
        print(f"Error: trace file not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)
    
    events = load_trace_events(args.trace_file)
    
    if not events:
        print("Error: no events found in trace file", file=sys.stderr)
        sys.exit(1)
    
    summary = trace_summary(events)
    
    if args.json:
        print(json.dumps(summary_to_dict(summary), indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
