import { describe, expect, it } from "vitest";
import type { TraceEvent } from "../types/trace";
import { buildTraceOutline } from "./traceOutline";

function makeEvent(overrides: Partial<TraceEvent>): TraceEvent {
  return {
    event_id: overrides.event_id ?? crypto.randomUUID(),
    trace_id: overrides.trace_id ?? "trace-1",
    execution_id: overrides.execution_id ?? "exec-1",
    timestamp: overrides.timestamp ?? new Date().toISOString(),
    sequence: overrides.sequence ?? 0,
    event_type: overrides.event_type ?? "on_chain_start",
    refs: overrides.refs ?? {},
    payload: overrides.payload ?? {},
    agent_id: overrides.agent_id ?? null,
    span_id: overrides.span_id ?? null,
    parent_span_id: overrides.parent_span_id ?? null,
  };
}

describe("buildTraceOutline", () => {
  it("builds nested outline rows and focusable operation ids", () => {
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "root-start",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          tags: ["graph:step:1"],
          inputs: { messages: [], next_agent: null },
        },
      }),
      makeEvent({
        event_id: "llm-start",
        event_type: "on_llm_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:00.500Z",
        span_id: "span-llm",
        refs: {
          langchain: { run_id: "run-llm", parent_run_id: "run-router" },
        },
        payload: { model_name: "gpt-4.1" },
      }),
      makeEvent({
        event_id: "llm-end",
        event_type: "on_llm_end",
        sequence: 3,
        timestamp: "2026-03-01T00:00:01.500Z",
        span_id: "span-llm",
        refs: {
          langchain: { run_id: "run-llm", parent_run_id: "run-router" },
        },
        payload: { output_text: "ok" },
      }),
      makeEvent({
        event_id: "tool-start",
        event_type: "on_tool_start",
        sequence: 4,
        timestamp: "2026-03-01T00:00:01.700Z",
        refs: {
          hint: { agent_id: "router" },
          langchain: { run_id: "run-tool", parent_run_id: "run-router" },
        },
        payload: { tool_name: "search_docs", tags: ["tracee:rag"] },
      }),
      makeEvent({
        event_id: "tool-end",
        event_type: "on_tool_end",
        sequence: 5,
        timestamp: "2026-03-01T00:00:02.200Z",
        refs: {
          langchain: { run_id: "run-tool", parent_run_id: "run-router" },
        },
        payload: { output: "docs" },
      }),
      makeEvent({
        event_id: "root-end",
        event_type: "on_chain_end",
        sequence: 6,
        timestamp: "2026-03-01T00:00:03.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          outputs: { messages: ["done"], next_agent: "planner" },
        },
      }),
    ];

    const outline = buildTraceOutline(events, { router: "Router" });

    expect(outline).toHaveLength(1);
    expect(outline[0]).toMatchObject({
      runId: "run-router",
      label: "Router",
      kind: "agent",
      nodeId: "router",
    });
    expect(outline[0].children).toEqual([
      expect.objectContaining({
        runId: "run-llm",
        kind: "llm_call",
        label: "gpt-4.1",
        nodeId: "router",
        operationId: "llm-end",
      }),
      expect.objectContaining({
        runId: "run-tool",
        kind: "rag_retrieve",
        label: "search_docs",
        nodeId: "router",
        operationId: "tool-end",
      }),
      expect.objectContaining({
        kind: "state_update",
        label: "state update",
        operationId: "root-end",
      }),
    ]);
  });

  it("adds an error leaf for failed nested runs", () => {
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "agent-start",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "planner" },
          langchain: { run_id: "run-planner" },
        },
      }),
      makeEvent({
        event_id: "tool-start",
        event_type: "on_tool_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:00.100Z",
        refs: {
          hint: { agent_id: "planner" },
          langchain: { run_id: "run-tool", parent_run_id: "run-planner" },
        },
        payload: { tool_name: "python_repl" },
      }),
      makeEvent({
        event_id: "tool-error",
        event_type: "on_tool_error",
        sequence: 3,
        timestamp: "2026-03-01T00:00:00.400Z",
        refs: {
          langchain: { run_id: "run-tool", parent_run_id: "run-planner" },
        },
        payload: { error_type: "RuntimeError" },
      }),
    ];

    const outline = buildTraceOutline(events, { planner: "Planner" });
    const toolRun = outline[0].children[0];

    expect(toolRun).toMatchObject({
      runId: "run-tool",
      kind: "code_exec",
      label: "python_repl",
      status: "error",
    });
    expect(toolRun.children).toEqual([
      expect.objectContaining({
        kind: "error",
        label: "RuntimeError",
        operationId: "tool-error",
      }),
    ]);
  });

  it("prefers graph-tagged top-level runs when present", () => {
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "untagged-start",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router-untagged" },
        },
      }),
      makeEvent({
        event_id: "tagged-start",
        event_type: "on_chain_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:01.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router-tagged" },
        },
        payload: {
          tags: ["graph:step:1"],
        },
      }),
    ];

    const outline = buildTraceOutline(events, { router: "Router" });

    expect(outline).toHaveLength(1);
    expect(outline[0].runId).toBe("run-router-tagged");
  });
});
