import { describe, expect, it } from "vitest";
import type { TraceEvent } from "../types/trace";
import { computeOverlay } from "./useTraceOverlay";

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

describe("computeOverlay", () => {
  it("computes llm/tool operations, token counts, and retries", () => {
    const nodeId = "interaction";
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "e1",
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-attempt-1" },
        },
      }),
      makeEvent({
        event_id: "e2",
        event_type: "on_llm_start",
        timestamp: "2026-03-01T00:00:01.000Z",
        span_id: "span-llm-1",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-llm-1", parent_run_id: "run-attempt-1" },
        },
        payload: { model_name: "gpt-4.1", prompts: ["hello"] },
      }),
      makeEvent({
        event_id: "e3",
        event_type: "on_llm_end",
        timestamp: "2026-03-01T00:00:02.000Z",
        span_id: "span-llm-1",
        refs: { langchain: { run_id: "run-llm-1", parent_run_id: "run-attempt-1" } },
        payload: {
          output_text: "world",
          token_usage: { prompt_tokens: 12, completion_tokens: 7 },
        },
      }),
      makeEvent({
        event_id: "e4",
        event_type: "on_tool_start",
        timestamp: "2026-03-01T00:00:03.000Z",
        span_id: "span-tool-1",
        refs: {
          langgraph: { node: "tools" },
          hint: { agent_id: nodeId },
          langchain: { run_id: "run-tool-1", parent_run_id: "run-attempt-1" },
        },
        payload: { tool_name: "retrieve_analysis_context_tool", input: { query: "sales" } },
      }),
      makeEvent({
        event_id: "e5",
        event_type: "on_tool_end",
        timestamp: "2026-03-01T00:00:04.000Z",
        span_id: "span-tool-1",
        refs: { langchain: { run_id: "run-tool-1", parent_run_id: "run-attempt-1" } },
        payload: { output: "docs" },
      }),
      makeEvent({
        event_id: "e6",
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:05.000Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-attempt-2" },
        },
      }),
      makeEvent({
        event_id: "e7",
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:05.500Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-nested-1", parent_run_id: "run-attempt-2" },
        },
      }),
    ];

    const overlay = computeOverlay(events, [nodeId]);
    const exec = overlay.get(nodeId);

    expect(exec).toBeDefined();
    expect(exec?.invoked).toBe(true);
    expect(exec?.status).toBe("success");
    expect(exec?.latencyMs).toBe(5500);
    expect(exec?.promptTokens).toBe(12);
    expect(exec?.completionTokens).toBe(7);
    expect(exec?.retryCount).toBe(2);
    expect(exec?.operations).toEqual([
      expect.objectContaining({ type: "llm_call", label: "gpt-4.1", status: "success", tokenCount: 19 }),
      expect.objectContaining({ type: "rag_retrieve", label: "retrieve_analysis_context_tool", status: "success" }),
    ]);
  });

  it("attributes tool events from tools node to hint agent", () => {
    const nodeId = "interaction";
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "t1",
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-agent-1" },
        },
      }),
      makeEvent({
        event_id: "t2",
        event_type: "on_tool_start",
        timestamp: "2026-03-01T00:00:01.000Z",
        span_id: "span-tool-rag",
        refs: {
          langgraph: { node: "tools" },
          hint: { agent_id: nodeId },
          langchain: { run_id: "run-tool-rag-1", parent_run_id: "run-agent-1" },
        },
        payload: { tool_name: "search_docs", input: { query: "rag" } },
      }),
      makeEvent({
        event_id: "t3",
        event_type: "on_tool_end",
        timestamp: "2026-03-01T00:00:01.800Z",
        span_id: "span-tool-rag",
        refs: {
          langchain: { run_id: "run-tool-rag-1", parent_run_id: "run-agent-1" },
        },
        payload: { output: "done" },
      }),
    ];

    const overlay = computeOverlay(events, [nodeId]);
    const exec = overlay.get(nodeId);

    expect(exec?.operations).toEqual([
      expect.objectContaining({
        type: "rag_retrieve",
        label: "search_docs",
        status: "success",
        latencyMs: 800,
      }),
    ]);
  });

  it("marks node as error when any *_error event exists", () => {
    const nodeId = "coding";
    const events: TraceEvent[] = [
      makeEvent({
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-coding" },
        },
      }),
      makeEvent({
        event_type: "on_tool_error",
        timestamp: "2026-03-01T00:00:01.000Z",
        refs: { langchain: { run_id: "run-tool-error", parent_run_id: "run-coding" } },
        payload: { error_type: "RuntimeError" },
      }),
    ];

    const overlay = computeOverlay(events, [nodeId]);
    const exec = overlay.get(nodeId);

    expect(exec?.status).toBe("error");
    expect(exec?.operations).toEqual([
      expect.objectContaining({ type: "error", label: "RuntimeError", status: "error" }),
    ]);
  });

  it("keeps explicit zero token usage values", () => {
    const nodeId = "summary";
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "z1",
        event_type: "on_chain_start",
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-summary-1" },
        },
      }),
      makeEvent({
        event_id: "z2",
        event_type: "on_llm_start",
        timestamp: "2026-03-01T00:00:01.000Z",
        span_id: "span-summary-llm-1",
        refs: {
          langgraph: { node: nodeId },
          langchain: { run_id: "run-summary-llm-1", parent_run_id: "run-summary-1" },
        },
        payload: { model_name: "gpt-4.1-mini", prompts: ["summarize"] },
      }),
      makeEvent({
        event_id: "z3",
        event_type: "on_llm_end",
        timestamp: "2026-03-01T00:00:02.000Z",
        span_id: "span-summary-llm-1",
        refs: { langchain: { run_id: "run-summary-llm-1", parent_run_id: "run-summary-1" } },
        payload: {
          output_text: "ok",
          token_usage: { prompt_tokens: 0, completion_tokens: 0 },
        },
      }),
    ];

    const overlay = computeOverlay(events, [nodeId]);
    const exec = overlay.get(nodeId);

    expect(exec?.promptTokens).toBe(0);
    expect(exec?.completionTokens).toBe(0);
  });
});
