import { describe, expect, it } from "vitest";
import type { TraceEvent } from "../types/trace";
import { computeExecutionFrames, getNodeFrameState } from "./useTracePlayback";

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

describe("computeExecutionFrames", () => {
  it("builds initial and per-agent frames from top-level chain events", () => {
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "f1",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          inputs: {
            messages: [],
            next_agent: null,
          },
        },
      }),
      makeEvent({
        event_id: "f2",
        event_type: "on_tool_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:01.000Z",
        refs: {
          hint: { agent_id: "router" },
          langchain: { run_id: "run-router-tool", parent_run_id: "run-router" },
        },
        payload: { tool_name: "search_docs" },
      }),
      makeEvent({
        event_id: "f3",
        event_type: "on_chain_end",
        sequence: 3,
        timestamp: "2026-03-01T00:00:02.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          outputs: {
            messages: ["routed"],
            next_agent: "planner",
          },
        },
      }),
      makeEvent({
        event_id: "f4",
        event_type: "on_chain_start",
        sequence: 4,
        timestamp: "2026-03-01T00:00:03.000Z",
        refs: {
          langgraph: { node: "planner" },
          langchain: { run_id: "run-planner" },
        },
        payload: {
          inputs: {
            messages: ["routed"],
            next_agent: "planner",
          },
        },
      }),
      makeEvent({
        event_id: "f5",
        event_type: "on_chain_end",
        sequence: 5,
        timestamp: "2026-03-01T00:00:04.000Z",
        refs: {
          langgraph: { node: "planner" },
          langchain: { run_id: "run-planner-nested", parent_run_id: "run-planner" },
        },
        payload: {
          outputs: {
            messages: ["nested"],
            next_agent: "nested",
          },
        },
      }),
      makeEvent({
        event_id: "f6",
        event_type: "on_chain_end",
        sequence: 6,
        timestamp: "2026-03-01T00:00:05.000Z",
        refs: {
          langgraph: { node: "planner" },
          langchain: { run_id: "run-planner" },
        },
        payload: {
          outputs: {
            messages: ["routed"],
            next_agent: "executor",
            analysis_plan: { steps: 3 },
          },
        },
      }),
    ];

    const frames = computeExecutionFrames(events, {
      router: "Router",
      planner: "Planner",
    });

    expect(frames).toHaveLength(3);
    expect(frames[0]).toMatchObject({
      index: 0,
      nodeId: null,
      label: "Initial state",
      changedKeys: [],
      stateSnapshot: {
        messages: [],
        next_agent: null,
      },
    });
    expect(frames[1]).toMatchObject({
      index: 1,
      nodeId: "router",
      label: "Router",
      changedKeys: ["messages", "next_agent"],
      stateSnapshot: {
        messages: ["routed"],
        next_agent: "planner",
      },
    });
    expect(frames[2]).toMatchObject({
      index: 2,
      nodeId: "planner",
      label: "Planner",
      changedKeys: ["analysis_plan", "next_agent"],
      stateSnapshot: {
        messages: ["routed"],
        next_agent: "executor",
        analysis_plan: { steps: 3 },
      },
    });
  });

  it("uses the earliest top-level agent start as the initial frame", () => {
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "n1",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router-nested", parent_run_id: "run-router" },
        },
        payload: {
          inputs: {
            nested: true,
          },
        },
      }),
      makeEvent({
        event_id: "n2",
        event_type: "on_chain_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:01.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          inputs: {
            messages: [],
            next_agent: null,
          },
        },
      }),
      makeEvent({
        event_id: "n3",
        event_type: "on_chain_end",
        sequence: 3,
        timestamp: "2026-03-01T00:00:02.000Z",
        refs: {
          langgraph: { node: "router" },
          langchain: { run_id: "run-router" },
        },
        payload: {
          outputs: {
            messages: ["done"],
            next_agent: "planner",
          },
        },
      }),
    ];

    const frames = computeExecutionFrames(events, {
      router: "Router",
    });

    expect(frames[0]).toMatchObject({
      label: "Initial state",
      stateSnapshot: {
        messages: [],
        next_agent: null,
      },
    });
  });
});

describe("getNodeFrameState", () => {
  it("classifies nodes as active, completed, upcoming, or idle", () => {
    const frames = [
      {
        index: 0,
        nodeId: null,
        label: "Initial state",
        timestamp: "2026-03-01T00:00:00.000Z",
        eventId: "initial",
        endSequence: 0,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 1,
        nodeId: "router",
        label: "Router",
        timestamp: "2026-03-01T00:00:01.000Z",
        eventId: "router-end",
        endSequence: 1,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 2,
        nodeId: "planner",
        label: "Planner",
        timestamp: "2026-03-01T00:00:02.000Z",
        eventId: "planner-end",
        endSequence: 2,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 3,
        nodeId: "reporter",
        label: "Reporter",
        timestamp: "2026-03-01T00:00:03.000Z",
        eventId: "reporter-end",
        endSequence: 3,
        changedKeys: [],
        stateSnapshot: {},
      },
    ];

    expect(getNodeFrameState("planner", frames, 2)).toBe("active");
    expect(getNodeFrameState("router", frames, 2)).toBe("completed");
    expect(getNodeFrameState("reporter", frames, 2)).toBe("upcoming");
    expect(getNodeFrameState("never-run", frames, 2)).toBe("idle");
  });
});
