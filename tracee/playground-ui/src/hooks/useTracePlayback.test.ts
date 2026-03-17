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

  it("does not create extra frames for internal chain calls inside a node", () => {
    // interaction has one graph-level invocation plus one internal helper chain —
    // the helper must not produce a second frame
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "ic1",
        event_type: "on_chain_start",
        sequence: 1,
        timestamp: "2026-03-01T00:00:00.000Z",
        refs: {
          langgraph: { node: "interaction" },
          langchain: { run_id: "run-interaction-1", parent_run_id: "run-graph" },
        },
        payload: {
          tags: ["graph:step:1"],
          inputs: { messages: [], next_agent: null },
        },
      }),
      // internal helper chain (no graph:step tag)
      makeEvent({
        event_id: "ic2",
        event_type: "on_chain_start",
        sequence: 2,
        timestamp: "2026-03-01T00:00:01.000Z",
        refs: {
          langgraph: { node: "interaction" },
          langchain: { run_id: "run-decision-llm", parent_run_id: "run-graph" },
        },
        payload: {
          inputs: { messages: [{ role: "user", content: "hi" }] },
        },
      }),
      makeEvent({
        event_id: "ic3",
        event_type: "on_chain_end",
        sequence: 3,
        timestamp: "2026-03-01T00:00:02.000Z",
        refs: {
          langgraph: { node: "interaction" },
          langchain: { run_id: "run-decision-llm" },
        },
        payload: {
          outputs: { messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }] },
        },
      }),
      makeEvent({
        event_id: "ic4",
        event_type: "on_chain_end",
        sequence: 4,
        timestamp: "2026-03-01T00:00:03.000Z",
        refs: {
          langgraph: { node: "interaction" },
          langchain: { run_id: "run-interaction-1" },
        },
        payload: {
          outputs: { messages: ["done"], next_agent: "end" },
        },
      }),
    ];

    const frames = computeExecutionFrames(events, { interaction: "Interaction" });

    // should be: frame 0 (initial state) + frame 1 (interaction) = 2 frames, not 3
    expect(frames).toHaveLength(2);
    expect(frames[1]).toMatchObject({ nodeId: "interaction" });
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
        eventOrder: 0,
        endSequence: 0,
        latencyMs: 0,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 1,
        nodeId: "router",
        label: "Router",
        timestamp: "2026-03-01T00:00:01.000Z",
        eventId: "router-end",
        eventOrder: 1,
        endSequence: 1,
        latencyMs: 1000,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 2,
        nodeId: "planner",
        label: "Planner",
        timestamp: "2026-03-01T00:00:02.000Z",
        eventId: "planner-end",
        eventOrder: 2,
        endSequence: 2,
        latencyMs: 1000,
        changedKeys: [],
        stateSnapshot: {},
      },
      {
        index: 3,
        nodeId: "reporter",
        label: "Reporter",
        timestamp: "2026-03-01T00:00:03.000Z",
        eventId: "reporter-end",
        eventOrder: 3,
        endSequence: 3,
        latencyMs: 1000,
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
