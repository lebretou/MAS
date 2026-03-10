import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import type { GraphEdgeData, GraphNodeData } from "../../../types/node-data";
import type { TraceSummary } from "../../../types/trace";
import { buildTraceMinimapModel } from "./TraceMinimapPreview";

function makeNode(
  id: string,
  type: "agent" | "terminal",
  x: number,
  y: number,
  nodeType: "agent" | "start" | "end" = "agent",
): Node<GraphNodeData> {
  return {
    id,
    type,
    position: { x, y },
    data: {
      label: id,
      nodeType,
    },
  };
}

function makeEdge(source: string, target: string): Edge<GraphEdgeData> {
  return {
    id: `${source}-${target}`,
    source,
    target,
    data: {
      conditional: false,
    },
  };
}

describe("buildTraceMinimapModel", () => {
  it("builds a scaled model that preserves layout bounds and highlights active paths", () => {
    const nodes: Node<GraphNodeData>[] = [
      makeNode("planner", "agent", 0, 0),
      makeNode("coder", "agent", 300, 40),
      makeNode("finish", "terminal", 620, 109, "end"),
    ];
    const edges: Edge<GraphEdgeData>[] = [
      makeEdge("planner", "coder"),
      makeEdge("coder", "finish"),
    ];
    const summary: TraceSummary = {
      execution_id: "exec-1",
      trace_id: "trace-1",
      agents: ["planner", "coder"],
      edges: [
        { from_agent: "planner", to_agent: "coder", message_count: 1 },
      ],
      messages_by_edge: {},
      failures: [],
      tool_usage: [],
      llm_usage: [],
      event_count: 42,
    };

    const model = buildTraceMinimapModel(nodes, edges, summary);

    expect(model?.viewBox).toBe("30 33 710 144");
    expect(model?.nodes).toEqual([
      expect.objectContaining({ id: "planner", x: 54, y: 57, isActive: true }),
      expect.objectContaining({ id: "coder", x: 354, y: 97, isActive: true }),
      expect.objectContaining({ id: "finish", x: 604, y: 97, isActive: false }),
    ]);
    expect(model?.edges).toEqual([
      expect.objectContaining({
        id: "planner-coder",
        source: "planner",
        target: "coder",
        isActive: true,
        x1: 110,
        y1: 85,
        x2: 410,
        y2: 125,
      }),
      expect.objectContaining({
        id: "coder-finish",
        source: "coder",
        target: "finish",
        isActive: false,
        x1: 410,
        y1: 125,
        x2: 660,
        y2: 125,
      }),
    ]);
  });

  it("renders an inactive preview when no summary is available", () => {
    const nodes: Node<GraphNodeData>[] = [makeNode("planner", "agent", 0, 0)];
    const edges: Edge<GraphEdgeData>[] = [];

    const model = buildTraceMinimapModel(nodes, edges, null);

    expect(model?.nodes).toEqual([
      expect.objectContaining({ id: "planner", isActive: false }),
    ]);
    expect(model?.edges).toEqual([]);
  });

  it("keeps rendering when a trace summary includes extra runtime-only agents", () => {
    const nodes: Node<GraphNodeData>[] = [makeNode("planner", "agent", 0, 0)];
    const edges: Edge<GraphEdgeData>[] = [];
    const summary: TraceSummary = {
      execution_id: "exec-2",
      trace_id: "trace-2",
      agents: ["planner", "reviewer"],
      edges: [],
      messages_by_edge: {},
      failures: [],
      tool_usage: [],
      llm_usage: [],
      event_count: 12,
    };

    const model = buildTraceMinimapModel(nodes, edges, summary);

    expect(model?.nodes).toEqual([
      expect.objectContaining({ id: "planner", isActive: true }),
    ]);
  });
});
