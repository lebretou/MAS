import { useEffect, useState } from "react";
import type { Node } from "@xyflow/react";
import { fetchTraceEvents } from "../api/traces";
import type { GraphNodeData, ExecutionData } from "../types/node-data";
import type { TraceEvent } from "../types/trace";

/**
 * groups trace events by their langgraph node and computes per-node execution data.
 * returns a map of node_id -> ExecutionData that can be merged onto graph nodes.
 */
function computeOverlay(events: TraceEvent[], nodeIds: string[]): Map<string, ExecutionData> {
  const nodeIdSet = new Set(nodeIds);
  const byNode = new Map<string, TraceEvent[]>();

  for (const event of events) {
    const node = (event.refs?.langgraph as Record<string, unknown> | undefined)?.node as string | undefined;
    if (!node || !nodeIdSet.has(node)) continue;
    const existing = byNode.get(node) ?? [];
    existing.push(event);
    byNode.set(node, existing);
  }

  const overlay = new Map<string, ExecutionData>();
  for (const nodeId of nodeIds) {
    const nodeEvents = byNode.get(nodeId);
    if (!nodeEvents || nodeEvents.length === 0) {
      overlay.set(nodeId, { invoked: false });
      continue;
    }

    const hasError = nodeEvents.some((e) =>
      e.event_type.endsWith("_error"),
    );

    // compute latency from first to last event
    const timestamps = nodeEvents.map((e) => new Date(e.timestamp).getTime()).sort((a, b) => a - b);
    const latencyMs = timestamps.length >= 2
      ? timestamps[timestamps.length - 1] - timestamps[0]
      : undefined;

    // extract LLM input/output from tool_call events
    const llmStart = nodeEvents.find(
      (e) => e.event_type === "tool_call" && e.payload?.phase === "start" && (e.payload?.tool_name as string)?.startsWith("llm"),
    );
    const llmEnd = nodeEvents.find(
      (e) => e.event_type === "tool_call" && e.payload?.phase === "end" && (e.payload?.tool_name as string)?.startsWith("llm"),
    );

    const llmInput = llmStart?.payload?.input
      ? JSON.stringify(llmStart.payload.input, null, 2).slice(0, 2000)
      : undefined;
    const llmOutput = llmEnd?.payload?.output
      ? JSON.stringify(llmEnd.payload.output, null, 2).slice(0, 2000)
      : undefined;

    overlay.set(nodeId, {
      invoked: true,
      status: hasError ? "error" : "success",
      latencyMs,
      llmInput,
      llmOutput,
      events: nodeEvents,
    });
  }

  return overlay;
}

export function useTraceOverlay(
  traceId: string | null,
  baseNodes: Node<GraphNodeData>[],
): Node<GraphNodeData>[] {
  const [overlaidNodes, setOverlaidNodes] = useState<Node<GraphNodeData>[]>(baseNodes);

  useEffect(() => {
    if (!traceId) {
      // clear execution data when no trace selected
      setOverlaidNodes(
        baseNodes.map((n) => ({
          ...n,
          data: { ...n.data, execution: undefined },
        })),
      );
      return;
    }

    let cancelled = false;
    fetchTraceEvents(traceId)
      .then((events) => {
        if (cancelled) return;
        const agentNodeIds = baseNodes
          .filter((n) => n.data.nodeType === "agent")
          .map((n) => n.id);
        const overlay = computeOverlay(events, agentNodeIds);

        setOverlaidNodes(
          baseNodes.map((n) => {
            const exec = overlay.get(n.id);
            return exec
              ? { ...n, data: { ...n.data, execution: exec } }
              : n;
          }),
        );
      })
      .catch(() => {
        if (!cancelled) setOverlaidNodes(baseNodes);
      });

    return () => { cancelled = true; };
  }, [traceId, baseNodes]);

  return overlaidNodes;
}
