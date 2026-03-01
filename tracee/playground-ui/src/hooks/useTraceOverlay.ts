import { useEffect, useState } from "react";
import type { Node } from "@xyflow/react";
import { fetchTraceEvents } from "../api/traces";
import type {
  AgentOperation,
  AgentOperationType,
  ExecutionData,
  GraphNodeData,
} from "../types/node-data";
import type { TraceEvent } from "../types/trace";

/**
 * groups trace events by their langgraph node and computes per-node execution data.
 * returns a map of node_id -> ExecutionData that can be merged onto graph nodes.
 */
function getNodeIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langgraph as Record<string, unknown> | undefined)?.node as string | undefined;
}

function getHintAgentIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.hint as Record<string, unknown> | undefined)?.agent_id as string | undefined;
}

function getRunIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langchain as Record<string, unknown> | undefined)?.run_id as string | undefined;
}

function getParentRunIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langchain as Record<string, unknown> | undefined)?.parent_run_id as string | undefined;
}

function parseTimestampMs(timestamp: string): number {
  return new Date(timestamp).getTime();
}

function classifyToolOperation(toolName: string): AgentOperationType {
  const normalized = toolName.toLowerCase();
  if (/(retrieve|rag|search|vector|embed)/.test(normalized)) return "rag_retrieve"; // can this be improved?
  if (/(execute|run_code|python|bash|shell)/.test(normalized)) return "code_exec";
  return "tool_call";
}

function buildOperationLabel(value: unknown, fallback: string): string {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

function eventExplicitlyBelongsToNode(event: TraceEvent, nodeId: string): boolean {
  const hintAgentId = getHintAgentIdFromEvent(event);
  if (hintAgentId === nodeId) return true;
  if (event.agent_id === nodeId) return true;
  return getNodeIdFromEvent(event) === nodeId;
}

export function computeOverlay(events: TraceEvent[], nodeIds: string[]): Map<string, ExecutionData> {
  const nodeIdSet = new Set(nodeIds);
  const byNode = new Map<string, TraceEvent[]>();
  const runToNode = new Map<string, string>();
  const spanToNode = new Map<string, string>();

  function resolveAgentNodeId(event: TraceEvent): string | undefined {
    const hintAgentId = getHintAgentIdFromEvent(event);
    if (hintAgentId && nodeIdSet.has(hintAgentId)) return hintAgentId;

    if (event.agent_id && nodeIdSet.has(event.agent_id)) return event.agent_id;

    const directNodeId = getNodeIdFromEvent(event);
    if (directNodeId && nodeIdSet.has(directNodeId)) return directNodeId;

    const runId = getRunIdFromEvent(event);
    if (runId && runToNode.has(runId)) return runToNode.get(runId);

    const parentRunId = getParentRunIdFromEvent(event);
    if (parentRunId && runToNode.has(parentRunId)) return runToNode.get(parentRunId);

    if (event.parent_span_id && spanToNode.has(event.parent_span_id)) return spanToNode.get(event.parent_span_id);
    if (event.span_id && spanToNode.has(event.span_id)) return spanToNode.get(event.span_id);

    return undefined;
  }

  for (const event of events) {
    const nodeId = resolveAgentNodeId(event);
    if (!nodeId) continue;
    const runId = getRunIdFromEvent(event);
    if (runId) runToNode.set(runId, nodeId);
    if (event.span_id) spanToNode.set(event.span_id, nodeId);
  }

  for (const event of events) {
    const node = resolveAgentNodeId(event);
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

    const hasError = nodeEvents.some((e) => e.event_type.endsWith("_error"));

    // compute latency from first to last event
    const timestamps = nodeEvents.map((e) => parseTimestampMs(e.timestamp)).sort((a, b) => a - b);
    const latencyMs = timestamps.length >= 2
      ? timestamps[timestamps.length - 1] - timestamps[0]
      : undefined;

    const orderedEvents = [...nodeEvents].sort((a, b) => {
      const sequenceA = a.sequence ?? Number.MAX_SAFE_INTEGER;
      const sequenceB = b.sequence ?? Number.MAX_SAFE_INTEGER;
      if (sequenceA !== sequenceB) return sequenceA - sequenceB;
      return parseTimestampMs(a.timestamp) - parseTimestampMs(b.timestamp);
    });

    const llmStartsBySpan = new Map<string, TraceEvent>();
    const toolStartsBySpan = new Map<string, TraceEvent>();
    const operations: AgentOperation[] = [];

    let promptTokensTotal = 0;
    let completionTokensTotal = 0;
    let hasAnyTokenUsage = false;

    for (const event of orderedEvents) {
      if (event.event_type === "on_llm_start" && event.span_id) {
        llmStartsBySpan.set(event.span_id, event);
      }
      if (event.event_type === "on_tool_start" && event.span_id) {
        toolStartsBySpan.set(event.span_id, event);
      }
      if (event.event_type === "on_llm_end") {
        const tokenUsage = event.payload?.token_usage as Record<string, unknown> | undefined;
        const promptTokens = typeof tokenUsage?.prompt_tokens === "number" ? tokenUsage.prompt_tokens : 0;
        const completionTokens = typeof tokenUsage?.completion_tokens === "number" ? tokenUsage.completion_tokens : 0;
        promptTokensTotal += promptTokens;
        completionTokensTotal += completionTokens;
        if (tokenUsage) hasAnyTokenUsage = true;

        const start = event.span_id ? llmStartsBySpan.get(event.span_id) : undefined;
        const modelName = buildOperationLabel(start?.payload?.model_name ?? event.payload?.model_name, "llm");
        const opLatency = start
          ? parseTimestampMs(event.timestamp) - parseTimestampMs(start.timestamp)
          : undefined;

        operations.push({
          id: event.event_id,
          type: "llm_call",
          label: modelName,
          status: "success",
          latencyMs: opLatency,
          tokenCount: promptTokens + completionTokens,
        });
      }

      if (event.event_type === "on_tool_end") {
        const start = event.span_id ? toolStartsBySpan.get(event.span_id) : undefined;
        const toolName = buildOperationLabel(start?.payload?.tool_name, "tool");
        const opLatency = start
          ? parseTimestampMs(event.timestamp) - parseTimestampMs(start.timestamp)
          : undefined;
        operations.push({
          id: event.event_id,
          type: classifyToolOperation(toolName),
          label: toolName,
          status: "success",
          latencyMs: opLatency,
        });
      }

      if (event.event_type.endsWith("_error")) {
        const errorType = buildOperationLabel(event.payload?.error_type, event.event_type);
        operations.push({
          id: event.event_id,
          type: "error",
          label: errorType,
          status: "error",
        });
      }
    }

    const nodeAttemptCount = orderedEvents.filter((event) => {
      if (event.event_type !== "on_chain_start") return false;
      if (!eventExplicitlyBelongsToNode(event, nodeId)) return false;
      const parentRunId = getParentRunIdFromEvent(event);
      const parentRunNodeId = parentRunId ? runToNode.get(parentRunId) : undefined;
      if (parentRunNodeId === nodeId) return false;
      const parentSpanNodeId = event.parent_span_id ? spanToNode.get(event.parent_span_id) : undefined;
      if (parentSpanNodeId === nodeId) return false;
      return true;
    }).length;
    const llmStart = orderedEvents.find((event) => event.event_type === "on_llm_start");
    const llmEnd = [...orderedEvents].reverse().find((event) => event.event_type === "on_llm_end");
    const llmInput = llmStart?.payload?.input
      ? JSON.stringify(llmStart.payload.input, null, 2).slice(0, 2000)
      : llmStart?.payload?.prompts
      ? JSON.stringify(llmStart.payload.prompts, null, 2).slice(0, 2000)
      : undefined;
    const llmOutput = llmEnd?.payload?.output_text
      ? String(llmEnd.payload.output_text).slice(0, 2000)
      : llmEnd?.payload?.output
      ? JSON.stringify(llmEnd.payload.output, null, 2).slice(0, 2000)
      : undefined;

    overlay.set(nodeId, {
      invoked: true,
      status: hasError ? "error" : "success",
      latencyMs,
      promptTokens: hasAnyTokenUsage ? promptTokensTotal : undefined,
      completionTokens: hasAnyTokenUsage ? completionTokensTotal : undefined,
      retryCount: nodeAttemptCount,
      llmInput,
      llmOutput,
      operations,
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
