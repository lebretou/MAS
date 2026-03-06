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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function diffStateKeys(previousState: Record<string, unknown>, nextState: Record<string, unknown>): string[] {
  return Array.from(new Set([...Object.keys(previousState), ...Object.keys(nextState)]))
    .filter((key) => JSON.stringify(previousState[key]) !== JSON.stringify(nextState[key]))
    .sort();
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

function parseEscapedString(value: string): string {
  return value
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\r")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'")
    .replace(/\\\\/g, "\\");
}

function parseMaybeJsonString(value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return trimmed;
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return trimmed;
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
}

function parseLangchainContentEnvelope(value: string): unknown {
  const marker = "content='";
  const start = value.indexOf(marker);
  if (start < 0) return parseMaybeJsonString(value);
  let i = start + marker.length;
  let content = "";
  while (i < value.length) {
    const ch = value[i];
    if (ch === "\\" && i + 1 < value.length) {
      content += ch + value[i + 1];
      i += 2;
      continue;
    }
    if (ch === "'") {
      const remainder = value.slice(i + 1).trimStart();
      if (remainder.length === 0 || /^,?\s*\w+=/.test(remainder)) break;
    }
    content += ch;
    i += 1;
  }
  if (!content) return parseMaybeJsonString(value);
  const decoded = parseEscapedString(content);
  return parseMaybeJsonString(decoded);
}

function normalizePayloadValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  if (value.includes("content='")) return parseLangchainContentEnvelope(value);
  return parseMaybeJsonString(value);
}

function getEventTags(event: TraceEvent): string[] {
  const rawTags = event.payload?.tags;
  if (!Array.isArray(rawTags)) return [];
  return rawTags.filter((tag): tag is string => typeof tag === "string");
}

function getRunMeta(event: TraceEvent): { runId?: string; parentRunId?: string } {
  return {
    runId: getRunIdFromEvent(event),
    parentRunId: getParentRunIdFromEvent(event),
  };
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
  const runParent = new Map<string, string>();
  for (const event of events) {
    const runId = getRunIdFromEvent(event);
    const parentRunId = getParentRunIdFromEvent(event);
    if (runId && parentRunId) runParent.set(runId, parentRunId);
  }

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

    const orderedEvents = [...nodeEvents].sort((a, b) => {
      const sequenceA = a.sequence ?? Number.MAX_SAFE_INTEGER;
      const sequenceB = b.sequence ?? Number.MAX_SAFE_INTEGER;
      if (sequenceA !== sequenceB) return sequenceA - sequenceB;
      return parseTimestampMs(a.timestamp) - parseTimestampMs(b.timestamp);
    });

    const rootRunIds = new Set(
      orderedEvents
        .filter((event) => event.event_type === "on_chain_start" && eventExplicitlyBelongsToNode(event, nodeId))
        .map((event) => getRunIdFromEvent(event))
        .filter((runId): runId is string => Boolean(runId)),
    );

    function belongsToNodeRunTree(event: TraceEvent): boolean {
      if (eventExplicitlyBelongsToNode(event, nodeId)) return true;
      const runId = getRunIdFromEvent(event);
      if (!runId || rootRunIds.size === 0) return false;
      let cursor: string | undefined = runId;
      while (cursor) {
        if (rootRunIds.has(cursor)) return true;
        cursor = runParent.get(cursor);
      }
      return false;
    }

    const scopedEvents = orderedEvents.filter(belongsToNodeRunTree);
    if (scopedEvents.length === 0) {
      overlay.set(nodeId, { invoked: false });
      continue;
    }

    const hasError = scopedEvents.some((e) => e.event_type.endsWith("_error"));

    // compute latency from first to last event
    const timestamps = scopedEvents.map((e) => parseTimestampMs(e.timestamp)).sort((a, b) => a - b);
    const latencyMs = timestamps.length >= 2
      ? timestamps[timestamps.length - 1] - timestamps[0]
      : undefined;

    const llmStartsBySpan = new Map<string, TraceEvent>();
    const toolStartsBySpan = new Map<string, TraceEvent>();
    const topLevelStartsByRunId = new Map<string, TraceEvent>();
    const operations: AgentOperation[] = [];

    let promptTokensTotal = 0;
    let completionTokensTotal = 0;
    let hasAnyTokenUsage = false;

    for (const event of scopedEvents) {
      if (event.event_type === "on_llm_start" && event.span_id) {
        llmStartsBySpan.set(event.span_id, event);
      }
      if (event.event_type === "on_tool_start" && event.span_id) {
        toolStartsBySpan.set(event.span_id, event);
      }
      if (event.event_type === "on_chain_start") {
        const runId = getRunIdFromEvent(event);
        if (!runId || !eventExplicitlyBelongsToNode(event, nodeId)) continue;
        const parentRunId = getParentRunIdFromEvent(event);
        const parentRunNodeId = parentRunId ? runToNode.get(parentRunId) : undefined;
        if (parentRunNodeId === nodeId) continue;
        const parentSpanNodeId = event.parent_span_id ? spanToNode.get(event.parent_span_id) : undefined;
        if (parentSpanNodeId === nodeId) continue;
        topLevelStartsByRunId.set(runId, event);
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
        const startTags = start ? getEventTags(start) : [];
        const endTags = getEventTags(event);
        const inputPayload = start?.payload?.input ?? start?.payload?.messages ?? start?.payload?.prompts;
        const outputPayload = event.payload?.output_text ?? event.payload?.output;

        operations.push({
          id: event.event_id,
          type: "llm_call",
          label: modelName,
          status: "success",
          latencyMs: opLatency,
          tokenCount: promptTokens + completionTokens,
          input: normalizePayloadValue(inputPayload),
          output: normalizePayloadValue(outputPayload),
          metadata: {
            modelName,
            finishReason: event.payload?.response_metadata
              ? (event.payload.response_metadata as Record<string, unknown>).finish_reason
              : undefined,
            tokenUsage,
            tags: [...startTags, ...endTags],
            ...getRunMeta(event),
          },
        });
      }

      if (event.event_type === "on_tool_end") {
        const start = event.span_id ? toolStartsBySpan.get(event.span_id) : undefined;
        const toolName = buildOperationLabel(start?.payload?.tool_name, "tool");
        const opLatency = start
          ? parseTimestampMs(event.timestamp) - parseTimestampMs(start.timestamp)
          : undefined;
        const startInput = start?.payload?.input;
        const endOutput = event.payload?.output;
        const startTags = start ? getEventTags(start) : [];
        const endTags = getEventTags(event);
        operations.push({
          id: event.event_id,
          type: classifyToolOperation(toolName),
          label: toolName,
          status: "success",
          latencyMs: opLatency,
          input: normalizePayloadValue(startInput),
          output: normalizePayloadValue(endOutput),
          metadata: {
            tags: [...startTags, ...endTags],
            ...getRunMeta(event),
          },
        });
      }

      if (event.event_type === "on_chain_end") {
        const runId = getRunIdFromEvent(event);
        const start = runId ? topLevelStartsByRunId.get(runId) : undefined;
        const previousState = start?.payload?.inputs;
        const nextState = event.payload?.outputs;
        if (!isRecord(previousState) || !isRecord(nextState)) continue;
        const changedKeys = diffStateKeys(previousState, nextState);
        if (changedKeys.length === 0) continue;

        operations.push({
          id: event.event_id,
          type: "state_update",
          label: "state update",
          status: "success",
          input: previousState,
          output: nextState,
          metadata: {
            changedKeys,
            ...getRunMeta(event),
          },
        });
      }

      if (event.event_type.endsWith("_error")) {
        const errorType = buildOperationLabel(event.payload?.error_type, event.event_type);
        const errorMessage = typeof event.payload?.error_message === "string"
          ? event.payload.error_message
          : undefined;
        operations.push({
          id: event.event_id,
          type: "error",
          label: errorType,
          status: "error",
          errorMessage,
          metadata: {
            tags: getEventTags(event),
            ...getRunMeta(event),
          },
        });
      }
    }

    const nodeAttemptCount = scopedEvents.filter((event) => {
      if (event.event_type !== "on_chain_start") return false;
      if (!eventExplicitlyBelongsToNode(event, nodeId)) return false;
      const parentRunId = getParentRunIdFromEvent(event);
      const parentRunNodeId = parentRunId ? runToNode.get(parentRunId) : undefined;
      if (parentRunNodeId === nodeId) return false;
      const parentSpanNodeId = event.parent_span_id ? spanToNode.get(event.parent_span_id) : undefined;
      if (parentSpanNodeId === nodeId) return false;
      return true;
    }).length;
    const llmStart = scopedEvents.find((event) => event.event_type === "on_llm_start");
    const llmEnd = [...scopedEvents].reverse().find((event) => event.event_type === "on_llm_end");
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
      events: scopedEvents,
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
