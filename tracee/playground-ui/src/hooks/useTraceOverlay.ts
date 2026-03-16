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
import {
  diffStateKeys,
  eventExplicitlyBelongsToNode,
  getEventTags,
  getParentRunIdFromEvent,
  getRunIdFromEvent,
  isRecord,
  isTopLevelNodeChainStart,
  parseTimestampMs,
  resolveAgentNodeId,
} from "../utils/traceEventUtils";

/**
 * groups trace events by their langgraph node and computes per-node execution data.
 * returns a map of node_id -> ExecutionData that can be merged onto graph nodes.
 */

// tracee: prefixed tags map to operation types, checked before heuristic
const TRACEE_TAG_PREFIX = "tracee:";
const TAG_TO_OP_TYPE: Record<string, AgentOperationType> = {
  "tracee:rag": "rag_retrieve",
  "tracee:code_exec": "code_exec",
  "tracee:tool": "tool_call",
};

// maps tool tags and name patterns to AgentOperationType for display
function classifyToolOperation(toolName: string, tags: string[]): AgentOperationType {
  for (const tag of tags) {
    if (tag.startsWith(TRACEE_TAG_PREFIX) && TAG_TO_OP_TYPE[tag]) {
      return TAG_TO_OP_TYPE[tag];
    }
  }
  // fallback heuristic with tighter patterns to avoid false positives
  const normalized = toolName.toLowerCase();
  if (/(retrieve|rag|vector_search|embed)/.test(normalized)) return "rag_retrieve";
  if (/(execute|run_code|python_repl|bash|shell)/.test(normalized)) return "code_exec";
  return "tool_call";
}

function isGraphStepTagged(event: TraceEvent): boolean {
  return getEventTags(event).some((tag) => tag.startsWith("graph:step:"));
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

function getRunMeta(event: TraceEvent): { runId?: string; parentRunId?: string } {
  return {
    runId: getRunIdFromEvent(event),
    parentRunId: getParentRunIdFromEvent(event),
  };
}

export function computeOverlay(events: TraceEvent[], nodeIds: string[]): Map<string, ExecutionData> {
  const nodeIdSet = new Set(nodeIds);
  const byNode = new Map<string, TraceEvent[]>();
  const runToNode = new Map<string, string>();
  const spanToNode = new Map<string, string>();
  const runParent = new Map<string, string>();
  // build run -> parent run map for scoping events to a node's run tree
  for (const event of events) {
    const runId = getRunIdFromEvent(event);
    const parentRunId = getParentRunIdFromEvent(event);
    if (runId && parentRunId) runParent.set(runId, parentRunId);
  }

  // populate runToNode and spanToNode so later events can resolve via run/span
  for (const event of events) {
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
    if (!nodeId) continue;
    const runId = getRunIdFromEvent(event);
    if (runId) runToNode.set(runId, nodeId);
    if (event.span_id) spanToNode.set(event.span_id, nodeId);
  }

  // group events by resolved agent node
  for (const event of events) {
    const node = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
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

    const hasGraphTaggedAttempts = orderedEvents.some((event) =>
      isTopLevelNodeChainStart(event, nodeId, runToNode, spanToNode) && isGraphStepTagged(event)
    );

    const isNodeAttemptStart = (event: TraceEvent): boolean => {
      if (!isTopLevelNodeChainStart(event, nodeId, runToNode, spanToNode)) return false;
      if (!hasGraphTaggedAttempts) return true;
      return isGraphStepTagged(event);
    };

    // top-level chain runs that explicitly started on this node (used to scope events)
    const rootRunIds = new Set(
      orderedEvents
        .filter((event) => event.event_type === "on_chain_start" && eventExplicitlyBelongsToNode(event, nodeId))
        .map((event) => getRunIdFromEvent(event))
        .filter((runId): runId is string => Boolean(runId)),
    );

    // include event if it belongs to this node or is under one of its root runs
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

    // latency from first to last scoped event
    const timestamps = scopedEvents.map((e) => parseTimestampMs(e.timestamp)).sort((a, b) => a - b);
    const latencyMs = timestamps.length >= 2
      ? timestamps[timestamps.length - 1] - timestamps[0]
      : undefined;

    // match end events to their start for latency and payload
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
        if (!runId || !isNodeAttemptStart(event)) continue;
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
        const allToolTags = [...startTags, ...endTags];
        operations.push({
          id: event.event_id,
          type: classifyToolOperation(toolName, allToolTags),
          label: toolName,
          status: "success",
          latencyMs: opLatency,
          input: normalizePayloadValue(startInput),
          output: normalizePayloadValue(endOutput),
          metadata: {
            tags: allToolTags,
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

    // count top-level chain starts on this node (retries)
    const nodeAttemptCount = scopedEvents.filter(isNodeAttemptStart).length;
    const llmStart = scopedEvents.find((event) => event.event_type === "on_llm_start");
    const llmEnd = [...scopedEvents].reverse().find((event) => event.event_type === "on_llm_end");
    const llmInput = llmStart?.payload?.input
      ? JSON.stringify(llmStart.payload.input, null, 2).slice(0, 2000)
      : llmStart?.payload?.prompts
      ? JSON.stringify(llmStart.payload.prompts, null, 2).slice(0, 2000)
      : undefined;
    const llmOutputValue = llmEnd
      ? normalizePayloadValue(llmEnd.payload?.output_text ?? llmEnd.payload?.output)
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
      llmOutputValue,
      operations,
      events: scopedEvents,
    });
  }

  return overlay;
}

/** fetches trace events and merges execution data onto agent nodes; clears when traceId is null */
export function useTraceOverlay(
  traceId: string | null,
  baseNodes: Node<GraphNodeData>[],
): Node<GraphNodeData>[] {
  const [overlaidNodes, setOverlaidNodes] = useState<Node<GraphNodeData>[]>(baseNodes);

  useEffect(() => {
    if (!traceId) {
      // clear execution when no trace selected
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
