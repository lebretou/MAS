import type { AgentOperationType, TraceOutlineItem, TraceOutlineItemKind } from "../types/node-data";
import type { TraceEvent } from "../types/trace";
import {
  diffStateKeys,
  eventExplicitlyBelongsToNode,
  getEventTags,
  getParentRunIdFromEvent,
  getRunIdFromEvent,
  isTopLevelNodeChainStart,
  isRecord,
  parseTimestampMs,
  resolveAgentNodeId,
  sortTraceEvents,
} from "./traceEventUtils";

interface MutableTraceRun {
  runId: string;
  parentRunId: string | null;
  nodeId: string | null;
  kind: TraceOutlineItemKind;
  label: string;
  status: "success" | "error";
  operationId?: string;
  startEvent?: TraceEvent;
  endEvent?: TraceEvent;
  errorEvent?: TraceEvent;
  children: MutableTraceRun[];
}

const TRACEE_TAG_PREFIX = "tracee:";
const TAG_TO_OP_TYPE: Record<string, AgentOperationType> = {
  "tracee:rag": "rag_retrieve",
  "tracee:code_exec": "code_exec",
  "tracee:tool": "tool_call",
};

function classifyToolOperation(toolName: string, tags: string[]): AgentOperationType {
  for (const tag of tags) {
    if (tag.startsWith(TRACEE_TAG_PREFIX) && TAG_TO_OP_TYPE[tag]) {
      return TAG_TO_OP_TYPE[tag];
    }
  }
  const normalized = toolName.toLowerCase();
  if (/(retrieve|rag|vector_search|embed)/.test(normalized)) return "rag_retrieve";
  if (/(execute|run_code|python_repl|bash|shell)/.test(normalized)) return "code_exec";
  return "tool_call";
}

function buildLabel(value: unknown, fallback: string): string {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : fallback;
}

function inferChainLabel(event: TraceEvent, nodeId: string | null, nodeLabels: Record<string, string>): string {
  const payloadName = buildLabel(
    event.payload?.name
      ?? (isRecord(event.payload?.serialized) ? event.payload.serialized.name : undefined)
      ?? event.payload?.run_name,
    "",
  );
  if (payloadName) return payloadName;
  if (nodeId && nodeLabels[nodeId]) return nodeLabels[nodeId];
  return "chain";
}

function inferKind(event: TraceEvent, nodeId: string | null, nodeLabels: Record<string, string>): TraceOutlineItemKind {
  if (event.event_type === "on_llm_start") return "llm_call";
  if (event.event_type === "on_tool_start") {
    return classifyToolOperation(buildLabel(event.payload?.tool_name, "tool"), getEventTags(event));
  }
  if (event.event_type === "on_chain_start") {
    return nodeId && nodeLabels[nodeId] ? "agent" : "subgraph_call";
  }
  if (event.event_type.endsWith("_error")) return "error";
  return nodeId && nodeLabels[nodeId] ? "agent" : "subgraph_call";
}

function inferLabel(event: TraceEvent, nodeId: string | null, nodeLabels: Record<string, string>): string {
  if (event.event_type === "on_llm_start") {
    return buildLabel(event.payload?.model_name, "llm");
  }
  if (event.event_type === "on_tool_start") {
    return buildLabel(event.payload?.tool_name, "tool");
  }
  if (event.event_type === "on_chain_start") {
    return inferChainLabel(event, nodeId, nodeLabels);
  }
  if (event.event_type.endsWith("_error")) {
    return buildLabel(event.payload?.error_type, "error");
  }
  return "operation";
}

function buildRunMaps(events: TraceEvent[], nodeLabels: Record<string, string>) {
  const nodeIdSet = new Set(Object.keys(nodeLabels));
  const runToNode = new Map<string, string>();
  const spanToNode = new Map<string, string>();

  for (const event of events) {
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
    if (!nodeId) continue;
    const runId = getRunIdFromEvent(event);
    if (runId) runToNode.set(runId, nodeId);
    if (event.span_id) spanToNode.set(event.span_id, nodeId);
  }

  return { runToNode, spanToNode };
}

function buildSupplementalChildren(run: MutableTraceRun): TraceOutlineItem[] {
  const items: TraceOutlineItem[] = [];

  if (
    run.kind === "agent"
    && run.startEvent?.event_type === "on_chain_start"
    && run.endEvent?.event_type === "on_chain_end"
    && isRecord(run.startEvent.payload?.inputs)
    && isRecord(run.endEvent.payload?.outputs)
  ) {
    const changedKeys = diffStateKeys(run.startEvent.payload.inputs, run.endEvent.payload.outputs);
    if (changedKeys.length > 0) {
      items.push({
        id: `${run.runId}:state`,
        runId: run.runId,
        parentRunId: run.runId,
        nodeId: run.nodeId,
        label: "state update",
        kind: "state_update",
        status: "success",
        operationId: run.endEvent.event_id,
        children: [],
      });
    }
  }

  if (run.errorEvent) {
    items.push({
      id: `${run.runId}:error`,
      runId: run.runId,
      parentRunId: run.runId,
      nodeId: run.nodeId,
      label: buildLabel(run.errorEvent.payload?.error_type, "error"),
      kind: "error",
      status: "error",
      operationId: run.errorEvent.event_id,
      children: [],
    });
  }

  return items;
}

function toOutlineItem(run: MutableTraceRun): TraceOutlineItem {
  const timestamps = [run.startEvent, run.endEvent, run.errorEvent]
    .filter((event): event is TraceEvent => Boolean(event))
    .map((event) => parseTimestampMs(event.timestamp))
    .sort((left, right) => left - right);
  const latencyMs = timestamps.length >= 2 ? timestamps[timestamps.length - 1] - timestamps[0] : undefined;

  const childRuns = run.children
    .sort((left, right) => {
      const leftTime = parseTimestampMs((left.startEvent ?? left.endEvent ?? left.errorEvent)?.timestamp ?? "");
      const rightTime = parseTimestampMs((right.startEvent ?? right.endEvent ?? right.errorEvent)?.timestamp ?? "");
      return leftTime - rightTime;
    })
    .map(toOutlineItem);

  const children = [...childRuns, ...buildSupplementalChildren(run)];

  return {
    id: run.runId,
    runId: run.runId,
    parentRunId: run.parentRunId,
    nodeId: run.nodeId,
    label: run.label,
    kind: run.kind,
    status: run.status,
    latencyMs,
    operationId: run.operationId,
    children,
  };
}

export function buildTraceOutline(events: TraceEvent[], nodeLabels: Record<string, string>): TraceOutlineItem[] {
  const orderedEvents = sortTraceEvents(events);
  const nodeIdSet = new Set(Object.keys(nodeLabels));
  const { runToNode, spanToNode } = buildRunMaps(orderedEvents, nodeLabels);
  const runs = new Map<string, MutableTraceRun>();

  function getOrCreateRun(event: TraceEvent): MutableTraceRun | null {
    const runId = getRunIdFromEvent(event);
    if (!runId) return null;
    const parentRunId = getParentRunIdFromEvent(event) ?? null;
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode) ?? null;
    const existing = runs.get(runId);
    if (existing) {
      if (!existing.parentRunId) existing.parentRunId = parentRunId;
      if (!existing.nodeId && nodeId) existing.nodeId = nodeId;
      return existing;
    }

    const created: MutableTraceRun = {
      runId,
      parentRunId,
      nodeId,
      kind: inferKind(event, nodeId, nodeLabels),
      label: inferLabel(event, nodeId, nodeLabels),
      status: "success",
      children: [],
    };
    runs.set(runId, created);
    return created;
  }

  for (const event of orderedEvents) {
    const run = getOrCreateRun(event);
    if (!run) continue;

    if (event.event_type.endsWith("_start")) {
      run.startEvent = event;
      run.kind = inferKind(event, run.nodeId, nodeLabels);
      run.label = inferLabel(event, run.nodeId, nodeLabels);
    }

    if (event.event_type.endsWith("_end")) {
      run.endEvent = event;
      run.status = "success";
      if (run.kind !== "agent" && run.kind !== "subgraph_call") {
        run.operationId = event.event_id;
      }
    }

    if (event.event_type.endsWith("_error")) {
      run.errorEvent = event;
      run.status = "error";
    }
  }

  for (const run of runs.values()) {
    if (!run.parentRunId) continue;
    const parent = runs.get(run.parentRunId);
    if (!parent) continue;
    parent.children.push(run);
  }

  const taggedRootsByNode = new Map<string, boolean>();
  for (const run of runs.values()) {
    if (!run.nodeId || run.startEvent?.event_type !== "on_chain_start") continue;
    if (!isTopLevelNodeChainStart(run.startEvent, run.nodeId, runToNode, spanToNode)) continue;
    if (getEventTags(run.startEvent).some((tag) => tag.startsWith("graph:step:"))) {
      taggedRootsByNode.set(run.nodeId, true);
    }
  }

  return [...runs.values()]
    .filter((run) => {
      if (!run.startEvent) return false;
      if (!run.nodeId) return false;
      if (run.startEvent.event_type !== "on_chain_start") return false;
      if (!isTopLevelNodeChainStart(run.startEvent, run.nodeId, runToNode, spanToNode)) return false;
      if (taggedRootsByNode.get(run.nodeId)) {
        return getEventTags(run.startEvent).some((tag) => tag.startsWith("graph:step:"));
      }
      return eventExplicitlyBelongsToNode(run.startEvent, run.nodeId);
    })
    .sort((left, right) => parseTimestampMs(left.startEvent?.timestamp ?? "") - parseTimestampMs(right.startEvent?.timestamp ?? ""))
    .map(toOutlineItem);
}
