import type { TraceEvent } from "../types/trace";

export function getNodeIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langgraph as Record<string, unknown> | undefined)?.node as string | undefined;
}

export function getHintAgentIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.hint as Record<string, unknown> | undefined)?.agent_id as string | undefined;
}

export function getRunIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langchain as Record<string, unknown> | undefined)?.run_id as string | undefined;
}

export function getParentRunIdFromEvent(event: TraceEvent): string | undefined {
  return (event.refs?.langchain as Record<string, unknown> | undefined)?.parent_run_id as string | undefined;
}

export function parseTimestampMs(timestamp: string): number {
  return new Date(timestamp).getTime();
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function valuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function diffStateKeys(
  previousState: Record<string, unknown>,
  nextState: Record<string, unknown>,
): string[] {
  return Array.from(new Set([...Object.keys(previousState), ...Object.keys(nextState)]))
    .filter((key) => !valuesEqual(previousState[key], nextState[key]))
    .sort();
}

export function sortTraceEvents(events: TraceEvent[]): TraceEvent[] {
  return [...events].sort((left, right) => {
    const leftSequence = left.sequence ?? Number.MAX_SAFE_INTEGER;
    const rightSequence = right.sequence ?? Number.MAX_SAFE_INTEGER;
    if (leftSequence !== rightSequence) return leftSequence - rightSequence;
    return parseTimestampMs(left.timestamp) - parseTimestampMs(right.timestamp);
  });
}

export function eventExplicitlyBelongsToNode(event: TraceEvent, nodeId: string): boolean {
  const hintAgentId = getHintAgentIdFromEvent(event);
  if (hintAgentId === nodeId) return true;
  if (event.agent_id === nodeId) return true;
  return getNodeIdFromEvent(event) === nodeId;
}

export function resolveAgentNodeId(
  event: TraceEvent,
  nodeIdSet: Set<string>,
  runToNode: Map<string, string>,
  spanToNode: Map<string, string>,
): string | undefined {
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
