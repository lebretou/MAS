import { useEffect, useMemo, useState } from "react";
import type { Node } from "@xyflow/react";
import { fetchTraceEvents } from "../api/traces";
import type { ExecutionFrame, GraphNodeData, NodeFrameState } from "../types/node-data";
import type { TraceEvent } from "../types/trace";
import { computeOverlay } from "./useTraceOverlay";

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

function cloneStateSnapshot(snapshot: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(snapshot)) as Record<string, unknown>;
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

function eventExplicitlyBelongsToNode(event: TraceEvent, nodeId: string): boolean {
  const hintAgentId = getHintAgentIdFromEvent(event);
  if (hintAgentId === nodeId) return true;
  if (event.agent_id === nodeId) return true;
  return getNodeIdFromEvent(event) === nodeId;
}

export function computeExecutionFrames(
  events: TraceEvent[],
  nodeLabels: Record<string, string>,
): ExecutionFrame[] {
  const orderedEvents = sortTraceEvents(events);
  const nodeIdSet = new Set(Object.keys(nodeLabels));
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

  for (const event of orderedEvents) {
    const nodeId = resolveAgentNodeId(event);
    if (!nodeId) continue;
    const runId = getRunIdFromEvent(event);
    if (runId) runToNode.set(runId, nodeId);
    if (event.span_id) spanToNode.set(event.span_id, nodeId);
  }

  const rootRunIdsByNode = new Map<string, Set<string>>();
  for (const event of orderedEvents) {
    if (event.event_type !== "on_chain_start") continue;
    const nodeId = resolveAgentNodeId(event);
    const runId = getRunIdFromEvent(event);
    if (!nodeId || !runId || !eventExplicitlyBelongsToNode(event, nodeId)) continue;
    const parentRunId = getParentRunIdFromEvent(event);
    const parentRunNodeId = parentRunId ? runToNode.get(parentRunId) : undefined;
    if (parentRunNodeId === nodeId) continue;
    const parentSpanNodeId = event.parent_span_id ? spanToNode.get(event.parent_span_id) : undefined;
    if (parentSpanNodeId === nodeId) continue;
    const existing = rootRunIdsByNode.get(nodeId) ?? new Set<string>();
    existing.add(runId);
    rootRunIdsByNode.set(nodeId, existing);
  }

  const frames: ExecutionFrame[] = [];
  let previousState: Record<string, unknown> = {};

  const initialEventIndex = orderedEvents.findIndex((event) => {
    if (event.event_type !== "on_chain_start" || !isRecord(event.payload?.inputs)) return false;
    const nodeId = resolveAgentNodeId(event);
    const runId = getRunIdFromEvent(event);
    if (!nodeId || !runId) return false;
    return rootRunIdsByNode.get(nodeId)?.has(runId) ?? false;
  });

  if (initialEventIndex >= 0) {
    const initialEvent = orderedEvents[initialEventIndex];
    const initialState = cloneStateSnapshot(initialEvent.payload.inputs as Record<string, unknown>);
    previousState = initialState;
    frames.push({
      index: 0,
      nodeId: null,
      label: "Initial state",
      timestamp: initialEvent.timestamp,
      eventId: initialEvent.event_id,
      eventOrder: initialEventIndex - 1,
      endSequence: Math.max(0, (initialEvent.sequence ?? 1) - 1),
      latencyMs: 0,
      changedKeys: [],
      stateSnapshot: initialState,
    });
  }

  orderedEvents.forEach((event, eventOrder) => {
    if (event.event_type !== "on_chain_end") return;
    const nodeId = resolveAgentNodeId(event);
    const runId = getRunIdFromEvent(event);
    if (!nodeId || !runId) return;
    if (!rootRunIdsByNode.get(nodeId)?.has(runId)) return;
    if (!isRecord(event.payload?.outputs)) return;

    const nextState = cloneStateSnapshot(event.payload.outputs as Record<string, unknown>);
    const previousFrame = frames[frames.length - 1];
    const latencyMs = parseTimestampMs(event.timestamp) - parseTimestampMs(previousFrame.timestamp);

    frames.push({
      index: frames.length,
      nodeId,
      label: nodeLabels[nodeId] ?? nodeId,
      timestamp: event.timestamp,
      eventId: event.event_id,
      eventOrder,
      endSequence: event.sequence ?? eventOrder,
      latencyMs: Math.max(0, latencyMs),
      changedKeys: diffStateKeys(previousState, nextState),
      stateSnapshot: nextState,
    });
    previousState = nextState;
  });

  return frames;
}

export function getNodeFrameState(
  nodeId: string,
  frames: ExecutionFrame[],
  activeFrameIndex: number | null,
): NodeFrameState {
  if (activeFrameIndex == null || activeFrameIndex < 0) return "idle";
  const activeFrame = frames[activeFrameIndex];
  if (activeFrame?.nodeId === nodeId) return "active";

  const nodeFrameIndexes = frames
    .filter((frame) => frame.nodeId === nodeId)
    .map((frame) => frame.index);

  if (nodeFrameIndexes.length === 0) return "idle";
  if (nodeFrameIndexes.some((frameIndex) => frameIndex < activeFrameIndex)) return "completed";
  return "upcoming";
}

interface TracePlaybackResult {
  nodes: Node<GraphNodeData>[];
  frames: ExecutionFrame[];
  activeFrame: ExecutionFrame | null;
  error: boolean;
}

export function useTracePlayback(
  traceId: string | null,
  baseNodes: Node<GraphNodeData>[],
  activeFrameIndex: number | null,
): TracePlaybackResult {
  const [events, setEvents] = useState<TraceEvent[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!traceId) {
      setEvents(null);
      setError(false);
      return;
    }

    let cancelled = false;
    setEvents(null);
    setError(false);
    fetchTraceEvents(traceId)
      .then((nextEvents) => {
        if (cancelled) return;
        setEvents(nextEvents);
        setError(false);
      })
      .catch(() => {
        if (cancelled) return;
        setEvents(null);
        setError(true);
      });

    return () => {
      cancelled = true;
    };
  }, [traceId]);

  const agentNodes = useMemo(
    () => baseNodes.filter((node) => node.data.nodeType === "agent"),
    [baseNodes],
  );

  const nodeLabels = useMemo(
    () =>
      Object.fromEntries(agentNodes.map((node) => [node.id, node.data.label])),
    [agentNodes],
  );

  const orderedEvents = useMemo(() => sortTraceEvents(events ?? []), [events]);
  const frames = useMemo(() => computeExecutionFrames(orderedEvents, nodeLabels), [orderedEvents, nodeLabels]);

  const resolvedFrameIndex = useMemo(() => {
    if (!traceId || frames.length === 0 || activeFrameIndex == null) return null;
    return Math.min(Math.max(activeFrameIndex, 0), frames.length - 1);
  }, [traceId, frames, activeFrameIndex]);

  const activeFrame = resolvedFrameIndex == null ? null : frames[resolvedFrameIndex] ?? null;

  const scopedEvents = useMemo(() => {
    if (!traceId) return [];
    if (!activeFrame) return orderedEvents;
    return orderedEvents.slice(0, activeFrame.eventOrder + 1);
  }, [traceId, orderedEvents, activeFrame]);

  const displayNodes = useMemo(() => {
    if (!traceId) {
      return baseNodes.map((node) => ({
        ...node,
        data: { ...node.data, execution: undefined, playback: undefined },
      }));
    }

    const overlay = computeOverlay(scopedEvents, agentNodes.map((node) => node.id));
    return baseNodes.map((node) => {
      if (node.data.nodeType !== "agent") return node;
      const execution = overlay.get(node.id);
      const playback = activeFrame
        ? { frameState: getNodeFrameState(node.id, frames, resolvedFrameIndex) }
        : undefined;

      return {
        ...node,
        data: {
          ...node.data,
          execution,
          playback,
        },
      };
    });
  }, [traceId, baseNodes, scopedEvents, agentNodes, activeFrame, frames, resolvedFrameIndex]);

  return {
    nodes: displayNodes,
    frames,
    activeFrame,
    error,
  };
}
