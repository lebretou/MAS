import { useEffect, useMemo, useState } from "react";
import type { Node, Edge } from "@xyflow/react";
import { fetchTraceEvents, fetchTraceSummary } from "../api/traces";
import type { ExecutionFrame, GraphNodeData, GraphEdgeData, NodeFrameState } from "../types/node-data";
import type { TraceEvent, TraceSummary } from "../types/trace";
import {
  diffStateKeys,
  getEventTags,
  getRunIdFromEvent,
  isRecord,
  isTopLevelNodeChainStart,
  parseTimestampMs,
  resolveAgentNodeId,
  sortTraceEvents,
} from "../utils/traceEventUtils";
import { computeOverlay } from "./useTraceOverlay";
import { ACTIVE_EDGE_STYLE, INACTIVE_EDGE_STYLE, INACTIVE_LABEL_STYLE } from "./edgeStyles";
import { buildTraceOutline } from "../utils/traceOutline";

function cloneStateSnapshot(snapshot: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(snapshot)) as Record<string, unknown>;
}

/** builds ordered frames from chain_end events (one per node transition); includes initial state frame if present */
export function computeExecutionFrames(
  events: TraceEvent[],
  nodeLabels: Record<string, string>,
): ExecutionFrame[] {
  const orderedEvents = sortTraceEvents(events);
  const nodeIdSet = new Set(Object.keys(nodeLabels));
  const runToNode = new Map<string, string>();
  const spanToNode = new Map<string, string>();

  // resolve run/span -> node so we can attribute chain events to nodes
  for (const event of orderedEvents) {
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
    if (!nodeId) continue;
    const runId = getRunIdFromEvent(event);
    if (runId) runToNode.set(runId, nodeId);
    if (event.span_id) spanToNode.set(event.span_id, nodeId);
  }

  // collect top-level chain start candidates per node, noting which carry graph:step:* tags
  const rootCandidatesByNode = new Map<string, Array<{ runId: string; isGraphTagged: boolean }>>();
  for (const event of orderedEvents) {
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
    const runId = getRunIdFromEvent(event);
    if (!nodeId || !runId) continue;
    if (!isTopLevelNodeChainStart(event, nodeId, runToNode, spanToNode)) continue;
    const candidates = rootCandidatesByNode.get(nodeId) ?? [];
    candidates.push({ runId, isGraphTagged: getEventTags(event).some((t) => t.startsWith("graph:step:")) });
    rootCandidatesByNode.set(nodeId, candidates);
  }

  // prefer graph:step:-tagged starts when present; fall back to all top-level starts
  const rootRunIdsByNode = new Map<string, Set<string>>();
  for (const [nodeId, candidates] of rootCandidatesByNode) {
    const hasTagged = candidates.some((c) => c.isGraphTagged);
    const filtered = hasTagged ? candidates.filter((c) => c.isGraphTagged) : candidates;
    rootRunIdsByNode.set(nodeId, new Set(filtered.map((c) => c.runId)));
  }

  const frames: ExecutionFrame[] = [];
  let previousState: Record<string, unknown> = {};

  // optional frame 0 from first on_chain_start inputs
  const initialEventIndex = orderedEvents.findIndex((event) => {
    if (event.event_type !== "on_chain_start" || !isRecord(event.payload?.inputs)) return false;
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
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
    const nodeId = resolveAgentNodeId(event, nodeIdSet, runToNode, spanToNode);
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

/** whether node is idle, active at current frame, completed, or upcoming */
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

/** derive the set of activated edge keys.
 *  for non-terminal edges, use the invoked agent set as a lightweight fallback.
 *  for END edges, only mark the branch from the final visible frame when it
 *  explicitly routes to end. */
export function computeActivatedEdgeKeys(
  summary: TraceSummary | null,
  baseEdges: Edge<GraphEdgeData>[],
  baseNodes: Node<GraphNodeData>[],
  finalVisibleFrame: ExecutionFrame | null,
): Set<string> {
  if (!summary) return new Set();
  const invokedAgents = new Set(summary.agents);
  const startIds = new Set(
    baseNodes.filter((n) => n.data.nodeType === "start").map((n) => n.id),
  );
  const endIds = new Set(
    baseNodes.filter((n) => n.data.nodeType === "end").map((n) => n.id),
  );
  const nextAgent = typeof finalVisibleFrame?.stateSnapshot?.next_agent === "string"
    ? finalVisibleFrame.stateSnapshot.next_agent
    : null;

  const keys = new Set<string>();
  for (const edge of baseEdges) {
    if (endIds.has(edge.target)) {
      if (nextAgent === "end" && finalVisibleFrame?.nodeId === edge.source) {
        keys.add(`${edge.source}->${edge.target}`);
      }
      continue;
    }

    const srcInvoked = invokedAgents.has(edge.source) || startIds.has(edge.source);
    const tgtInvoked = invokedAgents.has(edge.target);
    if (srcInvoked && tgtInvoked) {
      keys.add(`${edge.source}->${edge.target}`);
    }
  }
  return keys;
}

interface TracePlaybackResult {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  frames: ExecutionFrame[];
  outline: import("../types/node-data").TraceOutlineItem[];
  activeFrame: ExecutionFrame | null;
  loading: boolean;
  error: boolean;
}

/** fetches trace, builds frames from chain_end, and returns nodes with execution/playback state up to activeFrameIndex */
export function useTracePlayback(
  traceId: string | null,
  baseNodes: Node<GraphNodeData>[],
  baseEdges: Edge<GraphEdgeData>[],
  activeFrameIndex: number | null,
): TracePlaybackResult {
  const [events, setEvents] = useState<TraceEvent[] | null>(null);
  const [summary, setSummary] = useState<TraceSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!traceId) {
      setEvents(null);
      setSummary(null);
      setLoading(false);
      setError(false);
      return;
    }

    let cancelled = false;
    setEvents(null);
    setSummary(null);
    setLoading(true);
    setError(false);

    Promise.all([fetchTraceEvents(traceId), fetchTraceSummary(traceId)])
      .then(([nextEvents, nextSummary]) => {
        if (cancelled) return;
        setEvents(nextEvents);
        setSummary(nextSummary);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setEvents(null);
        setSummary(null);
        setLoading(false);
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
  const outline = useMemo(() => buildTraceOutline(orderedEvents, nodeLabels), [orderedEvents, nodeLabels]);

  const resolvedFrameIndex = useMemo(() => {
    if (!traceId || frames.length === 0 || activeFrameIndex == null) return null;
    return Math.min(Math.max(activeFrameIndex, 0), frames.length - 1);
  }, [traceId, frames, activeFrameIndex]);

  const activeFrame = resolvedFrameIndex == null ? null : frames[resolvedFrameIndex] ?? null;
  const finalVisibleFrame = useMemo(() => {
    if (frames.length === 0) return null;
    if (resolvedFrameIndex == null) return frames[frames.length - 1] ?? null;
    return frames[resolvedFrameIndex] ?? null;
  }, [frames, resolvedFrameIndex]);

  // events visible at current scrub position (all events if no frame selected)
  const scopedEvents = useMemo(() => {
    if (!traceId) return [];
    if (!activeFrame) return orderedEvents;
    return orderedEvents.slice(0, activeFrame.eventOrder + 1);
  }, [traceId, orderedEvents, activeFrame]);

  const activatedEdgeKeys = useMemo(
    () => computeActivatedEdgeKeys(summary, baseEdges, baseNodes, finalVisibleFrame),
    [summary, baseEdges, baseNodes, finalVisibleFrame],
  );

  // base nodes with execution overlay and playback frameState for current frame
  const displayNodes = useMemo(() => {
    if (!traceId) {
      return baseNodes.map((node) => ({
        ...node,
        data: { ...node.data, execution: undefined, playback: undefined },
      }));
    }

    const overlay = computeOverlay(scopedEvents, agentNodes.map((node) => node.id));

    // terminal nodes: START is always active when a trace is loaded;
    // END nodes are active only if an activated edge points to them
    const activeTerminalTargets = new Set(
      [...activatedEdgeKeys].map((key) => key.split("->")[1]),
    );

    return baseNodes.map((node) => {
      if (node.data.nodeType === "start") {
        return {
          ...node,
          data: { ...node.data, playback: { frameState: "completed" as NodeFrameState } },
        };
      }

      if (node.data.nodeType === "end") {
        const reached = activeTerminalTargets.has(node.id);
        return {
          ...node,
          data: { ...node.data, playback: { frameState: reached ? "completed" as NodeFrameState : "idle" as NodeFrameState } },
        };
      }

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
  }, [traceId, baseNodes, scopedEvents, agentNodes, activeFrame, frames, resolvedFrameIndex, activatedEdgeKeys]);

  const displayEdges = useMemo(() => {
    if (!traceId) return baseEdges;
    return baseEdges.map((edge) => {
      const key = `${edge.source}->${edge.target}`;
      const isActive = activatedEdgeKeys.has(key);
      return {
        ...edge,
        style: isActive ? ACTIVE_EDGE_STYLE : INACTIVE_EDGE_STYLE,
        labelStyle: isActive ? undefined : INACTIVE_LABEL_STYLE,
      };
    });
  }, [traceId, baseEdges, activatedEdgeKeys]);

  return {
    nodes: displayNodes,
    edges: displayEdges,
    frames,
    outline,
    activeFrame,
    loading,
    error,
  };
}
