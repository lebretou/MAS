import { useEffect, useMemo, useState } from "react";
import type { Node, Edge } from "@xyflow/react";
import { fetchCognition, runCognitionAnalysis } from "../api/cognition";
import type { TraceCognition } from "../types/cognition";
import type { GraphNodeData, GraphEdgeData } from "../types/node-data";


interface UseCognitionOverlayResult {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  cognition: TraceCognition | null;
  loading: boolean;
  analyzing: boolean;
  analyze: () => Promise<void>;
}

export function useCognitionOverlay(
  traceId: string | null,
  baseNodes: Node<GraphNodeData>[],
  baseEdges: Edge<GraphEdgeData>[],
  active: boolean,
): UseCognitionOverlayResult {
  const [cognition, setCognition] = useState<TraceCognition | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // fetch cached cognition when layer becomes active
  useEffect(() => {
    if (!active || !traceId) {
      setCognition(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    fetchCognition(traceId)
      .then((result) => {
        if (!cancelled) setCognition(result);
      })
      .catch(() => {
        if (!cancelled) setCognition(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [traceId, active]);

  const analyze = async () => {
    if (!traceId) return;
    setAnalyzing(true);
    try {
      const result = await runCognitionAnalysis(traceId);
      setCognition(result);
    } catch (err) {
      console.error("cognition analysis failed", err);
    } finally {
      setAnalyzing(false);
    }
  };

  // merge cognition data onto nodes
  const nodes = useMemo(() => {
    return baseNodes.map((n) => {
      if (!cognition || n.data.nodeType !== "agent") return n;
      const nodeCog = cognition.node_cognitions[n.id];
      if (!nodeCog) return n;
      return {
        ...n,
        data: { ...n.data, cognition: nodeCog },
      };
    });
  }, [baseNodes, cognition]);

  // edges pass through unchanged — no visual modifications in cognition layer
  const edges = baseEdges;

  return { nodes, edges, cognition, loading, analyzing, analyze };
}
