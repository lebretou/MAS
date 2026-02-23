import { useEffect, useState, useCallback } from "react";
import type { Node, Edge } from "@xyflow/react";
import { fetchGraph } from "../api/graphs";
import { fetchGraphs } from "../api/graphs";
import { fetchLatestVersion } from "../api/prompts";
import type { GraphNodeData, GraphEdgeData } from "../types/node-data";
import type { JsonSchema } from "../types/schema";
import { getLayoutedElements } from "../features/graph-viewer/layout";

interface UseGraphResult {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  stateSchema: JsonSchema | null;
  graphId: string | null;
  graphIds: string[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useGraph(requestedGraphId?: string | null): UseGraphResult {
  const [nodes, setNodes] = useState<Node<GraphNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge<GraphEdgeData>[]>([]);
  const [stateSchema, setStateSchema] = useState<JsonSchema | null>(null);
  const [graphId, setGraphId] = useState<string | null>(requestedGraphId ?? null);
  const [graphIds, setGraphIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);

    // resolve graph id: use the requested one, or pick the first available
    let targetId = requestedGraphId ?? graphId;
    const allGraphs = await fetchGraphs();
    const ids = allGraphs.map((g) => g.graph_id);
    setGraphIds(ids);

    if (!targetId && ids.length > 0) {
      targetId = ids[0];
    }
    if (!targetId) {
      setLoading(false);
      setError("no graphs registered");
      return;
    }
    setGraphId(targetId);

    const topology = await fetchGraph(targetId);
    setStateSchema((topology.state_schema as JsonSchema) ?? null);

    // build raw react-flow nodes
    const rawNodes: Node<GraphNodeData>[] = topology.nodes.map((n) => {
      const meta = n.metadata ?? {};
      const isAgent = n.node_type === "agent";
      const nodeData: GraphNodeData = {
        label: n.label,
        nodeType: n.node_type as "agent" | "start" | "end",
        promptId: n.prompt_id ?? undefined,
        metadata: isAgent
          ? {
              model: meta.model as string | undefined,
              temperature: meta.temperature as number | undefined,
              hasTools: meta.has_tools as boolean | undefined,
            }
          : undefined,
      };

      return {
        id: n.node_id,
        type: isAgent ? "agent" : "terminal",
        position: { x: 0, y: 0 },
        data: nodeData,
      };
    });

    // build raw edges
    const rawEdges: Edge<GraphEdgeData>[] = topology.edges.map((e, idx) => ({
      id: `e-${e.source}-${e.target}-${idx}`,
      source: e.source,
      target: e.target,
      label: e.label ?? undefined,
      animated: e.conditional,
      style: e.conditional ? { strokeDasharray: "5 5" } : undefined,
      data: { conditional: e.conditional, label: e.label ?? undefined },
    }));

    // hydrate prompt components for agent nodes
    // nodes without prompt id are dropped
    const agentNodes = rawNodes.filter((n) => n.data.promptId);
    const promptResults = await Promise.allSettled(
      agentNodes.map((n) => fetchLatestVersion(n.data.promptId!)),
    );

    for (let i = 0; i < agentNodes.length; i++) {
      const result = promptResults[i];
      if (result.status === "fulfilled") {
        const version = result.value;
        agentNodes[i].data = {
          ...agentNodes[i].data,
          promptVersionId: version.version_id,
          components: version.components,
        };
      }
    }

    const { nodes: layouted, edges: layoutedEdges } = getLayoutedElements(rawNodes, rawEdges, "LR");
    setNodes(layouted);
    setEdges(layoutedEdges);
    setLoading(false);
  }, [requestedGraphId, graphId]);

  useEffect(() => {
    load().catch((err) => {
      setError(err.message ?? "failed to load graph");
      setLoading(false);
    });
  }, [load]);

  return { nodes, edges, stateSchema, graphId, graphIds, loading, error, refetch: load };
}
