import ELK, { type ElkNode, type ElkExtendedEdge, type ElkPort } from "elkjs/lib/elk.bundled.js";
import type { Node, Edge, Position } from "@xyflow/react";
import type { GraphNodeData, PortInfo } from "../../types/node-data";
import type { GraphEdgeData } from "../../types/node-data";
import { NODE_DIMENSIONS } from "./constants";

const elk = new ELK();

export async function getLayoutedElements(
  nodes: Node<GraphNodeData>[],
  edges: Edge<GraphEdgeData>[],
  direction: "LR" | "TB" = "LR",
) {
  const isHorizontal = direction === "LR";

  // collect ports per node from edges
  const portsByNode = new Map<string, ElkPort[]>();
  const ensurePorts = (nodeId: string) => {
    if (!portsByNode.has(nodeId)) portsByNode.set(nodeId, []);
    return portsByNode.get(nodeId)!;
  };

  const elkEdges: ElkExtendedEdge[] = edges.map((edge) => {
    const srcPortId = `${edge.id}-src`;
    const tgtPortId = `${edge.id}-tgt`;

    const srcDims = NODE_DIMENSIONS[nodes.find((n) => n.id === edge.source)?.type ?? "agent"];
    const tgtDims = NODE_DIMENSIONS[nodes.find((n) => n.id === edge.target)?.type ?? "agent"];

    ensurePorts(edge.source).push({
      id: srcPortId,
      width: 1,
      height: 1,
      // place on right side for LR, bottom for TB
      x: isHorizontal ? srcDims.width : srcDims.width / 2,
      y: isHorizontal ? srcDims.height / 2 : srcDims.height,
      layoutOptions: {
        "elk.port.side": isHorizontal ? "EAST" : "SOUTH",
      },
    });

    ensurePorts(edge.target).push({
      id: tgtPortId,
      width: 1,
      height: 1,
      // place on left side for LR, top for TB
      x: isHorizontal ? 0 : tgtDims.width / 2,
      y: isHorizontal ? tgtDims.height / 2 : 0,
      layoutOptions: {
        "elk.port.side": isHorizontal ? "WEST" : "NORTH",
      },
    });

    return {
      id: edge.id,
      sources: [srcPortId],
      targets: [tgtPortId],
    };
  });

  const elkNodes: ElkNode[] = nodes.map((node) => {
    const dims = NODE_DIMENSIONS[node.type ?? "agent"];
    return {
      id: node.id,
      width: dims.width,
      height: dims.height,
      ports: portsByNode.get(node.id) ?? [],
      layoutOptions: {
        "elk.portConstraints": "FIXED_SIDE",
      },
    };
  });

  const graph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": isHorizontal ? "RIGHT" : "DOWN",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.nodeNodeBetweenLayers": "280",
      "elk.spacing.nodeNode": "200",
      "elk.spacing.edgeEdge": "100",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    },
    children: elkNodes,
    edges: elkEdges,
  };

  const layoutedGraph = await elk.layout(graph);

  // extract computed port positions per node
  const nodePortMap = new Map<string, PortInfo[]>();
  for (const elkNode of layoutedGraph.children ?? []) {
    if (!elkNode.ports?.length) continue;
    const ports: PortInfo[] = elkNode.ports.map((p) => ({
      id: p.id,
      type: p.id.endsWith("-src") ? "source" as const : "target" as const,
      x: p.x ?? 0,
      y: p.y ?? 0,
    }));
    nodePortMap.set(elkNode.id, ports);
  }

  const layoutedNodes = nodes.map((node) => {
    const elkNode = layoutedGraph.children?.find((n) => n.id === node.id);
    return {
      ...node,
      targetPosition: (isHorizontal ? "left" : "top") as Position,
      sourcePosition: (isHorizontal ? "right" : "bottom") as Position,
      position: {
        x: elkNode?.x ?? 0,
        y: elkNode?.y ?? 0,
      },
      data: {
        ...node.data,
        ports: nodePortMap.get(node.id),
      },
    };
  });

  // attach sourceHandle/targetHandle to edges so React Flow connects to specific ports
  const layoutedEdges = edges.map((edge) => ({
    ...edge,
    sourceHandle: `${edge.id}-src`,
    targetHandle: `${edge.id}-tgt`,
  }));

  return { nodes: layoutedNodes, edges: layoutedEdges };
}
