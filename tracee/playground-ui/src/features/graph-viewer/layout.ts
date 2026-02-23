import dagre from "@dagrejs/dagre";
import type { Node, Edge, Position } from "@xyflow/react";
import type { GraphNodeData } from "../../types/node-data";
import type { GraphEdgeData } from "../../types/node-data";
import { NODE_DIMENSIONS } from "./constants";

export function getLayoutedElements(
  nodes: Node<GraphNodeData>[],
  edges: Edge<GraphEdgeData>[],
  direction: "LR" | "TB" = "LR",
) {
  const g = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  const isHorizontal = direction === "LR";

  g.setGraph({ rankdir: direction, ranksep: 280, nodesep: 200, edgeSep: 100 });

  for (const node of nodes) {
    const dims = NODE_DIMENSIONS[node.type ?? "agent"];
    g.setNode(node.id, { width: dims.width, height: dims.height });
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const layoutedNodes = nodes.map((node) => {
    const pos = g.node(node.id);
    const dims = NODE_DIMENSIONS[node.type ?? "agent"];

    return {
      ...node,
      targetPosition: (isHorizontal ? "left" : "top") as Position,
      sourcePosition: (isHorizontal ? "right" : "bottom") as Position,
      position: {
        x: pos.x - dims.width / 2,
        y: pos.y - dims.height / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}
