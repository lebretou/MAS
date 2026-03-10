import { useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import { NODE_DIMENSIONS } from "../constants";
import type { GraphEdgeData, GraphNodeData } from "../../../types/node-data";
import type { TraceSummary } from "../../../types/trace";

const MINIMAP_PADDING = 24;
const LAYOUT_SCALE = 0.65;
const NODE_WIDTH = 100;
const NODE_HEIGHT = 100;
const NODE_RX = 30;
const ACTIVE_EDGE_STROKE = "#219ebc";
const INACTIVE_EDGE_STROKE = "#d4d4d8";
const ACTIVE_NODE_FILL = "#219ebc";
const INACTIVE_NODE_FILL = "#d4d4d8";
const EDGE_STROKE_ACTIVE = 33;
const EDGE_STROKE_INACTIVE = 30;

interface TraceMinimapPreviewProps {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  summary?: TraceSummary | null;
}

interface TraceMinimapNodeModel {
  id: string;
  x: number;
  y: number;
  isActive: boolean;
}

interface TraceMinimapEdgeModel {
  id: string;
  source: string;
  target: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  isActive: boolean;
}

export interface TraceMinimapModel {
  viewBox: string;
  nodes: TraceMinimapNodeModel[];
  edges: TraceMinimapEdgeModel[];
}

function getTraceEdgeKey(source: string, target: string): string {
  return `${source}->${target}`;
}

export function buildTraceMinimapModel(
  nodes: Node<GraphNodeData>[],
  edges: Edge<GraphEdgeData>[],
  summary?: TraceSummary | null,
): TraceMinimapModel | null {
  if (nodes.length === 0) return null;

  const activeAgents = new Set(summary?.agents ?? []);
  const activeEdges = new Set((summary?.edges ?? []).map((edge) => getTraceEdgeKey(edge.from_agent, edge.to_agent)));

  const previewNodes = nodes.map((node) => {
    const dims = NODE_DIMENSIONS[node.type ?? "agent"];
    const width = typeof node.measured?.width === "number" ? node.measured.width : dims.width;
    const height = typeof node.measured?.height === "number" ? node.measured.height : dims.height;
    const centerX = (node.position.x + width / 2) * LAYOUT_SCALE;
    const centerY = (node.position.y + height / 2) * LAYOUT_SCALE;

    return {
      id: node.id,
      x: centerX - NODE_WIDTH / 2,
      y: centerY - NODE_HEIGHT / 2,
      isActive: activeAgents.has(node.id),
    };
  });

  const nodeById = new Map(previewNodes.map((node) => [node.id, node]));
  const previewEdges = edges.flatMap((edge) => {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (!source || !target) return [];

    return [{
      id: edge.id,
      source: edge.source,
      target: edge.target,
      x1: source.x + NODE_WIDTH / 2,
      y1: source.y + NODE_HEIGHT / 2,
      x2: target.x + NODE_WIDTH / 2,
      y2: target.y + NODE_HEIGHT / 2,
      isActive: activeEdges.has(getTraceEdgeKey(edge.source, edge.target)),
    }];
  });

  const minX = Math.min(...previewNodes.map((node) => node.x));
  const minY = Math.min(...previewNodes.map((node) => node.y));
  const maxX = Math.max(...previewNodes.map((node) => node.x + NODE_WIDTH));
  const maxY = Math.max(...previewNodes.map((node) => node.y + NODE_HEIGHT));
  const viewBox = [
    minX - MINIMAP_PADDING,
    minY - MINIMAP_PADDING,
    maxX - minX + MINIMAP_PADDING * 2,
    maxY - minY + MINIMAP_PADDING * 2,
  ].join(" ");

  return {
    viewBox,
    nodes: previewNodes,
    edges: previewEdges,
  };
}

export function TraceMinimapPreview({ nodes, edges, summary }: TraceMinimapPreviewProps) {
  const model = useMemo(() => buildTraceMinimapModel(nodes, edges, summary), [nodes, edges, summary]);

  if (!model) return null;

  return (
    <div className="trace-selector-item__preview" aria-hidden="true">
      <svg className="trace-minimap" viewBox={model.viewBox} preserveAspectRatio="xMidYMid meet">
        <g>
          {model.edges.map((edge) => (
            <line
              key={edge.id}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              stroke={edge.isActive ? ACTIVE_EDGE_STROKE : INACTIVE_EDGE_STROKE}
              strokeWidth={edge.isActive ? EDGE_STROKE_ACTIVE : EDGE_STROKE_INACTIVE}
              strokeLinecap="round"
              opacity={edge.isActive ? 1 : 0.95}
            />
          ))}
        </g>
        <g>
          {model.nodes.map((node) => (
            <rect
              key={node.id}
              x={node.x}
              y={node.y}
              width={NODE_WIDTH}
              height={NODE_HEIGHT}
              rx={NODE_RX}
              fill={node.isActive ? ACTIVE_NODE_FILL : INACTIVE_NODE_FILL}
              opacity={node.isActive ? 1 : 0.9}
            />
          ))}
        </g>
      </svg>
    </div>
  );
}
