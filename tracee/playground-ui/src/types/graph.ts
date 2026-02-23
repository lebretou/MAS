export interface GraphNode {
  node_id: string;
  label: string;
  node_type: string;
  prompt_id?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  conditional: boolean;
  label?: string | null;
}

export interface GraphTopology {
  graph_id: string;
  name: string;
  description?: string | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
  state_schema?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}
