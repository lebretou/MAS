export interface NodeCognition {
  agent_id: string;
  description: string;
  handoff_description: string;
}

export interface TraceCognition {
  trace_id: string;
  graph_id: string | null;
  node_cognitions: Record<string, NodeCognition>;
  narrative: string;
  created_at: string;
}
