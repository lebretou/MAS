export interface TraceEvent {
  event_id: string;
  trace_id: string;
  execution_id: string;
  timestamp: string;
  sequence?: number | null;
  event_type: string;
  agent_id?: string | null;
  span_id?: string | null;
  parent_span_id?: string | null;
  refs: Record<string, Record<string, unknown>>;
  payload: Record<string, unknown>;
}

export interface TraceMetadata {
  trace_id: string;
  event_count: number;
  graph_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface TraceSummaryEdge {
  from_agent: string;
  to_agent: string;
  message_count: number;
}

export interface TraceSummaryFailure {
  type: string;
  agent_id?: string | null;
  error_type?: string | null;
  message?: string | null;
  timestamp: string;
}

export interface TraceSummaryUsage {
  tool_name: string;
  call_count: number;
  avg_latency_ms?: number | null;
}

export interface TraceSummary {
  execution_id: string;
  trace_id: string;
  agents: string[];
  edges: TraceSummaryEdge[];
  messages_by_edge: Record<string, number>;
  failures: TraceSummaryFailure[];
  tool_usage: TraceSummaryUsage[];
  llm_usage: TraceSummaryUsage[];
  event_count: number;
}
