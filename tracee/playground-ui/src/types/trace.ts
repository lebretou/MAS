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
  created_at: string;
  updated_at: string;
}
