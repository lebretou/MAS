import client from "./client";
import type { TraceMetadata, TraceEvent, TraceSummary } from "../types/trace";

export async function fetchTraces(limit = 100, offset = 0, graphId?: string | null): Promise<TraceMetadata[]> {
  const params: Record<string, string | number> = { limit, offset };
  if (graphId) params.graph_id = graphId;
  const { data } = await client.get<TraceMetadata[]>("/traces", { params });
  return data;
}

export async function fetchTraceEvents(traceId: string): Promise<TraceEvent[]> {
  const { data } = await client.get<TraceEvent[]>(`/traces/${traceId}`);
  return data;
}

export async function fetchTraceSummary(traceId: string): Promise<TraceSummary> {
  const { data } = await client.get<TraceSummary>(`/traces/${traceId}/summary`);
  return data;
}
