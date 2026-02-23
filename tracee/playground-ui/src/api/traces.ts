import client from "./client";
import type { TraceMetadata, TraceEvent } from "../types/trace";

export async function fetchTraces(limit = 100, offset = 0): Promise<TraceMetadata[]> {
  const { data } = await client.get<TraceMetadata[]>("/traces", { params: { limit, offset } });
  return data;
}

export async function fetchTraceEvents(traceId: string): Promise<TraceEvent[]> {
  const { data } = await client.get<TraceEvent[]>(`/traces/${traceId}`);
  return data;
}

export async function fetchTraceSummary(traceId: string): Promise<Record<string, unknown>> {
  const { data } = await client.get<Record<string, unknown>>(`/traces/${traceId}/summary`);
  return data;
}
