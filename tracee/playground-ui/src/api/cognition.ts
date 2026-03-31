import client from "./client";
import type { TraceCognition } from "../types/cognition";

export async function fetchCognition(traceId: string): Promise<TraceCognition | null> {
  const { data } = await client.get<TraceCognition>(`/traces/${traceId}/cognition`).catch((err) => {
    if (err.response?.status === 404) return { data: null };
    throw err;
  });
  return data;
}

export async function runCognitionAnalysis(
  traceId: string,
  graphId?: string,
): Promise<TraceCognition> {
  const params = graphId ? { graph_id: graphId } : {};
  const { data } = await client.post<TraceCognition>(`/traces/${traceId}/analyze`, null, { params });
  return data;
}
