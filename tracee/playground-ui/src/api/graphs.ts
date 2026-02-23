import client from "./client";
import type { GraphTopology } from "../types/graph";

export async function fetchGraphs(): Promise<GraphTopology[]> {
  const { data } = await client.get<GraphTopology[]>("/graphs");
  return data;
}

export async function fetchGraph(graphId: string): Promise<GraphTopology> {
  const { data } = await client.get<GraphTopology>(`/graphs/${graphId}`);
  return data;
}
