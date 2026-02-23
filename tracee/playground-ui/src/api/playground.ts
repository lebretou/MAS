import client from "./client";
import type { PlaygroundRun, PlaygroundRunCreate, PlaygroundRunResponse } from "../types/playground";

export async function createRun(req: PlaygroundRunCreate): Promise<PlaygroundRun> {
  const { data } = await client.post<PlaygroundRunResponse>("/playground/run/", req);
  return data.run;
}

export async function fetchRuns(): Promise<PlaygroundRun[]> {
  const { data } = await client.get<PlaygroundRun[]>("/playground/runs/");
  return data;
}

export async function fetchRun(runId: string): Promise<PlaygroundRun> {
  const { data } = await client.get<PlaygroundRun>(`/playground/runs/${runId}/`);
  return data;
}
