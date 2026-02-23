import client from "./client";
import type { AgentRegistryEntry } from "../types/agent";

export async function fetchAgents(): Promise<AgentRegistryEntry[]> {
  const { data } = await client.get<AgentRegistryEntry[]>("/agents");
  return data;
}

export async function fetchAgent(agentId: string): Promise<AgentRegistryEntry> {
  const { data } = await client.get<AgentRegistryEntry>(`/agents/${agentId}`);
  return data;
}
