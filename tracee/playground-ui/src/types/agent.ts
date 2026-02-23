export interface AgentRegistryEntry {
  agent_id: string;
  prompt_id?: string | null;
  prompt_version_id?: string | null;
  model?: string | null;
  temperature?: number | null;
  has_tools: boolean;
  metadata?: Record<string, unknown> | null;
  updated_at: string;
}
