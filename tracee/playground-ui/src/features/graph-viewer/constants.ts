import type { PromptComponentType } from "../../types/prompt";

export const NODE_DIMENSIONS: Record<string, { width: number; height: number }> = {
  agent: { width: 220, height: 170 },
  terminal: { width: 80, height: 32 },
};

export const componentColors: Record<PromptComponentType, string> = {
  role: "#8b5cf6",
  task: "#3b82f6",
  constraints: "#f59e0b",
  inputs: "#10b981",
  outputs: "#06b6d4",
  examples: "#ec4899",
  safety: "#ef4444",
  tool_instructions: "#6366f1",
  external_information: "#84cc16",
};
