import type { PromptComponent } from "./prompt";
import type { JsonSchema } from "./schema";
import type { TraceEvent } from "./trace";

export type AgentOperationType =
  | "llm_call"
  | "tool_call"
  | "rag_retrieve"
  | "code_exec"
  | "subgraph_call"
  | "error";

export interface AgentOperation {
  id: string;
  type: AgentOperationType;
  label: string;
  status: "success" | "error";
  latencyMs?: number;
  tokenCount?: number;
}

export interface ExecutionData {
  invoked: boolean;
  status?: "success" | "error";
  latencyMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  retryCount?: number;
  llmInput?: string;
  llmOutput?: string;
  operations?: AgentOperation[];
  events?: TraceEvent[];
}

export interface GraphNodeData extends Record<string, unknown> {
  label: string;
  nodeType: "agent" | "start" | "end";
  promptId?: string;
  promptVersionId?: string;
  components?: PromptComponent[];
  stateSchema?: JsonSchema;
  outputSchema?: JsonSchema;
  metadata?: {
    model?: string;
    temperature?: number;
    hasTools?: boolean;
    hasRetry?: boolean;
  };
  execution?: ExecutionData;
}

export interface GraphEdgeData extends Record<string, unknown> {
  conditional: boolean;
  label?: string;
}
