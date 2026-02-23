import type { PromptComponent } from "./prompt";
import type { JsonSchema } from "./schema";
import type { TraceEvent } from "./trace";

export interface ExecutionData {
  invoked: boolean;
  status?: "success" | "error";
  latencyMs?: number;
  promptTokens?: number;
  completionTokens?: number;
  llmInput?: string;
  llmOutput?: string;
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
  };
  execution?: ExecutionData;
}

export interface GraphEdgeData extends Record<string, unknown> {
  conditional: boolean;
  label?: string;
}
