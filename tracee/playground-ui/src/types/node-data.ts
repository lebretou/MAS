import type { NodeCognition } from "./cognition";
import type { PromptComponent } from "./prompt";
import type { JsonSchema } from "./schema";
import type { TraceEvent } from "./trace";

export type AgentOperationType =
  | "llm_call"
  | "tool_call"
  | "rag_retrieve"
  | "code_exec"
  | "subgraph_call"
  | "state_update"
  | "error";

export interface ExecutionFrame {
  index: number;
  nodeId: string | null;
  label: string;
  timestamp: string;
  eventId: string;
  eventOrder: number;
  endSequence: number;
  latencyMs: number;
  changedKeys: string[];
  stateSnapshot: Record<string, unknown>;
}

export type TraceOutlineItemKind = "agent" | AgentOperationType;

export interface TraceOutlineItem {
  id: string;
  runId: string;
  parentRunId: string | null;
  nodeId: string | null;
  label: string;
  kind: TraceOutlineItemKind;
  status: "success" | "error";
  latencyMs?: number;
  operationId?: string;
  children: TraceOutlineItem[];
}

export type NodeFrameState = "active" | "completed" | "upcoming" | "idle";

export interface AgentOperation {
  id: string;
  type: AgentOperationType;
  label: string;
  status: "success" | "error";
  latencyMs?: number;
  tokenCount?: number;
  input?: unknown;
  output?: unknown;
  metadata?: Record<string, unknown>;
  errorMessage?: string;
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
  llmOutputValue?: unknown;
  operations?: AgentOperation[];
  events?: TraceEvent[];
}

export interface PortInfo {
  id: string;
  type: "source" | "target";
  x: number;
  y: number;
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
  cognition?: NodeCognition;
  playback?: {
    frameState: NodeFrameState;
  };
  ports?: PortInfo[];
}

export interface GraphEdgeData extends Record<string, unknown> {
  conditional: boolean;
  label?: string;
}
