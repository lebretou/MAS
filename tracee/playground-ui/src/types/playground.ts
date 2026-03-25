import type { PromptComponent, PromptTool } from "./prompt";

export interface PlaygroundToolCall {
  call_id?: string | null;
  name: string;
  arguments?: unknown;
}

export interface PlaygroundRun {
  run_id: string;
  created_at: string;
  prompt_id: string;
  version_id: string;
  model: string;
  provider: string;
  temperature: number;
  max_tokens: number | null;
  input_variables: Record<string, string>;
  resolved_prompt: string;
  output_schema?: Record<string, unknown> | null;
  tools?: PromptTool[] | null;
  tool_calls?: PlaygroundToolCall[] | null;
  output: string;
  output_schema_used?: boolean;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  model_config_id: string | null;
  created_by: string | null;
  tags: string[] | null;
  notes: string | null;
}

export interface PlaygroundRunCreate {
  model_config_id?: string | null;
  prompt_id: string;
  version_id?: string;
  input_variables?: Record<string, string>;
  components?: PromptComponent[] | null;
  output_schema?: Record<string, unknown> | null;
  disable_output_schema?: boolean;
  tools?: PromptTool[] | null;
  model?: string;
  provider?: string;
  temperature?: number;
  max_tokens?: number | null;
  tags?: string[] | null;
  notes?: string | null;
}

export interface PlaygroundRunResponse {
  run: PlaygroundRun;
  message: string;
}

export interface PlaygroundAnalysisItem {
  id: string;
  group_id: string;
  label: string;
  output: string;
}

export interface PlaygroundAnalysisRequest {
  items: PlaygroundAnalysisItem[];
  embedding_model?: string;
}

export interface PlaygroundAnalysisPoint {
  id: string;
  group_id: string;
  label: string;
  x: number;
  y: number;
  average_similarity: number;
}

export interface PlaygroundAnalysisResponse {
  points: PlaygroundAnalysisPoint[];
}

export interface PlaygroundAnalysisGroup {
  id: string;
  label: string;
  tone: 'primary' | 'compare';
  promptId: string | null;
  versionId: string | null;
  results: Array<PlaygroundRun | null>;
  runErrors: Array<string | null>;
}
