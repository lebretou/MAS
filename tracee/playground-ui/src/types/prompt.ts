export type PromptComponentType =
  | "role"
  | "constraints"
  | "task"
  | "inputs"
  | "outputs"
  | "examples"
  | "safety"
  | "tool_instructions"
  | "external_information";

export interface PromptComponent {
  type: PromptComponentType;
  content: string;
  enabled: boolean;
}

export interface Prompt {
  prompt_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  latest_version_id?: string | null;
}

export interface PromptVersion {
  prompt_id: string;
  version_id: string;
  name: string;
  components: PromptComponent[];
  variables?: Record<string, string> | null;
  output_schema?: Record<string, unknown> | null;
  created_at: string;
}

export interface CreatePromptRequest {
  prompt_id: string;
  name: string;
  description?: string | null;
}

export interface CreateVersionRequest {
  name: string;
  components: PromptComponent[];
  variables?: Record<string, string> | null;
  output_schema?: Record<string, unknown> | null;
}
