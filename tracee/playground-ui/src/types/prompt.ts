export type PromptComponentType = 
  | "role"
  | "goal"
  | "constraints"
  | "io_rules"
  | "examples"
  | "safety"
  | "tool_instructions";

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
  output_schema: Record<string, unknown> | null;
  created_at: string;
}

export interface PromptListItem {
  prompt_id: string;
  name: string;
  description: string | null;
  latest_version_id: string | null;
  version_count: number;
  created_at: string;
  updated_at: string;
}

export interface PromptWithVersions {
  prompt: Prompt;
  versions: PromptVersion[];
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

export type SchemaPropertyType = 'string' | 'number' | 'integer' | 'boolean' | 'null' | 'array';

export type SchemaArrayItemType = 'string' | 'number' | 'integer' | 'boolean';

export interface SchemaProperty {
  id: string;
  name: string;
  type: SchemaPropertyType;
  description: string;
  required: boolean;
  items?: SchemaArrayItemType;
}