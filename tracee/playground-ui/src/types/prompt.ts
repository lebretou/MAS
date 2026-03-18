export type PromptComponentType =
  | "role"
  | "goal"
  | "constraints"
  | "task"
  | "io_rules"
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

export type ToolArgumentType =
  | "string"
  | "number"
  | "integer"
  | "boolean"
  | "array"
  | "object";

export interface PromptToolArgument {
  name: string;
  description?: string | null;
  type: ToolArgumentType;
  required: boolean;
  allowed_values?: string[] | null;
}

export interface PromptTool {
  name: string;
  description: string;
  arguments: PromptToolArgument[];
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
  tools?: PromptTool[];
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
  tools?: PromptTool[];
}

export type SchemaPropertyType = "string" | "number" | "integer" | "boolean" | "null" | "array";

export type SchemaArrayItemType = "string" | "number" | "integer" | "boolean";

export interface SchemaProperty {
  id: string;
  name: string;
  type: SchemaPropertyType;
  description: string;
  required: boolean;
  items?: SchemaArrayItemType;
}
