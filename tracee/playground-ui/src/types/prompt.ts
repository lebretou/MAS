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
  | "external_information"
  | "custom";

export type PromptMessageRole = "system" | "human" | "ai";

export interface PromptComponent {
  component_id?: string | null;
  type: PromptComponentType;
  name?: string | null;
  message_role?: PromptMessageRole | null;
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
  argument_id?: string | null;
  name: string;
  description?: string | null;
  type: ToolArgumentType;
  required: boolean;
  allowed_values?: string[] | null;
}

export interface PromptTool {
  tool_id?: string | null;
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
  parent_version_id?: string | null;
  root_version_id?: string | null;
  branch_id?: string | null;
  branch_name?: string | null;
  revision_note?: string | null;
  source_template_id?: string | null;
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

export interface PromptTemplateField {
  field_id: string;
  label: string;
  description?: string | null;
  input_type: string;
  required: boolean;
  placeholder?: string | null;
  default_value?: string;
}

export interface PromptTemplate {
  template_id: string;
  name: string;
  description?: string | null;
  archetype?: string | null;
  fields: PromptTemplateField[];
  components: PromptComponent[];
  suggested_tools?: PromptTool[];
  suggested_output_schema?: Record<string, unknown> | null;
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
  parent_version_id?: string | null;
  branch_name?: string | null;
  revision_note?: string | null;
  source_template_id?: string | null;
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
