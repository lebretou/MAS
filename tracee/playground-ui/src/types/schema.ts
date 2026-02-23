export interface JsonSchema extends Record<string, unknown> {
  type?: string | string[];
  description?: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema | JsonSchema[];
  required?: string[];
  additionalProperties?: boolean | JsonSchema;
  enum?: Array<string | number | boolean | null>;
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
}
