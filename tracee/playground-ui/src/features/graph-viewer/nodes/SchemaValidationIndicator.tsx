import type { JsonSchema } from "../../../types/schema";
import type { TraceEvent } from "../../../types/trace";
import {
  hasOutputSchemaProperties,
  resolveOutputValueForSchema,
  validateOutputAgainstSchema,
} from "../../../utils/schema-validation";

interface Props {
  outputSchema?: JsonSchema;
  outputValue?: unknown;
  events?: TraceEvent[];
}

export function SchemaValidationIndicator({ outputSchema, outputValue, events }: Props) {
  if (!outputSchema) {
    return null;
  }

  if (!hasOutputSchemaProperties(outputSchema)) {
    return null;
  }

  const resolvedOutputValue = resolveOutputValueForSchema(events, outputSchema, outputValue);
  if (resolvedOutputValue == null) {
    return null;
  }

  const results = validateOutputAgainstSchema(resolvedOutputValue, outputSchema);
  if (results.length === 0) {
    return null;
  }

  return (
    <>
      <div className="agent-node__components-header">
        <span className="agent-node__components-label">JSON VALIDATION</span>
      </div>
      <div className="agent-node__schema-validation">
        {results.map((result) => (
          <span
            key={result.key}
            className={`agent-node__schema-square agent-node__schema-square--${result.state}`}
            title={`${result.key}: ${result.state}`}
            aria-label={`${result.key}: ${result.state}`}
          />
        ))}
      </div>
    </>
  );
}
