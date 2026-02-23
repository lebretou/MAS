import type { JsonSchema } from "../../../types/schema";

interface Props {
  schema: JsonSchema;
}

interface FlatRow {
  path: string;
  type: string;
  description?: string;
  required: boolean;
  enumValues?: (string | number)[];
}

function flattenSchema(
  currentSchema: JsonSchema,
  prefix = "",
  isRequired = false,
  rows: FlatRow[] = [],
): FlatRow[] {
  const typeLabel = Array.isArray(currentSchema.type)
    ? currentSchema.type.join(" | ")
    : currentSchema.type;

  if (prefix || (!currentSchema.properties && typeLabel)) {
    rows.push({
      path: prefix || "(root)",
      type: typeLabel || "unknown",
      description: currentSchema.description,
      required: isRequired,
      enumValues: currentSchema.enum as (string | number)[] | undefined,
    });
  }

  if (currentSchema.properties) {
    const requiredSet = new Set(currentSchema.required ?? []);
    for (const [propName, propSchema] of Object.entries(currentSchema.properties)) {
      const newPrefix = prefix ? `${prefix}.${propName}` : propName;
      flattenSchema(propSchema, newPrefix, requiredSet.has(propName), rows);
    }
  }

  if (currentSchema.items) {
    if (Array.isArray(currentSchema.items)) {
      currentSchema.items.forEach((item, idx) => {
        flattenSchema(item, `${prefix}[${idx}]`, true, rows);
      });
    } else {
      flattenSchema(currentSchema.items, `${prefix}[]`, false, rows);
    }
  }

  return rows;
}

export function SchemaTable({ schema }: Props) {
  const flatRows = flattenSchema(schema);

  if (flatRows.length === 0) {
    return <p className="side-panel__empty">Empty schema.</p>;
  }

  return (
    <div className="schema-table-wrapper">
      <table className="schema-table">
        <thead>
          <tr>
            <th>Property</th>
            <th>Type</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {flatRows.map((row, idx) => (
            <tr key={idx}>
              <td className="schema-table__cell-property">
                <span className="schema-table__path">{row.path}</span>
                {row.required && <span className="schema-table__required-badge">required</span>}
              </td>
              <td className="schema-table__cell-type">
                <span className="schema-table__type">{row.type}</span>
                {row.enumValues && (
                  <div className="schema-table__enum">
                    {row.enumValues.map((val, i) => (
                      <span key={i} className="schema-table__enum-val">{String(val)}</span>
                    ))}
                  </div>
                )}
              </td>
              <td className="schema-table__cell-description">
                {row.description || <span className="schema-table__no-desc">&mdash;</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
