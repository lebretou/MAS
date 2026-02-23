import { Panel } from "@xyflow/react";
import type { JsonSchema } from "../../../types/schema";
import dictIcon from "../../../assets/dict.svg";
import listIcon from "../../../assets/list.svg";
import numberIcon from "../../../assets/number.svg";
import stringIcon from "../../../assets/string.svg";

interface Props {
  schema: JsonSchema;
}

const TYPE_ICONS: Record<string, string> = {
  object: dictIcon,
  array: listIcon,
  number: numberIcon,
  string: stringIcon,
  boolean: stringIcon,
};

export function StateSchemaPanel({ schema }: Props) {
  const properties = schema.properties || {};

  return (
    <Panel position="top-right" className="state-schema-panel">
      <div className="state-schema-panel__header">
        <h3 className="state-schema-panel__title">State Schema</h3>
        {schema.description && (
          <p className="state-schema-panel__subtitle">{schema.description}</p>
        )}
      </div>
      <div className="state-schema-panel__content">
        {Object.entries(properties).map(([key, propSchema]) => {
          const typeStr = Array.isArray(propSchema.type)
            ? propSchema.type[0]
            : propSchema.type || "unknown";

          const displayTypeStr = Array.isArray(propSchema.type)
            ? propSchema.type.join(" | ")
            : propSchema.type || "unknown";

          const iconSrc = TYPE_ICONS[typeStr] || stringIcon;

          return (
            <div key={key} className="state-schema-panel__row">
              <span className="state-schema-panel__key">{key}</span>
              <div className="state-schema-panel__type">
                <img src={iconSrc} alt={typeStr} className="state-schema-panel__type-icon" />
                <span>{displayTypeStr}</span>
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
