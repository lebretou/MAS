import React from 'react';
import type { SchemaArrayItemType, SchemaProperty, SchemaPropertyType } from '../../../types/prompt';

interface Props {
  properties: SchemaProperty[];
  onChange: (properties: SchemaProperty[]) => void;
}

const TYPE_OPTIONS: SchemaPropertyType[] = ['string', 'number', 'integer', 'boolean', 'null', 'array'];
const ARRAY_ITEM_OPTIONS: SchemaArrayItemType[] = ['string', 'number', 'integer', 'boolean'];

export function createSchemaProperty(): SchemaProperty {
  const randomId =
    globalThis.crypto?.randomUUID?.() ??
    `schema-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  return {
    id: randomId,
    name: '',
    type: 'string',
    description: '',
    required: false,
  };
}

export function getSchemaValidationError(props: SchemaProperty[]): string | null {
  if (props.length === 0) {
    return 'Add at least one output field.';
  }

  const seen = new Set<string>();

  for (const prop of props) {
    const name = prop.name.trim();

    if (!name) {
      return 'Each output field needs a name.';
    }

    if (seen.has(name)) {
      return 'Output field names must be unique.';
    }

    seen.add(name);
  }

  return null;
}

export function toJsonSchema(props: SchemaProperty[]): Record<string, unknown> {
  const validProps = props.map(prop => ({
    ...prop,
    name: prop.name.trim(),
  }));

  return {
    type: 'object',
    additionalProperties: false,
    properties: Object.fromEntries(
      validProps.map(p => [
        p.name.trim(),
        {
          type: p.type,
          ...(p.description ? { description: p.description } : {}),
          ...(p.type === 'array' ? { items: { type: p.items ?? 'string' } } : {}),
        },
      ])
    ),
    required: validProps.filter(p => p.required).map(p => p.name.trim()),
  };
}

const SchemaBuilder: React.FC<Props> = ({ properties, onChange }) => {
  const addProperty = () => {
    onChange([
      ...properties,
      createSchemaProperty(),
    ]);
  };

  const removeProperty = (index: number) => {
    onChange(properties.filter((_, i) => i !== index));
  };

  const updateProperty = (index: number, patch: Partial<SchemaProperty>) => {
    onChange(properties.map((p, i) => (i === index ? { ...p, ...patch } : p)));
  };

  return (
    <div className="schema-builder">
      {properties.length > 0 && (
        <div className="card schema-builder__card">
          <div className="card__body schema-builder__body">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Type</th>
                  <th>Description</th>
                  <th>Required</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {properties.map((prop, i) => (
                  <tr key={prop.id}>
                    <td>
                      <input
                        className="input"
                        value={prop.name}
                        onChange={e => updateProperty(i, { name: e.target.value })}
                        placeholder="property_name"
                      />
                    </td>
                    <td>
                      <select
                        className="select"
                        value={prop.type}
                        onChange={e => updateProperty(i, {
                          type: e.target.value as SchemaPropertyType,
                          ...(e.target.value === 'array' ? { items: prop.items ?? 'string' } : { items: undefined }),
                        })}
                      >
                        {TYPE_OPTIONS.map(t => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                      {prop.type === 'array' && (
                        <select
                          className="select schema-builder__items-select"
                          value={prop.items ?? 'string'}
                          onChange={e => updateProperty(i, { items: e.target.value as SchemaArrayItemType })}
                        >
                          {ARRAY_ITEM_OPTIONS.map(t => (
                            <option key={t} value={t}>items: {t}</option>
                          ))}
                        </select>
                      )}
                    </td>
                    <td>
                      <input
                        className="input"
                        value={prop.description}
                        onChange={e => updateProperty(i, { description: e.target.value })}
                        placeholder="Optional description"
                      />
                    </td>
                    <td className="schema-builder__center">
                      <input
                        type="checkbox"
                        checked={prop.required}
                        onChange={e => updateProperty(i, { required: e.target.checked })}
                        className="schema-builder__checkbox"
                      />
                    </td>
                    <td>
                      <button
                        type="button"
                        className="icon-btn icon-btn--close"
                        onClick={() => removeProperty(i)}
                      >
                        &times;
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <button
        type="button"
        className="btn btn--secondary btn--sm"
        onClick={addProperty}
      >
        + Add property
      </button>
    </div>
  );
};

export default SchemaBuilder;
