import React from 'react';
import { SchemaProperty, SchemaPropertyType } from '../types/prompt';

interface Props {
  properties: SchemaProperty[];
  onChange: (properties: SchemaProperty[]) => void;
}

const TYPE_OPTIONS: SchemaPropertyType[] = ['string', 'number', 'integer', 'boolean', 'null'];

export function toJsonSchema(props: SchemaProperty[]): Record<string, unknown> {
  return {
    type: 'object',
    properties: Object.fromEntries(
      props.map(p => [
        p.name,
        {
          type: p.type,
          ...(p.description ? { description: p.description } : {}),
        },
      ])
    ),
    required: props.filter(p => p.required).map(p => p.name),
  };
}

const SchemaBuilder: React.FC<Props> = ({ properties, onChange }) => {
  const addProperty = () => {
    onChange([
      ...properties,
      { name: '', type: 'string', description: '', required: false },
    ]);
  };

  const removeProperty = (index: number) => {
    onChange(properties.filter((_, i) => i !== index));
  };

  const updateProperty = (index: number, patch: Partial<SchemaProperty>) => {
    onChange(properties.map((p, i) => (i === index ? { ...p, ...patch } : p)));
  };

  return (
    <div style={{ marginTop: '8px' }}>
      {properties.length > 0 && (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '8px' }}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Type</th>
              <th style={thStyle}>Description</th>
              <th style={thStyle}>Required</th>
              <th style={thStyle}></th>
            </tr>
          </thead>
          <tbody>
            {properties.map((prop, i) => (
              <tr key={i}>
                <td style={tdStyle}>
                  <input
                    value={prop.name}
                    onChange={e => updateProperty(i, { name: e.target.value })}
                    placeholder="property_name"
                    style={{ width: '100%' }}
                  />
                </td>
                <td style={tdStyle}>
                  <select
                    value={prop.type}
                    onChange={e => updateProperty(i, { type: e.target.value as SchemaPropertyType })}
                  >
                    {TYPE_OPTIONS.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </td>
                <td style={tdStyle}>
                  <input
                    value={prop.description}
                    onChange={e => updateProperty(i, { description: e.target.value })}
                    placeholder="Optional description"
                    style={{ width: '100%' }}
                  />
                </td>
                <td style={{ ...tdStyle, textAlign: 'center' }}>
                  <input
                    type="checkbox"
                    checked={prop.required}
                    onChange={e => updateProperty(i, { required: e.target.checked })}
                  />
                </td>
                <td style={tdStyle}>
                  <button type="button" onClick={() => removeProperty(i)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <button
        type="button"
        onClick={addProperty}
        style={{ fontSize: '0.85rem' }}
      >
        + Add property
      </button>
    </div>
  );
};

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '4px 8px',
  borderBottom: '1px solid #ccc',
  fontSize: '0.8rem',
  color: '#666',
};

const tdStyle: React.CSSProperties = {
  padding: '4px 8px',
  verticalAlign: 'middle',
};

export default SchemaBuilder;
