import { describe, expect, it } from 'vitest';
import { createSchemaProperty, toJsonSchema } from './SchemaBuilder';

describe('SchemaBuilder', () => {
  it('creates new schema properties as required', () => {
    expect(createSchemaProperty().required).toBe(true);
  });

  it('marks every schema property as required in generated json schema', () => {
    expect(toJsonSchema([
      {
        id: 'one',
        name: 'tool_used',
        type: 'boolean',
        description: 'whether the tool was used',
        required: true,
      },
      {
        id: 'two',
        name: 'response',
        type: 'string',
        description: '',
        required: false,
      },
    ])).toEqual({
      type: 'object',
      additionalProperties: false,
      properties: {
        tool_used: {
          type: 'boolean',
          description: 'whether the tool was used',
        },
        response: {
          type: 'string',
        },
      },
      required: ['tool_used', 'response'],
    });
  });
});
