import { describe, expect, it } from 'vitest';
import {
  getOutlineSchemaFields,
  getOutlineToolNames,
  getOutlineVariableStates,
} from './PromptStructureOutline';

describe('PromptStructureOutline helpers', () => {
  it('returns trimmed tool names only', () => {
    expect(
      getOutlineToolNames([
        { name: ' search_docs ', description: 'Search docs', arguments: [] },
        { name: '', description: 'Missing name', arguments: [] },
        { name: 'lookup_user', description: 'Lookup a user', arguments: [] },
      ])
    ).toEqual(['search_docs', 'lookup_user']);
  });

  it('summarizes schema fields with readable type labels', () => {
    expect(
      getOutlineSchemaFields({
        type: 'object',
        properties: {
          answer: { type: 'string' },
          confidence: { type: 'number' },
          tags: { type: 'array', items: { type: 'string' } },
          metadata: { type: 'object' },
        },
      }).map(({ name, typeLabel }) => ({ name, typeLabel }))
    ).toEqual([
      { name: 'answer', typeLabel: 'string' },
      { name: 'confidence', typeLabel: 'number' },
      { name: 'tags', typeLabel: 'array<string>' },
      { name: 'metadata', typeLabel: 'object' },
    ]);
  });

  it('returns no schema fields when schema properties are missing', () => {
    expect(getOutlineSchemaFields(null)).toEqual([]);
    expect(getOutlineSchemaFields({ type: 'object' })).toEqual([]);
  });

  it('tracks whether detected variables are filled', () => {
    expect(
      getOutlineVariableStates(['release_brief', 'region', 'empty_value'], {
        release_brief: 'hello',
        region: '  west  ',
        empty_value: '   ',
      })
    ).toEqual([
      { name: 'release_brief', isFilled: true },
      { name: 'region', isFilled: true },
      { name: 'empty_value', isFilled: false },
    ]);
  });
});
