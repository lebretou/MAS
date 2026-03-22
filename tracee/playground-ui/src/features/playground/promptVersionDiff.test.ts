import { describe, expect, it } from 'vitest';
import { getPromptVersionDiffSummary } from './promptVersionDiff';

describe('getPromptVersionDiffSummary', () => {
  it('tracks component name, role, tool, and schema changes', () => {
    const summary = getPromptVersionDiffSummary(
      {
        components: [
          {
            component_id: 'task-1',
            type: 'task',
            name: 'Acceptance criteria',
            message_role: 'ai',
            content: 'Return a checklist.',
            enabled: true,
          },
        ],
        tools: [{ name: 'search_docs', description: 'Search docs', arguments: [] }],
        outputSchema: {
          type: 'object',
          properties: {
            answer: { type: 'string' },
          },
        },
      },
      {
        components: [
          {
            component_id: 'task-1',
            type: 'task',
            name: 'Task',
            message_role: 'human',
            content: 'Return a checklist.',
            enabled: true,
          },
        ],
        tools: [],
        outputSchema: null,
      }
    );

    expect(summary.changed).toEqual(['Acceptance criteria']);
    expect(summary.toolChanged).toBe(true);
    expect(summary.schemaChanged).toBe(true);
  });
});
