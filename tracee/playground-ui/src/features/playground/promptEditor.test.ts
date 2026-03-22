import { describe, expect, it } from 'vitest';
import {
  getPromptComponentDisplayName,
  normalizePromptComponents,
  resolvePromptText,
  resolvePromptMessages,
  serializePromptMessages,
} from './promptEditor';

describe('promptEditor', () => {
  it('fills default message roles from semantic component types', () => {
    const components = normalizePromptComponents([
      { type: 'role', content: 'You are precise.', enabled: true },
      { type: 'task', content: 'Answer {{question}}.', enabled: true },
      { type: 'examples', content: 'Example output', enabled: true },
    ]);

    expect(components.map((component) => component.message_role)).toEqual([
      'system',
      'human',
      'ai',
    ]);
    expect(components.map((component) => component.name)).toEqual([
      'Role',
      'Task',
      'Examples',
    ]);
  });

  it('resolves enabled components, variables, and schema text', () => {
    const resolved = resolvePromptText(
      [
        { type: 'role', name: 'Role', content: 'You are precise.', enabled: true, message_role: 'system' },
        { type: 'task', name: 'Task', content: 'Answer {{question}}.', enabled: true, message_role: 'human' },
        { type: 'safety', content: 'Never leak secrets.', enabled: false, message_role: 'system' },
      ],
      { question: 'What changed?' },
      {
        type: 'object',
        properties: {
          answer: { type: 'string' },
        },
      }
    );

    expect(resolved).toContain('Role:\nYou are precise.\n\nTask:\nAnswer What changed?.');
    expect(resolved).toContain('Respond with a JSON object that conforms to the following JSON Schema:');
    expect(resolved).not.toContain('Never leak secrets.');
  });

  it('uses the editable component name when present', () => {
    expect(
      getPromptComponentDisplayName({
        type: 'custom',
        name: 'Acceptance criteria',
        content: '',
        enabled: true,
      })
    ).toBe('Acceptance criteria');
  });

  it('keeps unresolved variables visible when no input value is provided', () => {
    expect(
      resolvePromptText(
        [
          { type: 'task', name: 'Task', content: 'Answer {{question}}.', enabled: true, message_role: 'human' },
        ],
        {},
        null
      )
    ).toContain('Task:\nAnswer {{question}}.');
  });

  it('builds chat messages from component message roles', () => {
    const messages = resolvePromptMessages(
      [
        { type: 'role', name: 'Role', content: 'You are precise.', enabled: true, message_role: 'system' },
        { type: 'task', name: 'Task', content: 'Answer {{question}}.', enabled: true, message_role: 'human' },
        { type: 'examples', name: 'Examples', content: 'Sure, here is the answer.', enabled: true, message_role: 'ai' },
      ],
      { question: 'What changed?' }
    );

    expect(messages).toEqual([
      { role: 'system', content: 'Role:\nYou are precise.' },
      { role: 'user', content: 'Task:\nAnswer What changed?.' },
      { role: 'assistant', content: 'Examples:\nSure, here is the answer.' },
    ]);
    expect(serializePromptMessages(messages)).toContain('System:\nRole:\nYou are precise.');
  });
});
