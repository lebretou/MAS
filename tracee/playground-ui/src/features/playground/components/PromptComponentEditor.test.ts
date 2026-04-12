import { describe, expect, it } from 'vitest';
import type { PromptComponent } from '../../../types/prompt';
import { findPromptVariableMatch } from './PromptComponentEditor';

describe('findPromptVariableMatch', () => {
  it('returns the first matching variable occurrence and its range', () => {
    const components: PromptComponent[] = [
      {
        component_id: 'role-1',
        type: 'role',
        content: 'You are a helpful editor.',
        enabled: true,
      },
      {
        component_id: 'task-1',
        type: 'task',
        content: 'Summarize {{release_brief}} for {{audience}}.',
        enabled: true,
      },
    ];

    expect(findPromptVariableMatch(components, 'release_brief')).toEqual({
      componentKey: 'task-1',
      componentIndex: 1,
      start: 10,
      end: 27,
      token: '{{release_brief}}',
    });
  });

  it('returns null when the variable does not appear in any component', () => {
    const components: PromptComponent[] = [
      {
        component_id: 'task-1',
        type: 'task',
        content: 'Summarize the launch plan.',
        enabled: true,
      },
    ];

    expect(findPromptVariableMatch(components, 'release_brief')).toBeNull();
  });

  it('prefers enabled matches before disabled sections', () => {
    const components: PromptComponent[] = [
      {
        component_id: 'task-disabled',
        type: 'task',
        content: 'Use {{release_brief}} from the archive.',
        enabled: false,
      },
      {
        component_id: 'task-enabled',
        type: 'task',
        content: 'Use {{release_brief}} in the current draft.',
        enabled: true,
      },
    ];

    expect(findPromptVariableMatch(components, 'release_brief')).toMatchObject({
      componentKey: 'task-enabled',
      componentIndex: 1,
      token: '{{release_brief}}',
    });
  });
});
