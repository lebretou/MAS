import { describe, expect, it } from 'vitest';

import type { PromptComponent } from '../types/prompt';
import { planRunExecution } from './useRunExecution';

const components: PromptComponent[] = [
  {
    type: 'role',
    content: 'You are a careful analyst.',
    enabled: true,
  },
];
const loadedSignature = '{"components":[{"content":"You are a careful analyst.","enabled":true,"type":"role"}],"outputSchema":null,"tools":[]}';

describe('planRunExecution', () => {
  it('reuses the saved version when the existing prompt matches the loaded signature', () => {
    const plan = planRunExecution({
      components,
      tools: [],
      inputVariables: {},
      model: 'gpt-4o',
      provider: 'openai',
      temperature: 0,
      numRuns: 1,
      outputSchema: null,
      promptContext: {
        promptId: 'test-prompt',
        promptName: 'Test prompt',
        versionId: 'v5',
        loadedSignature,
      },
    });

    expect(plan).toMatchObject({
      kind: 'existing_saved',
      promptId: 'test-prompt',
      versionId: 'v5',
    });
  });

  it('runs existing prompt edits as an inline draft instead of creating a saved version', () => {
    const plan = planRunExecution({
      components: [
        ...components,
        {
          type: 'task',
          content: 'review the latest brief',
          enabled: true,
        },
      ],
      tools: [
        {
          name: 'lookup_dataset',
          description: 'inspect the dataset',
          arguments: [],
        },
      ],
      inputVariables: { brief: 'release notes' },
      model: 'gpt-4o',
      provider: 'openai',
      temperature: 0,
      numRuns: 1,
      outputSchema: {
        type: 'object',
        properties: {
          answer: { type: 'string' },
        },
      },
      promptContext: {
        promptId: 'test-prompt',
        promptName: 'Test prompt',
        versionId: 'draft-v5',
        loadedSignature,
      },
    });

    expect(plan).toMatchObject({
      kind: 'existing_draft',
      promptId: 'test-prompt',
      versionId: 'draft-v5',
      requestOverrides: {
        components: [
          {
            type: 'role',
            content: 'You are a careful analyst.',
            enabled: true,
          },
          {
            type: 'task',
            content: 'review the latest brief',
            enabled: true,
          },
        ],
        tools: [
          {
            name: 'lookup_dataset',
            description: 'inspect the dataset',
            arguments: [],
          },
        ],
        output_schema: {
          type: 'object',
          properties: {
            answer: { type: 'string' },
          },
        },
        disable_output_schema: true,
      },
    });
  });
});
