import type {
  PromptComponent,
  PromptComponentType,
  PromptMessageRole,
} from '../../types/prompt';

export const PROVIDED_COMPONENT_TYPES: PromptComponentType[] = [
  'role',
  'goal',
  'task',
  'constraints',
  'io_rules',
  'inputs',
  'outputs',
  'examples',
  'safety',
  'tool_instructions',
  'external_information',
];

export const COMPONENT_LABELS: Record<PromptComponentType, string> = {
  role: 'Role',
  goal: 'Goal',
  task: 'Task',
  constraints: 'Constraints',
  io_rules: 'I/O Rules',
  inputs: 'Inputs',
  outputs: 'Outputs',
  examples: 'Examples',
  safety: 'Safety',
  tool_instructions: 'Tool Instructions',
  external_information: 'External Information',
  custom: 'Custom Section',
};

export const COMPONENT_PLACEHOLDERS: Record<PromptComponentType, string> = {
  role: 'You are a helpful assistant that...',
  goal: 'Your primary goal is to...',
  task: 'Your primary task is to...',
  constraints: 'You must not... You should always...',
  io_rules: 'Input format: ... Output format: ...',
  inputs: 'Available inputs, variables, or context...',
  outputs: 'Expected output format or requirements...',
  examples: 'Example 1:\nInput: ...\nOutput: ...',
  safety: 'Never reveal... Always verify...',
  tool_instructions: 'When using the search tool...',
  external_information: 'Relevant external facts or background context...',
  custom: 'Add the custom section content here...',
};

const SYSTEM_COMPONENT_TYPES: PromptComponentType[] = [
  'role',
  'goal',
  'constraints',
  'io_rules',
  'outputs',
  'safety',
  'tool_instructions',
  'custom',
];

const HUMAN_COMPONENT_TYPES: PromptComponentType[] = [
  'task',
  'inputs',
  'external_information',
];

export function getDefaultPromptMessageRole(type: PromptComponentType): PromptMessageRole {
  if (SYSTEM_COMPONENT_TYPES.includes(type)) {
    return 'system';
  }

  if (HUMAN_COMPONENT_TYPES.includes(type)) {
    return 'human';
  }

  return 'ai';
}

export function getPromptComponentLabel(type: PromptComponentType): string {
  return COMPONENT_LABELS[type];
}

export function getDefaultPromptComponentName(type: PromptComponentType): string {
  return getPromptComponentLabel(type);
}

export function getPromptComponentDisplayName(component: PromptComponent): string {
  return component.name?.trim() || getDefaultPromptComponentName(component.type);
}

export function createPromptComponentId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `component-${Math.random().toString(36).slice(2, 10)}`;
}

export function withDefaultMessageRole(component: PromptComponent): PromptComponent {
  return {
    ...component,
    name: getPromptComponentDisplayName(component),
    message_role: component.message_role ?? getDefaultPromptMessageRole(component.type),
  };
}

export function normalizePromptComponents(components: PromptComponent[]): PromptComponent[] {
  return components.map(withDefaultMessageRole);
}

export function getPromptComponentKey(component: PromptComponent, index: number): string {
  return component.component_id ?? `${component.type}:${index}`;
}

export function preparePromptComponentsForEditor(components: PromptComponent[]): PromptComponent[] {
  return normalizePromptComponents(components).map((component, index) => ({
    ...component,
    component_id: component.component_id ?? `${component.type}:${index}`,
  }));
}

export type PromptChatRole = 'system' | 'user' | 'assistant';

export interface PromptChatMessage {
  role: PromptChatRole;
  content: string;
}

function substitutePromptVariables(content: string, inputVars: Record<string, string>): string {
  return content.replace(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g, (match, variableName: string) => {
    return inputVars[variableName] ?? match;
  });
}

export function getChatRoleForPromptMessageRole(role: PromptMessageRole): PromptChatRole {
  if (role === 'human') {
    return 'user';
  }

  if (role === 'ai') {
    return 'assistant';
  }

  return 'system';
}

export function resolvePromptMessages(
  components: PromptComponent[],
  inputVars: Record<string, string>,
): PromptChatMessage[] {
  return normalizePromptComponents(components)
    .filter((component) => component.enabled)
    .map((component) => {
      const resolvedContent = substitutePromptVariables(component.content, inputVars);
      const header = `${getPromptComponentDisplayName(component)}:`;
      return {
        role: getChatRoleForPromptMessageRole(
          component.message_role ?? getDefaultPromptMessageRole(component.type)
        ),
        content: resolvedContent ? `${header}\n${resolvedContent}` : header,
      };
    });
}

export function serializePromptMessages(messages: PromptChatMessage[]): string {
  return messages
    .map((message) => `${message.role.charAt(0).toUpperCase()}${message.role.slice(1)}:\n${message.content}`)
    .join('\n\n');
}

export function resolvePromptText(
  components: PromptComponent[],
  inputVars: Record<string, string>,
  outputSchema: Record<string, unknown> | null,
  schemaMode: 'full' | 'hint' | 'none' = 'full',
): string {
  const text = normalizePromptComponents(components)
    .filter((component) => component.enabled)
    .map((component) => {
      const resolvedContent = substitutePromptVariables(component.content, inputVars);
      const header = `${getPromptComponentDisplayName(component)}:`;
      return resolvedContent ? `${header}\n${resolvedContent}` : header;
    })
    .join('\n\n');

  if (!outputSchema || schemaMode === 'none') {
    return text;
  }

  const schemaBlock = schemaMode === 'hint'
    ? 'Your response should be structured JSON matching the requested schema.'
    : [
        'Respond with a JSON object that conforms to the following JSON Schema:',
        '```json',
        JSON.stringify(outputSchema, null, 2),
        '```',
      ].join('\n');

  return text ? `${text}\n\n${schemaBlock}` : schemaBlock;
}
