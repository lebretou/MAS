import type { PromptComponent, PromptTool } from '../../types/prompt';
import { getPromptComponentDisplayName, normalizePromptComponents } from './promptEditor';

interface PromptVersionLike {
  components: PromptComponent[];
  tools?: PromptTool[] | null;
  outputSchema?: Record<string, unknown> | null;
  variables?: Record<string, string> | null;
}

export interface PromptVersionDiffSummary {
  added: string[];
  removed: string[];
  changed: string[];
  toolChanged: boolean;
  schemaChanged: boolean;
  variableChanged: boolean;
}

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(',')}]`;
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, nestedValue]) => `${JSON.stringify(key)}:${stableSerialize(nestedValue)}`);

    return `{${entries.join(',')}}`;
  }

  return JSON.stringify(value);
}

export function getPromptVersionDiffSummary(
  current: PromptVersionLike,
  parent: PromptVersionLike | null,
): PromptVersionDiffSummary {
  if (!parent) {
    return {
      added: [],
      removed: [],
      changed: [],
      toolChanged: false,
      schemaChanged: false,
      variableChanged: false,
    };
  }

  const normalizedCurrentComponents = normalizePromptComponents(current.components);
  const normalizedParentComponents = normalizePromptComponents(parent.components);
  const currentMap = new Map(
    normalizedCurrentComponents.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ]),
  );
  const parentMap = new Map(
    normalizedParentComponents.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ]),
  );

  const added: string[] = [];
  const removed: string[] = [];
  const changed: string[] = [];

  currentMap.forEach((component, key) => {
    const previous = parentMap.get(key);
    if (!previous) {
      added.push(getPromptComponentDisplayName(component));
      return;
    }

    if (
      previous.content !== component.content
      || previous.enabled !== component.enabled
      || previous.message_role !== component.message_role
      || getPromptComponentDisplayName(previous) !== getPromptComponentDisplayName(component)
    ) {
      changed.push(getPromptComponentDisplayName(component));
    }
  });

  parentMap.forEach((component, key) => {
    if (!currentMap.has(key)) {
      removed.push(getPromptComponentDisplayName(component));
    }
  });

  return {
    added,
    removed,
    changed,
    toolChanged: stableSerialize(current.tools ?? []) !== stableSerialize(parent.tools ?? []),
    schemaChanged: stableSerialize(current.outputSchema ?? null) !== stableSerialize(parent.outputSchema ?? null),
    variableChanged: stableSerialize(current.variables ?? {}) !== stableSerialize(parent.variables ?? {}),
  };
}
