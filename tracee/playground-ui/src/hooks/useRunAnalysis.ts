import Ajv from 'ajv';
import { useMemo } from 'react';
import type { PlaygroundAnalysisGroup, PlaygroundRun } from '../types/playground';
import { computeSimilarity } from '../utils/cosineSimilarity';
import type { SimilarityResult } from '../utils/cosineSimilarity';

export interface AnchorPoint {
  output: string;
  label: string;
  source: 'example' | 'run';
  runIndex: number | null;
  selectionId: string | null;
}

export interface ComparisonReference {
  kind: 'anchor';
  label: string;
  output: string;
  runIndex: number | null;
  anchorSelectionId: string | null;
}

export type RunState = 'ready' | 'failed' | 'non_json' | 'schema_invalid';
export type FieldType = 'string' | 'number' | 'boolean' | 'array';
export type ArrayItemType = 'string' | 'number' | 'boolean' | 'mixed' | 'unknown';

export interface AnalyzedRun {
  selectionId: string;
  globalIndex: number;
  index: number;
  groupId: string;
  groupLabel: string;
  groupVersionId: string | null;
  groupTone: 'primary' | 'compare';
  run: PlaygroundRun | null;
  error: string | null;
  parsed: unknown | null;
  parseFailed: boolean;
  state: RunState;
  validationErrors: Array<{ message?: string; instancePath?: string }>;
  anchorSimilarity: number | null;
}

export interface ScatterPoint {
  id: string;
  x: number;
  y: number;
  selectionId: string | null;
  similarity: number;
  groupId: string;
  groupLabel: string;
  groupVersionId: string | null;
  groupTone: 'primary' | 'compare';
  isAnchor: boolean;
  isFailed: boolean;
  label: string;
}

export interface FieldOption {
  path: string;
  label: string;
  type: FieldType;
  arrayItemType?: ArrayItemType;
}

export interface FieldValueEntry {
  selectionId: string;
  groupId: string;
  groupLabel: string;
  groupVersionId: string | null;
  groupTone: 'primary' | 'compare';
  runIndex: number;
  label: string;
  value: unknown;
}

export interface ProjectionItem {
  id: string;
  kind: 'run' | 'anchor';
  output: string;
  selectionId: string | null;
  groupId: string;
  groupLabel: string;
  groupVersionId: string | null;
  groupTone: 'primary' | 'compare';
  label: string;
  isFailed: boolean;
}

interface VisualizationSimilarity extends SimilarityResult {
  entries: ProjectionItem[];
}

export interface RunAnalysis {
  analyzed: AnalyzedRun[];
  reference: ComparisonReference | null;
  projectionItems: ProjectionItem[];
  failureCount: number;
  runCount: number;
  fieldOptions: FieldOption[];
}

function getRunVersionLabel(
  groupVersionId: string | null | undefined,
  runVersionId: string | null | undefined,
  fallbackLabel: string,
) {
  return groupVersionId || runVersionId || fallbackLabel;
}

function parseJsonValue(value: string): { parsed: unknown | null; parseFailed: boolean } {
  const normalized = value.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return { parsed: JSON.parse(normalized), parseFailed: false };
  } catch {
    return { parsed: null, parseFailed: true };
  }
}

function getValueType(value: unknown): FieldType | 'object' | 'null' | 'unsupported' {
  if (value === null || value === undefined) {
    return 'null';
  }
  if (Array.isArray(value)) {
    return 'array';
  }
  if (typeof value === 'string') {
    return 'string';
  }
  if (typeof value === 'number') {
    return 'number';
  }
  if (typeof value === 'boolean') {
    return 'boolean';
  }
  if (typeof value === 'object') {
    return 'object';
  }
  return 'unsupported';
}

function getArrayItemType(value: unknown[]): ArrayItemType {
  const itemTypes = new Set<ArrayItemType>();
  value.forEach((item) => {
    const itemType = getValueType(item);
    if (itemType === 'string' || itemType === 'number' || itemType === 'boolean') {
      itemTypes.add(itemType);
      return;
    }
    if (itemType === 'array' || itemType === 'object') {
      itemTypes.add('mixed');
    }
  });

  if (itemTypes.size === 0) {
    return 'unknown';
  }
  if (itemTypes.size === 1) {
    return Array.from(itemTypes)[0] ?? 'unknown';
  }
  return 'mixed';
}

function readValueAtPath(value: unknown, path: string): unknown {
  if (!path) {
    return value;
  }

  return path.split('.').reduce<unknown>((current, segment) => {
    if (!current || typeof current !== 'object' || Array.isArray(current)) {
      return undefined;
    }

    return (current as Record<string, unknown>)[segment];
  }, value);
}

export function analyzeRunsWithReference(
  groups: PlaygroundAnalysisGroup[],
): Omit<AnalyzedRun, 'anchorSimilarity'>[] {
  const ajv = new Ajv();
  let globalIndex = 0;

  return groups.flatMap((group) => group.results.map((run, i) => {
    const error = group.runErrors[i];
    const nextGlobalIndex = globalIndex++;
    if (error || !run) {
      return {
        selectionId: `${group.id}:${i}`,
        globalIndex: nextGlobalIndex,
        index: i,
        groupId: group.id,
        groupLabel: group.label,
        groupVersionId: group.versionId,
        groupTone: group.tone,
        run,
        error,
        parsed: null,
        parseFailed: false,
        state: 'failed' as const,
        validationErrors: [],
      };
    }

    const { parsed, parseFailed } = parseJsonValue(run.output);
    let validationErrors: Array<{ message?: string; instancePath?: string }> = [];
    const schemaValidationDisabled = Boolean(run.tools?.length || run.tool_calls?.length);

    if (!schemaValidationDisabled && run.output_schema && parseFailed) {
      validationErrors = [{ message: 'Output was not valid JSON for the configured schema.' }];
    } else if (!schemaValidationDisabled && parsed !== null && run.output_schema) {
      try {
        const validate = ajv.compile(run.output_schema as object);
        validate(parsed);
        validationErrors = validate.errors ?? [];
      } catch {
        validationErrors = [{ message: 'Configured output schema could not be evaluated.' }];
      }
    }

    return {
      selectionId: `${group.id}:${i}`,
      globalIndex: nextGlobalIndex,
      index: i,
      groupId: group.id,
      groupLabel: group.label,
      groupVersionId: group.versionId,
      groupTone: group.tone,
      run,
      error,
      parsed,
      parseFailed,
      state: parseFailed
        ? 'non_json'
        : validationErrors.length > 0
          ? 'schema_invalid'
          : 'ready',
      validationErrors,
    };
  }));
}

export function buildVisualizationEntries(
  analyzed: Array<Omit<AnalyzedRun, 'anchorSimilarity'>>,
  anchor: AnchorPoint | null,
): ProjectionItem[] {
  const entries: ProjectionItem[] = analyzed
    .filter((run) => run.run && !run.error)
    .map((run) => ({
      id: run.selectionId,
      kind: 'run',
      output: run.run?.output ?? '',
      selectionId: run.selectionId,
      groupId: run.groupId,
      groupLabel: run.groupLabel,
      groupVersionId: run.groupVersionId ?? run.run?.version_id ?? null,
      groupTone: run.groupTone,
      label: `${getRunVersionLabel(run.groupVersionId, run.run?.version_id, run.groupLabel)} · Run ${run.index + 1}`,
      isFailed: run.state === 'failed',
    }));

  const anchoredRun = anchor?.selectionId
    ? analyzed.find((run) => run.selectionId === anchor.selectionId)
    : anchor?.source === 'run' && anchor.runIndex !== null
      ? analyzed.find((run) => run.index === anchor.runIndex && run.groupTone === 'primary')
      : null;
  const shouldReuseRunPoint = anchoredRun?.run?.output === anchor?.output;

  if (anchor?.output.trim() && !shouldReuseRunPoint) {
    entries.push({
      id: 'anchor',
      kind: 'anchor',
      output: anchor.output,
      selectionId: null,
      groupId: 'anchor',
      groupLabel: 'Anchor',
      groupVersionId: null,
      groupTone: 'compare',
      label: anchor.label,
      isFailed: false,
    });
  }

  return entries;
}

export function buildVisualizationSimilarity(
  analyzed: Array<Omit<AnalyzedRun, 'anchorSimilarity'>>,
  anchor: AnchorPoint | null,
): VisualizationSimilarity | null {
  const entries = buildVisualizationEntries(analyzed, anchor);

  if (entries.length < 2) {
    return null;
  }

  const similarity = computeSimilarity(entries.map((entry) => entry.output));
  return { ...similarity, entries };
}

export function buildAnchorSimilarityMap(
  similarity: VisualizationSimilarity | null,
  anchor: AnchorPoint | null,
): Map<string, number> {
  const anchorIndex = similarity?.entries.findIndex((entry) => {
    if (entry.kind === 'anchor') {
      return true;
    }

    if (anchor?.selectionId && entry.selectionId === anchor.selectionId) {
      return true;
    }

    return anchor?.source === 'run'
      && anchor.runIndex !== null
      && entry.selectionId?.endsWith(`:${anchor.runIndex}`)
      && anchor.output === entry.output;
  }) ?? -1;

  if (!similarity || anchorIndex < 0) {
    return new Map();
  }

  const similarities = new Map<string, number>();
  similarity.entries.forEach((entry, entryIndex) => {
    if (entry.kind === 'run' && entry.selectionId) {
      similarities.set(entry.selectionId, similarity.matrix[entryIndex]?.[anchorIndex] ?? 0);
    }
  });

  return similarities;
}

export function collectFieldOptions(analyzed: Array<Omit<AnalyzedRun, 'anchorSimilarity'>>): FieldOption[] {
  const fieldMap = new Map<string, {
    type: FieldType;
    arrayItemTypes: Set<ArrayItemType>;
    groups: Set<string>;
  }>();
  const totalGroups = new Set(analyzed.map((run) => run.groupId)).size;

  const visit = (value: unknown, path: string, groupId: string) => {
    const valueType = getValueType(value);
    if (valueType === 'object') {
      Object.entries(value as Record<string, unknown>).forEach(([key, nestedValue]) => {
        visit(nestedValue, path ? `${path}.${key}` : key, groupId);
      });
      return;
    }

    if (!path || valueType === 'null' || valueType === 'unsupported') {
      return;
    }

    const current = fieldMap.get(path);
    if (!current) {
      fieldMap.set(path, {
        type: valueType,
        arrayItemTypes: valueType === 'array'
          ? new Set<ArrayItemType>([getArrayItemType(value as unknown[])])
          : new Set<ArrayItemType>(),
        groups: new Set([groupId]),
      });
      return;
    }

    if (current.type !== valueType) {
      fieldMap.delete(path);
      return;
    }

    current.groups.add(groupId);
    if (valueType === 'array') {
      current.arrayItemTypes.add(getArrayItemType(value as unknown[]));
    }
  };

  analyzed.forEach((run) => {
    if (run.parsed && typeof run.parsed === 'object' && !Array.isArray(run.parsed)) {
      visit(run.parsed, '', run.groupId);
    }
  });

  return Array.from(fieldMap.entries())
    .filter(([, config]) => config.groups.size === totalGroups)
    .map(([path, config]) => ({
      path,
      label: path,
      type: config.type,
      arrayItemType: config.type === 'array'
        ? (config.arrayItemTypes.size === 1 ? Array.from(config.arrayItemTypes)[0] : 'mixed')
        : undefined,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

export function collectFieldValues(
  analyzed: AnalyzedRun[],
  path: string,
): FieldValueEntry[] {
  return analyzed.flatMap((run) => {
    if (run.parsed === null) {
      return [];
    }

    const value = readValueAtPath(run.parsed, path);
    if (value === undefined) {
      return [];
    }

    return [{
      selectionId: run.selectionId,
      groupId: run.groupId,
      groupLabel: run.groupLabel,
      groupVersionId: run.groupVersionId ?? run.run?.version_id ?? null,
      groupTone: run.groupTone,
      runIndex: run.index,
      label: `${getRunVersionLabel(run.groupVersionId, run.run?.version_id, run.groupLabel)} · Run ${run.index + 1}`,
      value,
    }];
  });
}

export function useRunAnalysis(
  groups: PlaygroundAnalysisGroup[],
  anchor: AnchorPoint | null,
): RunAnalysis {
  const baseAnalyzed = useMemo(
    () => analyzeRunsWithReference(groups),
    [groups],
  );

  const visualizationSimilarity = useMemo(
    () => buildVisualizationSimilarity(baseAnalyzed, anchor),
    [baseAnalyzed, anchor],
  );

  const anchorSimilarityMap = useMemo(
    () => buildAnchorSimilarityMap(visualizationSimilarity, anchor),
    [visualizationSimilarity, anchor],
  );

  const analyzed = useMemo(
    () => baseAnalyzed.map((run) => ({
      ...run,
      anchorSimilarity: anchorSimilarityMap.get(run.selectionId) ?? null,
    })),
    [baseAnalyzed, anchorSimilarityMap],
  );

  const projectionItems = useMemo(
    () => buildVisualizationEntries(baseAnalyzed, anchor),
    [baseAnalyzed, anchor],
  );

  const reference = useMemo(() => {
    if (!anchor?.output.trim()) {
      return null;
    }

    return {
      kind: 'anchor' as const,
      label: anchor.label,
      output: anchor.output,
      runIndex: anchor.runIndex,
      anchorSelectionId: anchor.selectionId ?? null,
    };
  }, [anchor]);

  const fieldOptions = useMemo(
    () => collectFieldOptions(baseAnalyzed),
    [baseAnalyzed],
  );

  const failureCount = useMemo(
    () => analyzed.filter((run) => run.state === 'failed').length,
    [analyzed],
  );

  const runCount = useMemo(
    () => analyzed.filter((run) => run.run && !run.error).length,
    [analyzed],
  );

  return {
    analyzed,
    reference,
    projectionItems,
    failureCount,
    runCount,
    fieldOptions,
  };
}
