import { describe, expect, it } from 'vitest';
import type { PlaygroundAnalysisGroup, PlaygroundRun } from '../types/playground';
import {
  analyzeRunsWithReference,
  buildAnchorSimilarityMap,
  buildVisualizationEntries,
  buildVisualizationSimilarity,
  collectFieldOptions,
  collectFieldValues,
} from './useRunAnalysis';

function makeRun(index: number, output: string, outputSchema: Record<string, unknown> | null = null): PlaygroundRun {
  return {
    run_id: `run-${index}`,
    created_at: '2026-03-18T00:00:00.000Z',
    prompt_id: 'prompt-1',
    version_id: 'version-1',
    model: 'gpt-4o',
    provider: 'openai',
    temperature: 0,
    max_tokens: null,
    input_variables: {},
    resolved_prompt: 'prompt',
    output_schema: outputSchema,
    tools: null,
    tool_calls: null,
    output,
    output_schema_used: false,
    latency_ms: 120,
    prompt_tokens: 20,
    completion_tokens: 40,
    total_tokens: 60,
    model_config_id: null,
    created_by: null,
    tags: null,
    notes: null,
  };
}

function makeGroup(
  id: string,
  label: string,
  results: Array<PlaygroundRun | null>,
  runErrors: Array<string | null>,
  tone: 'primary' | 'compare' = 'primary',
): PlaygroundAnalysisGroup {
  return {
    id,
    label,
    tone,
    promptId: null,
    versionId: null,
    results,
    runErrors,
  };
}

describe('analyzeRunsWithReference', () => {
  it('marks missing runs as failed', () => {
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [makeRun(0, JSON.stringify({ answer: 'ok' })), null], [null, 'request failed'])],
    );

    expect(analysis[1]).toMatchObject({
      state: 'failed',
      error: 'request failed',
    });
  });

  it('marks non-json outputs without computing deviations', () => {
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [makeRun(0, 'plain text output')], [null])],
    );

    expect(analysis[0]).toMatchObject({
      state: 'non_json',
      parseFailed: true,
      validationErrors: [],
    });
  });

  it('keeps schema validation errors as explicit run state', () => {
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [
        makeRun(0, JSON.stringify({ answer: 42 }), {
          type: 'object',
          properties: {
            answer: { type: 'string' },
          },
          required: ['answer'],
        }),
      ], [null])],
    );

    expect(analysis[0].state).toBe('schema_invalid');
    expect(analysis[0].validationErrors).toHaveLength(1);
  });

  it('skips schema validation when tools are enabled for the run', () => {
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [
        {
          ...makeRun(0, JSON.stringify({ answer: 42 }), {
            type: 'object',
            properties: {
              answer: { type: 'string' },
            },
            required: ['answer'],
          }),
          tools: [
            {
              name: 'lookup_dataset',
              description: 'inspect the dataset',
              arguments: [],
            },
          ],
        },
      ], [null])],
    );

    expect(analysis[0].state).toBe('ready');
    expect(analysis[0].validationErrors).toEqual([]);
  });

  it('reuses the existing run point when a matching run is promoted as anchor', () => {
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [
        makeRun(0, JSON.stringify({ answer: 'ok' })),
        makeRun(1, JSON.stringify({ answer: 'better' })),
      ], [null, null])],
    );

    const entries = buildVisualizationEntries(analysis, {
      output: JSON.stringify({ answer: 'better' }),
      label: 'Anchor from run 2',
      source: 'run',
      runIndex: 1,
    });

    expect(entries).toHaveLength(2);
    expect(entries.filter((entry) => entry.kind === 'anchor')).toHaveLength(0);
  });

  it('keeps anchor similarity metrics when a promoted run reuses its point', () => {
    const anchor = {
      output: JSON.stringify({ answer: 'better' }),
      label: 'Anchor from run 2',
      source: 'run' as const,
      runIndex: 1,
    };
    const analysis = analyzeRunsWithReference(
      [makeGroup('primary', 'Current prompt', [
        makeRun(0, JSON.stringify({ answer: 'ok' })),
        makeRun(1, JSON.stringify({ answer: 'better' })),
      ], [null, null])],
    );

    const similarity = buildVisualizationSimilarity(analysis, anchor);
    const anchorSimilarityMap = buildAnchorSimilarityMap(similarity, anchor);

    expect(anchorSimilarityMap.get('primary:1')).toBe(1);
    expect(anchorSimilarityMap.get('primary:0')).toBeLessThan(1);
  });

  it('collects comparable field options from parsed outputs', () => {
    const analysis = analyzeRunsWithReference([
      makeGroup('primary', 'Current prompt', [
        makeRun(0, JSON.stringify({
          title: 'alpha',
          score: 4,
          approved: true,
          tags: ['a', 'b'],
        })),
      ], [null]),
      makeGroup('compare', 'Previous version', [
        makeRun(1, JSON.stringify({
          title: 'beta',
          score: 6,
          approved: false,
          tags: ['b'],
        })),
      ], [null], 'compare'),
    ]);

    const fields = collectFieldOptions(analysis);

    expect(fields).toEqual([
      { path: 'approved', label: 'approved', type: 'boolean', arrayItemType: undefined },
      { path: 'score', label: 'score', type: 'number', arrayItemType: undefined },
      { path: 'tags', label: 'tags', type: 'array', arrayItemType: 'string' },
      { path: 'title', label: 'title', type: 'string', arrayItemType: undefined },
    ]);
  });

  it('extracts field values across multiple groups', () => {
    const analysis = analyzeRunsWithReference([
      makeGroup('primary', 'Current prompt', [makeRun(0, JSON.stringify({ score: 4 }))], [null]),
      makeGroup('compare', 'Previous version', [makeRun(1, JSON.stringify({ score: 6 }))], [null], 'compare'),
    ]).map((run) => ({
      ...run,
      anchorSimilarity: null,
    }));

    const values = collectFieldValues(analysis, 'score');

    expect(values).toHaveLength(2);
    expect(values.map((entry) => entry.value)).toEqual([4, 6]);
    expect(values.map((entry) => entry.groupId)).toEqual(['primary', 'compare']);
  });

  it('only keeps fields that are present across all comparison groups', () => {
    const analysis = analyzeRunsWithReference([
      makeGroup('primary', 'Current prompt', [
        makeRun(0, JSON.stringify({ shared: 4, primary_only: true })),
      ], [null]),
      makeGroup('compare', 'Previous version', [
        makeRun(1, JSON.stringify({ shared: 6 })),
      ], [null], 'compare'),
    ]);

    const fields = collectFieldOptions(analysis);

    expect(fields).toEqual([
      { path: 'shared', label: 'shared', type: 'number', arrayItemType: undefined },
    ]);
  });
});
