import { describe, expect, it } from 'vitest';
import type { PlaygroundRun } from '../types/playground';
import {
  analyzeRunsWithReference,
  buildAnchorSimilarityMap,
  buildVisualizationEntries,
  buildVisualizationSimilarity,
} from './useRunAnalysis';

function makeRun(index: number, output: string): PlaygroundRun {
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
    output_schema: null,
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

describe('analyzeRunsWithReference', () => {
  it('uses the anchor schema when a json anchor is provided', () => {
    const results = [
      makeRun(0, JSON.stringify({ answer: 'ok', score: 0.9 })),
      makeRun(1, JSON.stringify({ answer: 'ok' })),
    ];

    const analysis = analyzeRunsWithReference(results, [null, null], {
      output: JSON.stringify({ answer: 'ok', score: 0.9, reason: 'expected' }),
      label: 'Example anchor',
      source: 'example',
      runIndex: null,
    });

    expect(analysis.referenceSchema?.fields.map(field => field.path)).toEqual(['answer', 'score', 'reason']);
    expect(analysis.referenceSchemaKind).toBe('anchor');
    expect(analysis.analyzed[0].deviations).toEqual([
      { path: 'reason', type: 'missing', expected: 'string' },
    ]);
    expect(analysis.analyzed[1].deviations).toEqual([
      { path: 'score', type: 'missing', expected: 'number' },
      { path: 'reason', type: 'missing', expected: 'string' },
    ]);
  });

  it('falls back to consensus schema when the anchor is not json', () => {
    const results = [
      makeRun(0, JSON.stringify({ answer: 'ok', score: 0.9 })),
      makeRun(1, JSON.stringify({ answer: 'ok' })),
    ];

    const analysis = analyzeRunsWithReference(results, [null, null], {
      output: 'plain text anchor',
      label: 'Example anchor',
      source: 'example',
      runIndex: null,
    });

    expect(analysis.consensus?.fields.map(field => field.path)).toEqual(['answer', 'score']);
    expect(analysis.referenceSchema).toEqual(analysis.consensus);
    expect(analysis.referenceSchemaKind).toBe('consensus');
  });

  it('keeps failures classified when a run is missing', () => {
    const results = [
      makeRun(0, JSON.stringify({ answer: 'ok' })),
      null,
    ];

    const analysis = analyzeRunsWithReference(results, [null, 'request failed'], null);

    expect(analysis.analyzed[1]).toMatchObject({
      classification: 'failure',
      error: 'request failed',
    });
  });

  it('reuses the existing run point when a matching run is promoted as anchor', () => {
    const analysis = analyzeRunsWithReference(
      [
        makeRun(0, JSON.stringify({ answer: 'ok' })),
        makeRun(1, JSON.stringify({ answer: 'better' })),
      ],
      [null, null],
      null,
    );

    const entries = buildVisualizationEntries(analysis.analyzed, {
      output: JSON.stringify({ answer: 'better' }),
      label: 'Anchor from run 2',
      source: 'run',
      runIndex: 1,
    });

    expect(entries).toHaveLength(2);
    expect(entries.filter(entry => entry.kind === 'anchor')).toHaveLength(0);
  });

  it('keeps anchor similarity metrics when a promoted run reuses its point', () => {
    const anchor = {
      output: JSON.stringify({ answer: 'better' }),
      label: 'Anchor from run 2',
      source: 'run' as const,
      runIndex: 1,
    };
    const analysis = analyzeRunsWithReference(
      [
        makeRun(0, JSON.stringify({ answer: 'ok' })),
        makeRun(1, JSON.stringify({ answer: 'better' })),
      ],
      [null, null],
      anchor,
    );

    const similarity = buildVisualizationSimilarity(analysis.analyzed, anchor);
    const anchorSimilarityMap = buildAnchorSimilarityMap(similarity, anchor);

    expect(anchorSimilarityMap.get(1)).toBe(1);
    expect(anchorSimilarityMap.get(0)).toBeLessThan(1);
  });
});
