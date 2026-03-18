import Ajv from 'ajv';
import { useMemo } from 'react';
import type { PlaygroundRun } from '../types/playground';
import {
  classifyRun,
  computeConsensusSchema,
  findDeviations,
} from '../utils/schemaAggregation';
import type { ConsensusSchema, FieldDeviation, RunClassification } from '../utils/schemaAggregation';
import { computeSimilarity } from '../utils/cosineSimilarity';
import type { SimilarityResult } from '../utils/cosineSimilarity';
import { findConsensusOutputIndex } from '../utils/jsonDiff';

export interface AnchorPoint {
  output: string;
  label: string;
  source: 'example' | 'run';
  runIndex: number | null;
}

export interface ComparisonReference {
  kind: 'anchor' | 'consensus';
  label: string;
  output: string;
  runIndex: number | null;
}

export interface AnalyzedRun {
  index: number;
  run: PlaygroundRun | null;
  error: string | null;
  parsed: unknown | null;
  parseFailed: boolean;
  classification: RunClassification;
  deviations: FieldDeviation[];
  validationErrors: Array<{ message?: string; instancePath?: string }>;
  anchorSimilarity: number | null;
}

export interface ScatterPoint {
  x: number;
  y: number;
  index: number | null;
  classification: RunClassification;
  similarity: number;
  isAnchor: boolean;
  label: string;
}

interface VisualizationEntry {
  kind: 'run' | 'anchor';
  output: string;
  runIndex: number | null;
  label: string;
}

interface VisualizationSimilarity extends SimilarityResult {
  entries: VisualizationEntry[];
}

export interface RunAnalysis {
  analyzed: AnalyzedRun[];
  consensus: ConsensusSchema | null;
  referenceSchema: ConsensusSchema | null;
  referenceSchemaKind: 'anchor' | 'consensus' | null;
  reference: ComparisonReference | null;
  similarity: SimilarityResult | null;
  scatterPoints: ScatterPoint[];
  consensusOutputIndex: number;
  counts: Record<RunClassification, number>;
}

function parseJsonValue(value: string): { parsed: unknown | null; parseFailed: boolean } {
  const normalized = value.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return { parsed: JSON.parse(normalized), parseFailed: false };
  } catch {
    return { parsed: null, parseFailed: true };
  }
}

function createAnchorSchema(anchor: AnchorPoint | null): ConsensusSchema | null {
  if (!anchor?.output.trim()) {
    return null;
  }

  const { parsed } = parseJsonValue(anchor.output);
  if (parsed === null || typeof parsed !== 'object') {
    return null;
  }

  return computeConsensusSchema([{ parsed, index: -1 }]);
}

export function analyzeRunsWithReference(
  results: Array<PlaygroundRun | null>,
  errors: Array<string | null>,
  anchor: AnchorPoint | null,
): {
  analyzed: Omit<AnalyzedRun, 'anchorSimilarity'>[];
  consensus: ConsensusSchema | null;
  referenceSchema: ConsensusSchema | null;
  referenceSchemaKind: 'anchor' | 'consensus' | null;
} {
  const ajv = new Ajv();

  const baseRuns: Array<Omit<AnalyzedRun, 'anchorSimilarity'>> = results.map((run, i) => {
    const error = errors[i];
    if (error || !run) {
      return {
        index: i,
        run,
        error,
        parsed: null,
        parseFailed: false,
        classification: 'failure',
        deviations: [],
        validationErrors: [],
      };
    }

    const { parsed, parseFailed } = parseJsonValue(run.output);
    let validationErrors: Array<{ message?: string; instancePath?: string }> = [];

    if (run.output_schema && parseFailed) {
      validationErrors = [{ message: 'Output was not valid JSON for the configured schema.' }];
    } else if (parsed !== null && run.output_schema) {
      try {
        const validate = ajv.compile(run.output_schema as object);
        validate(parsed);
        validationErrors = validate.errors ?? [];
      } catch {
        validationErrors = [];
      }
    }

    return {
      index: i,
      run,
      error,
      parsed,
      parseFailed,
      classification: validationErrors.length > 0 ? 'major_deviation' : 'conforming',
      deviations: [],
      validationErrors,
    };
  });

  const parsedOutputs = baseRuns
    .filter(run => run.parsed !== null)
    .map(run => ({ parsed: run.parsed, index: run.index }));

  const consensus = parsedOutputs.length >= 2
    ? computeConsensusSchema(parsedOutputs)
    : null;
  const anchorSchema = createAnchorSchema(anchor);
  const referenceSchema = anchorSchema ?? consensus;
  const referenceSchemaKind = anchorSchema
    ? 'anchor'
    : consensus
      ? 'consensus'
      : null;

  const analyzed = referenceSchema
    ? baseRuns.map((run) => {
        if (run.parsed === null) {
          return run;
        }

        const deviations = findDeviations(run.parsed, referenceSchema);
        return {
          ...run,
          deviations,
          classification: run.validationErrors.length > 0
            ? 'major_deviation'
            : classifyRun(deviations),
        };
      })
    : baseRuns;

  return { analyzed, consensus, referenceSchema, referenceSchemaKind };
}

export function buildVisualizationEntries(
  analyzed: Array<Omit<AnalyzedRun, 'anchorSimilarity'>>,
  anchor: AnchorPoint | null,
): VisualizationEntry[] {
  const entries: VisualizationEntry[] = analyzed
    .filter(run => run.run && !run.error)
    .map(run => ({
      kind: 'run',
      output: run.run?.output ?? '',
      runIndex: run.index,
      label: `Run ${run.index + 1}`,
    }));

  const shouldReuseRunPoint = anchor?.source === 'run'
    && anchor.runIndex !== null
    && analyzed[anchor.runIndex]?.run?.output === anchor.output;

  if (anchor?.output.trim() && !shouldReuseRunPoint) {
    entries.push({
      kind: 'anchor',
      output: anchor.output,
      runIndex: anchor.runIndex,
      label: anchor.label,
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

  const similarity = computeSimilarity(entries.map(entry => entry.output));
  return { ...similarity, entries };
}

export function buildAnchorSimilarityMap(
  similarity: VisualizationSimilarity | null,
  anchor: AnchorPoint | null,
): Map<number, number> {
  const anchorIndex = similarity?.entries.findIndex((entry) => {
    if (entry.kind === 'anchor') {
      return true;
    }

    return anchor?.source === 'run'
      && anchor.runIndex === entry.runIndex
      && anchor.output === entry.output;
  }) ?? -1;

  if (!similarity || anchorIndex < 0) {
    return new Map();
  }

  const similarities = new Map<number, number>();
  similarity.entries.forEach((entry, entryIndex) => {
    if (entry.kind === 'run' && entry.runIndex !== null) {
      similarities.set(entry.runIndex, similarity.matrix[entryIndex]?.[anchorIndex] ?? 0);
    }
  });

  return similarities;
}

export function useRunAnalysis(
  results: Array<PlaygroundRun | null>,
  errors: Array<string | null>,
  anchor: AnchorPoint | null,
): RunAnalysis {
  const { analyzed: baseAnalyzed, consensus, referenceSchema, referenceSchemaKind } = useMemo(
    () => analyzeRunsWithReference(results, errors, anchor),
    [results, errors, anchor],
  );

  const similarity = useMemo(() => {
    const outputs = baseAnalyzed
      .filter(run => run.run && !run.error)
      .map(run => run.run?.output ?? '');

    if (outputs.length < 2) {
      return null;
    }

    return computeSimilarity(outputs);
  }, [baseAnalyzed]);

  const visualizationSimilarity = useMemo(
    () => buildVisualizationSimilarity(baseAnalyzed, anchor),
    [baseAnalyzed, anchor],
  );

  const anchorSimilarityMap = useMemo(
    () => buildAnchorSimilarityMap(visualizationSimilarity, anchor),
    [visualizationSimilarity, anchor],
  );

  const analyzed = useMemo(
    () => baseAnalyzed.map(run => ({
      ...run,
      anchorSimilarity: anchorSimilarityMap.get(run.index) ?? null,
    })),
    [baseAnalyzed, anchorSimilarityMap],
  );

  const scatterPoints = useMemo(() => {
    if (!visualizationSimilarity) {
      return [];
    }

    return visualizationSimilarity.points2D.map((point, entryIndex) => {
      const entry = visualizationSimilarity.entries[entryIndex];
      const analyzedRun = entry.runIndex !== null ? analyzed[entry.runIndex] : null;

      return {
        x: point.x,
        y: point.y,
        index: entry.runIndex,
        classification: analyzedRun?.classification ?? 'conforming',
        similarity: visualizationSimilarity.averageSimilarity[entryIndex] ?? 1,
        isAnchor: entry.kind === 'anchor'
          || (anchor?.source === 'run' && anchor.runIndex === entry.runIndex && analyzedRun?.run?.output === anchor.output),
        label: entry.kind === 'anchor'
          ? 'Anchor'
          : String((entry.runIndex ?? 0) + 1),
      };
    });
  }, [visualizationSimilarity, analyzed, anchor]);

  const consensusOutputIndex = useMemo(() => {
    if (!similarity) {
      return -1;
    }

    const validIndices = baseAnalyzed
      .filter(run => run.run && !run.error)
      .map(run => run.index);

    return findConsensusOutputIndex(similarity.averageSimilarity, validIndices);
  }, [similarity, baseAnalyzed]);

  const reference = useMemo(() => {
    if (anchor?.output.trim()) {
      return {
        kind: 'anchor' as const,
        label: anchor.label,
        output: anchor.output,
        runIndex: anchor.runIndex,
      };
    }

    if (consensusOutputIndex < 0) {
      return null;
    }

    const run = analyzed[consensusOutputIndex]?.run;
    if (!run) {
      return null;
    }

    return {
      kind: 'consensus' as const,
      label: `Run ${consensusOutputIndex + 1}`,
      output: run.output,
      runIndex: consensusOutputIndex,
    };
  }, [anchor, consensusOutputIndex, analyzed]);

  const counts = useMemo(() => {
    const map: Record<RunClassification, number> = {
      conforming: 0,
      minor_deviation: 0,
      major_deviation: 0,
      failure: 0,
    };

    for (const run of analyzed) {
      map[run.classification]++;
    }

    return map;
  }, [analyzed]);

  return {
    analyzed,
    consensus,
    referenceSchema,
    referenceSchemaKind,
    reference,
    similarity,
    scatterPoints,
    consensusOutputIndex,
    counts,
  };
}
