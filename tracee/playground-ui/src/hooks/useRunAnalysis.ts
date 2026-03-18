import Ajv from 'ajv';
import { useMemo } from 'react';
import type { PlaygroundRun } from '../types/playground';
import {
  computeConsensusSchema,
  classifyRun,
  findDeviations,
} from '../utils/schemaAggregation';
import type { FieldDeviation, RunClassification, ConsensusSchema } from '../utils/schemaAggregation';
import { computeSimilarity } from '../utils/cosineSimilarity';
import type { SimilarityResult } from '../utils/cosineSimilarity';
import { findConsensusOutputIndex } from '../utils/jsonDiff';

export interface AnalyzedRun {
  index: number;
  run: PlaygroundRun | null;
  error: string | null;
  parsed: unknown | null;
  parseFailed: boolean;
  classification: RunClassification;
  deviations: FieldDeviation[];
  validationErrors: Array<{ message?: string; instancePath?: string }>;
}

export interface RunAnalysis {
  analyzed: AnalyzedRun[];
  consensus: ConsensusSchema | null;
  similarity: SimilarityResult | null;
  scatterPoints: Array<{
    x: number;
    y: number;
    index: number;
    classification: RunClassification;
    similarity: number;
  }>;
  consensusOutputIndex: number;
  counts: Record<RunClassification, number>;
}

export function useRunAnalysis(
  results: Array<PlaygroundRun | null>,
  errors: Array<string | null>,
): RunAnalysis {
  const { analyzed, consensus } = useMemo(() => {
    const ajv = new Ajv();

    const baseRuns: AnalyzedRun[] = results.map((run, i) => {
      const error = errors[i];
      if (error || !run) {
        return {
          index: i, run, error, parsed: null, parseFailed: false,
          classification: 'failure' as RunClassification,
          deviations: [], validationErrors: [],
        };
      }

      let parsed: unknown = null;
      let parseFailed = false;
      let validationErrors: Array<{ message?: string; instancePath?: string }> = [];

      try {
        parsed = JSON.parse(run.output);
      } catch {
        parseFailed = true;
      }

      if (parsed !== null && run.output_schema) {
        try {
          const validate = ajv.compile(run.output_schema as object);
          validate(parsed);
          validationErrors = validate.errors ?? [];
        } catch {
          // Schema compilation error — skip
        }
      }

      const classification = validationErrors.length > 0
        ? 'major_deviation'
        : 'conforming';

      return {
        index: i, run, error, parsed, parseFailed,
        classification,
        deviations: [], validationErrors,
      };
    });

    const parsedOutputs = baseRuns
      .filter(r => r.parsed !== null)
      .map(r => ({ parsed: r.parsed, index: r.index }));

    const consensusResult = parsedOutputs.length >= 2
      ? computeConsensusSchema(parsedOutputs)
      : null;

    const finalRuns = consensusResult
      ? baseRuns.map(run => {
          if (run.parsed === null) return run;
          const deviations = findDeviations(run.parsed, consensusResult);
          const classification = run.validationErrors.length > 0
            ? 'major_deviation'
            : classifyRun(deviations);
          return { ...run, deviations, classification };
        })
      : baseRuns;

    return { analyzed: finalRuns, consensus: consensusResult };
  }, [results, errors]);

  const similarity = useMemo(() => {
    const outputs = results.map((run, i) => {
      if (errors[i] || !run) return '';
      return run.output;
    });
    const validOutputs = outputs.filter(o => o.length > 0);
    if (validOutputs.length < 2) return null;
    return computeSimilarity(validOutputs);
  }, [results, errors]);

  const scatterPoints = useMemo(() => {
    if (!similarity) return [];
    const validIndices = analyzed
      .filter(r => r.run && !r.error)
      .map(r => r.index);

    return similarity.points2D.map((pt, i) => ({
      x: pt.x,
      y: pt.y,
      index: validIndices[i] ?? i,
      classification: analyzed[validIndices[i] ?? i]?.classification ?? 'conforming',
      similarity: similarity.averageSimilarity[i] ?? 1,
    }));
  }, [similarity, analyzed]);

  const consensusOutputIndex = useMemo(() => {
    if (!similarity) return -1;
    const validIndices = analyzed
      .filter(r => r.run && !r.error)
      .map(r => r.index);
    return findConsensusOutputIndex(similarity.averageSimilarity, validIndices);
  }, [similarity, analyzed]);

  const counts = useMemo(() => {
    const map: Record<RunClassification, number> = {
      conforming: 0, minor_deviation: 0, major_deviation: 0, failure: 0,
    };
    for (const r of analyzed) map[r.classification]++;
    return map;
  }, [analyzed]);

  return { analyzed, consensus, similarity, scatterPoints, consensusOutputIndex, counts };
}
