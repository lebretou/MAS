import React from 'react';
import type { AnalyzedRun, ComparisonReference, ScatterPoint } from '../../../hooks/useRunAnalysis';
import {
  getClassificationBg,
  getClassificationColor,
} from '../../../utils/schemaAggregation';
import type { ConsensusSchema, RunClassification } from '../../../utils/schemaAggregation';
import SimilarityScatterplot from './SimilarityScatterplot';
import DeviationHeatmap from './DeviationHeatmap';

interface Props {
  analyzed: AnalyzedRun[];
  reference: ComparisonReference | null;
  referenceSchema: ConsensusSchema | null;
  referenceSchemaKind: 'anchor' | 'consensus' | null;
  scatterPoints: ScatterPoint[];
  counts: Record<RunClassification, number>;
  selectedRun: number | null;
  onSelectRun: (index: number) => void;
  onPromoteRun: (index: number) => void;
}

const ResultsComparison: React.FC<Props> = ({
  analyzed,
  reference,
  referenceSchema,
  referenceSchemaKind,
  scatterPoints,
  counts,
  selectedRun,
  onSelectRun,
  onPromoteRun,
}) => {
  if (analyzed.length === 0) {
    return (
      <div className="card create-run__empty-card">
        <div className="empty-state create-run__empty-body">
          <div className="empty-state__icon">&#9881;</div>
          <div className="empty-state__title">No results yet</div>
          <div className="empty-state__desc">
            Build the prompt, open model or schema panels only when needed, then run a few repetitions to inspect consistency.
          </div>
          <div className="create-run__empty-steps">
            <span>1. shape the prompt</span>
            <span>2. set model, tools, or schema</span>
            <span>3. execute repeated runs</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-col results__container">
      <div className="summary-bar">
        {counts.conforming > 0 && (
          <div className="summary-stat">
            <span className="summary-stat__dot" style={{ background: '#065f46' }} />
            {counts.conforming} conforming
          </div>
        )}
        {counts.minor_deviation > 0 && (
          <div className="summary-stat">
            <span className="summary-stat__dot" style={{ background: '#92400e' }} />
            {counts.minor_deviation} minor deviation
          </div>
        )}
        {counts.major_deviation > 0 && (
          <div className="summary-stat">
            <span className="summary-stat__dot" style={{ background: '#991b1b' }} />
            {counts.major_deviation} major deviation
          </div>
        )}
        {counts.failure > 0 && (
          <div className="summary-stat">
            <span className="summary-stat__dot" style={{ background: '#991b1b' }} />
            {counts.failure} failed
          </div>
        )}
      </div>

      {counts.failure > 0 && (
        <div className="alert alert--danger">
          <span className="alert__icon">!</span>
          {counts.failure} run{counts.failure > 1 ? 's' : ''} failed to produce output.
        </div>
      )}

      {reference?.kind === 'anchor' && (
        <div className="card results__anchor-card">
          <div className="card__body results__anchor-body">
            <div>
              <div className="results__anchor-title">Anchor active</div>
              <div className="field__hint">
                {reference.label}
                {referenceSchemaKind === 'anchor'
                  ? ' is driving deviations and diffs.'
                  : ' is shown in the scatterplot and diff view.'}
              </div>
            </div>
            <span className="badge badge--primary">anchor</span>
          </div>
        </div>
      )}

      {scatterPoints.length >= 2 && (
        <SimilarityScatterplot
          points={scatterPoints}
          selectedIndex={selectedRun}
          onSelectRun={onSelectRun}
        />
      )}

      {referenceSchema && (
        <DeviationHeatmap
          consensus={referenceSchema}
            title={referenceSchemaKind === 'anchor' ? 'Anchor Deviation Heatmap' : 'Deviation Heatmap'}
            hint={referenceSchemaKind === 'anchor' ? 'Fields vs. anchor output' : 'Fields vs. runs'}
          runDeviations={analyzed.map(r => r.deviations)}
          onCellClick={(runIndex) => onSelectRun(runIndex)}
        />
      )}

      <div>
        <span className="section-label">All Runs</span>
        <div className="run-grid results__run-grid">
          {analyzed.map((run) => {
            const isSelected = run.index === selectedRun;

            let cardClass = 'run-card';
            if (isSelected) cardClass += ' run-card--selected';
            if (run.classification === 'failure') cardClass += ' run-card--failure';
            else if (run.classification !== 'conforming') cardClass += ' run-card--deviation';

            return (
              <div
                key={run.index}
                className={cardClass}
                role="button"
                tabIndex={0}
                onClick={() => onSelectRun(run.index)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectRun(run.index);
                  }
                }}
              >
                <div className="run-card__header">
                  <span className="run-card__title">Run {run.index + 1}</span>
                  <div className="results__badge-group">
                    {reference?.kind === 'anchor' && reference.runIndex === run.index && (
                      <span className="badge badge--primary">anchor</span>
                    )}
                    <span
                      className="badge"
                      style={{
                        background: getClassificationBg(run.classification),
                        color: getClassificationColor(run.classification),
                      }}
                    >
                      {run.classification === 'failure' ? 'failed' : run.classification.replace('_', ' ')}
                    </span>
                  </div>
                </div>
                {run.error && (
                  <div className="results__error-text">{run.error}</div>
                )}
                <div className="run-card__meta">
                  {run.run?.latency_ms && <span>{run.run.latency_ms}ms</span>}
                  {run.run?.total_tokens && <span>{run.run.total_tokens} tokens</span>}
                  {run.run?.tool_calls?.length ? <span>{run.run.tool_calls.length} tool call{run.run.tool_calls.length > 1 ? 's' : ''}</span> : null}
                  {reference?.kind === 'anchor' && run.anchorSimilarity !== null && (
                    <span>{Math.round(run.anchorSimilarity * 100)}% to anchor</span>
                  )}
                  {run.deviations.length > 0 && (
                    <span>{run.deviations.length} deviation{run.deviations.length > 1 ? 's' : ''}</span>
                  )}
                </div>
                {run.run && (
                  <div className="results__card-actions">
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onPromoteRun(run.index);
                      }}
                      onKeyDown={(e) => e.stopPropagation()}
                    >
                      promote as anchor
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ResultsComparison;
