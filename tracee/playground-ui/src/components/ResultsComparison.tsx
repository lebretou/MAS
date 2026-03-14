import React from 'react';
import { AnalyzedRun } from '../hooks/useRunAnalysis';
import {
  ConsensusSchema,
  RunClassification,
  getClassificationBg,
  getClassificationColor,
} from '../utils/schemaAggregation';
import SimilarityScatterplot from './SimilarityScatterplot';
import DeviationHeatmap from './DeviationHeatmap';

interface Props {
  analyzed: AnalyzedRun[];
  consensus: ConsensusSchema | null;
  scatterPoints: Array<{
    x: number;
    y: number;
    index: number;
    classification: RunClassification;
    similarity: number;
  }>;
  counts: Record<RunClassification, number>;
  selectedRun: number | null;
  onSelectRun: (index: number) => void;
}

const ResultsComparison: React.FC<Props> = ({
  analyzed,
  consensus,
  scatterPoints,
  counts,
  selectedRun,
  onSelectRun,
}) => {
  if (analyzed.length === 0) {
    return (
      <div className="card create-run__empty-card">
        <div className="empty-state create-run__empty-body">
          <div className="empty-state__icon">&#9881;</div>
          <div className="empty-state__title">No results yet</div>
          <div className="empty-state__desc">
            Configure your prompt and execute a run to see the visualization and analysis here.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-col results__container">
      {/* Summary bar */}
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

      {/* Failure banner */}
      {counts.failure > 0 && (
        <div className="alert alert--danger">
          <span className="alert__icon">!</span>
          {counts.failure} run{counts.failure > 1 ? 's' : ''} failed to produce output.
        </div>
      )}

      {/* Scatterplot */}
      {scatterPoints.length >= 2 && (
        <SimilarityScatterplot
          points={scatterPoints}
          selectedIndex={selectedRun}
          onSelectRun={onSelectRun}
        />
      )}

      {/* Heatmap */}
      {consensus && (
        <DeviationHeatmap
          consensus={consensus}
          runDeviations={analyzed.map(r => r.deviations)}
          onCellClick={(runIndex) => onSelectRun(runIndex)}
        />
      )}

      {/* Run cards */}
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
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelectRun(run.index); }}
              >
                <div className="run-card__header">
                  <span className="run-card__title">Run {run.index + 1}</span>
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
                {run.error && (
                  <div className="results__error-text">{run.error}</div>
                )}
                <div className="run-card__meta">
                  {run.run?.latency_ms && <span>{run.run.latency_ms}ms</span>}
                  {run.run?.total_tokens && <span>{run.run.total_tokens} tokens</span>}
                  {run.deviations.length > 0 && (
                    <span>{run.deviations.length} deviation{run.deviations.length > 1 ? 's' : ''}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ResultsComparison;
