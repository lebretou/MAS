import React, { useEffect, useMemo, useState } from 'react';
import type { AnalyzedRun } from '../hooks/useRunAnalysis';
import {
  getClassificationBg,
  getClassificationColor,
} from '../utils/schemaAggregation';
import { computeDiff } from '../utils/jsonDiff';
import OutputDiffView from './OutputDiffView';
import JsonTreeView from './JsonTreeView';

interface Props {
  analyzed: AnalyzedRun[];
  selectedRun: number | null;
  consensusOutputIndex: number;
  onBack: () => void;
}

const RunDetailView: React.FC<Props> = ({
  analyzed,
  selectedRun,
  consensusOutputIndex,
  onBack,
}) => {
  const [detailMode, setDetailMode] = useState<'tree' | 'raw'>('tree');
  const [diffDismissed, setDiffDismissed] = useState(false);

  useEffect(() => {
    setDiffDismissed(false);
  }, [selectedRun]);

  const selected = selectedRun !== null ? analyzed[selectedRun] : null;

  const selectedDiff = useMemo(() => {
    if (selectedRun === null || consensusOutputIndex < 0) return null;
    if (selectedRun === consensusOutputIndex) return null;

    const selectedOutput = analyzed[selectedRun]?.run?.output;
    const consensusOutput = analyzed[consensusOutputIndex]?.run?.output;
    if (!selectedOutput || !consensusOutput) return null;

    const format = (s: string): string => {
      try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; }
    };

    return computeDiff(format(consensusOutput), format(selectedOutput));
  }, [selectedRun, consensusOutputIndex, analyzed]);

  if (!selected) {
    return (
      <div className="card create-run__empty-card">
        <div className="empty-state create-run__empty-body">
          <div className="empty-state__title">No run selected</div>
          <div className="empty-state__desc">
            Click a run card or dot in the scatterplot to view details.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-view__content">
      <div className="flex-col results__container">
        {/* Back button */}
        <button className="btn btn--ghost btn--sm" onClick={onBack}>
          &#8592; Back to prompt
        </button>

        {/* Header */}
        <div className="detail-view__header">
          <span className="detail-view__title">
            Run {selected.index + 1}
            <span
              className="badge results__badge-ml"
              style={{
                background: getClassificationBg(selected.classification),
                color: getClassificationColor(selected.classification),
              }}
            >
              {selected.classification.replace('_', ' ')}
            </span>
            {selected.parseFailed && (
              <span className="badge badge--warning results__badge-ml-sm">non-JSON</span>
            )}
          </span>
        </div>

        {/* Metadata */}
        {selected.run && (
          <div className="results__meta-grid">
            <div className="meta-card">
              <div className="meta-card__key">Latency</div>
              <div className="meta-card__value">{selected.run.latency_ms ?? '-'}ms</div>
            </div>
            <div className="meta-card">
              <div className="meta-card__key">Tokens</div>
              <div className="meta-card__value">{selected.run.total_tokens ?? '-'}</div>
            </div>
            <div className="meta-card">
              <div className="meta-card__key">Model</div>
              <div className="meta-card__value">{selected.run.model}</div>
            </div>
          </div>
        )}

        {/* Validation errors */}
        {selected.validationErrors.length > 0 && (
          <div className="alert alert--warning results__alert-spaced">
            <span className="alert__icon">!</span>
            <div>
              <strong className="results__validation-title">Schema validation issues:</strong>
              <ul className="results__validation-list">
                {selected.validationErrors.map((e, i) => (
                  <li key={i}>
                    {e.instancePath ? `'${e.instancePath.replace(/^\//, '')}' ` : ''}
                    {e.message}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Deviation summary */}
        {selected.deviations.length > 0 && (
          <div className="alert alert--info results__alert-spaced">
            <span className="alert__icon">i</span>
            <div className="results__deviation-detail">
              {selected.deviations.filter(d => d.type === 'missing').length > 0 && (
                <div>Missing: {selected.deviations.filter(d => d.type === 'missing').map(d => d.path).join(', ')}</div>
              )}
              {selected.deviations.filter(d => d.type === 'type_mismatch').length > 0 && (
                <div>Type mismatch: {selected.deviations.filter(d => d.type === 'type_mismatch').map(d => `${d.path} (${d.expected} → ${d.actual})`).join(', ')}</div>
              )}
              {selected.deviations.filter(d => d.type === 'extra').length > 0 && (
                <div>Extra: {selected.deviations.filter(d => d.type === 'extra').map(d => d.path).join(', ')}</div>
              )}
            </div>
          </div>
        )}

        {/* Diff view */}
        {selectedDiff && selectedRun !== null && !diffDismissed && (
          <OutputDiffView
            diff={selectedDiff}
            consensusRunLabel={`Run ${consensusOutputIndex + 1}`}
            selectedRunLabel={`Run ${selectedRun + 1}`}
            onClose={() => setDiffDismissed(true)}
          />
        )}

        {/* View toggle */}
        <div className="seg-control results__seg-control">
          <button
            className={`seg-control__btn ${detailMode === 'tree' ? 'is-active' : ''}`}
            onClick={() => setDetailMode('tree')}
          >
            Tree View
          </button>
          <button
            className={`seg-control__btn ${detailMode === 'raw' ? 'is-active' : ''}`}
            onClick={() => setDetailMode('raw')}
          >
            Raw
          </button>
        </div>

        {/* Output content */}
        {selected.error ? (
          <div className="alert alert--danger">
            <span className="alert__icon">!</span>
            {selected.error}
          </div>
        ) : selected.run && (
          detailMode === 'tree' && selected.parsed !== null ? (
            <JsonTreeView data={selected.parsed} deviations={selected.deviations} />
          ) : (
            <pre className="results__raw-output">
              {selected.run.output}
            </pre>
          )
        )}
      </div>
    </div>
  );
};

export default RunDetailView;
