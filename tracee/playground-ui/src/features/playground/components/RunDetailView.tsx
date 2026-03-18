import React, { useEffect, useMemo, useState } from 'react';
import type { AnalyzedRun, ComparisonReference } from '../../../hooks/useRunAnalysis';
import {
  getClassificationBg,
  getClassificationColor,
} from '../../../utils/schemaAggregation';
import { computeDiff } from '../../../utils/jsonDiff';
import OutputDiffView from './OutputDiffView';
import JsonTreeView from './JsonTreeView';

interface Props {
  analyzed: AnalyzedRun[];
  selectedRun: number | null;
  reference: ComparisonReference | null;
  onBack: () => void;
  onPromoteRun: (index: number) => void;
}

const RunDetailView: React.FC<Props> = ({
  analyzed,
  selectedRun,
  reference,
  onBack,
  onPromoteRun,
}) => {
  const [detailMode, setDetailMode] = useState<'tree' | 'raw'>('tree');
  const [diffDismissed, setDiffDismissed] = useState(false);

  useEffect(() => {
    setDiffDismissed(false);
  }, [selectedRun, reference?.kind, reference?.output, reference?.runIndex]);

  const selected = selectedRun !== null ? analyzed[selectedRun] : null;

  const selectedDiff = useMemo(() => {
    if (selectedRun === null || !reference) return null;
    if (reference.runIndex === selectedRun) return null;

    const selectedOutput = analyzed[selectedRun]?.run?.output;
    if (!selectedOutput || !reference.output) return null;

    const format = (s: string): string => {
      const normalized = s.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();

      try { return JSON.stringify(JSON.parse(normalized), null, 2); } catch { return normalized; }
    };

    return computeDiff(format(reference.output), format(selectedOutput));
  }, [selectedRun, reference, analyzed]);

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
        <button className="btn btn--ghost btn--sm" onClick={onBack}>
          Close detail
        </button>

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
            {reference?.kind === 'anchor' && reference.runIndex === selected.index && (
              <span className="badge badge--primary results__badge-ml-sm">anchor</span>
            )}
          </span>
          {selected.run && (
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => onPromoteRun(selected.index)}
            >
              promote as anchor
            </button>
          )}
        </div>

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

        {selected.deviations.length > 0 && (
          <div className="alert alert--info results__alert-spaced">
            <span className="alert__icon">i</span>
            <div className="results__deviation-detail">
              {selected.deviations.filter(d => d.type === 'missing').length > 0 && (
                <div>Missing: {selected.deviations.filter(d => d.type === 'missing').map(d => d.path).join(', ')}</div>
              )}
              {selected.deviations.filter(d => d.type === 'type_mismatch').length > 0 && (
                <div>Type mismatch: {selected.deviations.filter(d => d.type === 'type_mismatch').map(d => `${d.path} (${d.expected} -> ${d.actual})`).join(', ')}</div>
              )}
              {selected.deviations.filter(d => d.type === 'extra').length > 0 && (
                <div>Extra: {selected.deviations.filter(d => d.type === 'extra').map(d => d.path).join(', ')}</div>
              )}
            </div>
          </div>
        )}

        {selected.run?.tool_calls?.length ? (
          <div className="card">
            <div className="card__header">
              <span className="section-label">Tool Calls</span>
            </div>
            <div className="card__body results__tool-calls">
              {selected.run.tool_calls.map((toolCall, index) => (
                <div key={toolCall.call_id ?? `${toolCall.name}-${index}`} className="results__tool-call-card">
                  <div className="results__tool-call-head">
                    <span className="results__tool-call-name">{toolCall.name}</span>
                    <span className="badge">call {index + 1}</span>
                  </div>
                  <pre className="results__tool-call-args">
                    {JSON.stringify(toolCall.arguments ?? {}, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {selectedDiff && selectedRun !== null && !diffDismissed && (
          <OutputDiffView
            diff={selectedDiff}
            referenceLabel={reference?.label ?? 'reference'}
            selectedRunLabel={`Run ${selectedRun + 1}`}
            referenceKind={reference?.kind ?? 'consensus'}
            onClose={() => setDiffDismissed(true)}
          />
        )}

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
