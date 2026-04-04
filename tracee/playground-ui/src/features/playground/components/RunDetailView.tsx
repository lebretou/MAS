import React, { useEffect, useMemo, useState } from 'react';
import type { AnalyzedRun, ComparisonReference } from '../../../hooks/useRunAnalysis';
import { computeDiff } from '../../../utils/jsonDiff';
import OutputDiffView from './OutputDiffView';
import JsonTreeView from './JsonTreeView';
import iconAnchor from '../../../assets/icon-anchor.svg';
import iconCopy from '../../../assets/icon-copy.svg';

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

interface Props {
  analyzed: AnalyzedRun[];
  selectedRun: string | null;
  reference: ComparisonReference | null;
  onPromoteRun: (selectionId: string) => void;
  onRemoveAnchor?: () => void;
}

const RunDetailView: React.FC<Props> = ({
  analyzed,
  selectedRun,
  reference,
  onPromoteRun,
  onRemoveAnchor,
}) => {
  const [detailMode, setDetailMode] = useState<'tree' | 'raw'>('tree');
  const [diffDismissed, setDiffDismissed] = useState(false);
  const formatLatency = (latencyMs: number | null | undefined) => (
    typeof latencyMs === 'number' ? `${latencyMs.toFixed(1)}ms` : '-'
  );
  const getRunVersionLabel = (versionId: string | null | undefined, fallbackLabel: string) => (
    versionId || fallbackLabel
  );

  useEffect(() => {
    setDiffDismissed(false);
  }, [selectedRun, reference?.kind, reference?.output, reference?.anchorSelectionId]);

  const selected = selectedRun !== null
    ? analyzed.find((run) => run.selectionId === selectedRun) ?? null
    : null;
  const selectedRunLabel = selected
    ? `${getRunVersionLabel(selected.groupVersionId ?? selected.run?.version_id, selected.groupLabel)} · Run ${selected.index + 1}`
    : '';

  const selectedDiff = useMemo(() => {
    if (!selected || !reference) return null;
    if (reference.anchorSelectionId === selected.selectionId) return null;

    const selectedOutput = selected.run?.output;
    if (!selectedOutput || !reference.output) return null;

    const format = (s: string): string => {
      const normalized = s.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();

      try { return JSON.stringify(JSON.parse(normalized), null, 2); } catch { return normalized; }
    };

    return computeDiff(format(reference.output), format(selectedOutput));
  }, [selected, reference]);

  if (!selected) {
    return (
      <div className="empty-state create-run__empty-body detail-view__empty-state">
        <div className="empty-state__title">No run selected</div>
        <div className="empty-state__desc">
          Click a run card or dot in the scatterplot to view details.
        </div>
      </div>
    );
  }

  return (
    <div className="detail-view__content">
      <div className="flex-col results__container">
        <div className="detail-view__header">
          <div className="detail-view__title-wrap">
            <span className="detail-view__title">
              {selectedRunLabel}
              {selected.state === 'ready' && (
                <span className="badge badge--success results__badge-ml">success</span>
              )}
              {selected.state === 'failed' && (
                <span className="badge badge--danger results__badge-ml">fail</span>
              )}
              {selected.state === 'non_json' && (
                <span className="badge badge--warning results__badge-ml-sm">non-json</span>
              )}
              {selected.state === 'schema_invalid' && (
                <span className="badge badge--warning results__badge-ml-sm">schema issues</span>
              )}
              {reference?.anchorSelectionId === selected.selectionId && (
                <span className="badge badge--primary results__badge-ml-sm">anchor</span>
              )}
            </span>
          </div>
        </div>

        {selected.run && (
          <div className="results__meta-grid">
            <div className="meta-card">
              <div className="meta-card__key">Latency</div>
              <div className="meta-card__value">{formatLatency(selected.run.latency_ms)}</div>
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

        {selectedDiff && !diffDismissed && (
          <OutputDiffView
            diff={selectedDiff}
            referenceLabel={reference?.label ?? 'reference'}
            selectedRunLabel={selectedRunLabel}
            onClose={() => setDiffDismissed(true)}
          />
        )}

        <div className="detail-view__toolbar">
          <div className="seg-control results__seg-control">
            <button
              type="button"
              className={`seg-control__btn ${detailMode === 'tree' ? 'is-active' : ''}`}
              onClick={() => setDetailMode('tree')}
            >
              Tree View
            </button>
            <button
              type="button"
              className={`seg-control__btn ${detailMode === 'raw' ? 'is-active' : ''}`}
              onClick={() => setDetailMode('raw')}
            >
              Raw
            </button>
          </div>
          <div className="detail-view__toolbar-actions">
            {selected.run && (
              <button
                type="button"
                className="btn btn--secondary create-run__action-btn"
                onClick={() => {
                  if (selected.run?.output && globalThis.navigator?.clipboard?.writeText) {
                    void globalThis.navigator.clipboard.writeText(selected.run.output);
                  }
                }}
              >
                <span
                  className="create-run__action-icon"
                  style={getMaskIconStyle(iconCopy)}
                  aria-hidden
                />
                copy output
              </button>
            )}
            {selected.run && (() => {
              const isAnchor = reference?.anchorSelectionId === selected.selectionId;
              return isAnchor ? (
                <button
                  type="button"
                  className="btn btn--secondary create-run__action-btn create-run__action-btn--danger"
                  onClick={() => onRemoveAnchor?.()}
                >
                  <span
                    className="create-run__action-icon"
                    style={getMaskIconStyle(iconAnchor)}
                    aria-hidden
                  />
                  remove anchor
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn--secondary create-run__action-btn"
                  onClick={() => onPromoteRun(selected.selectionId)}
                >
                  <span
                    className="create-run__action-icon"
                    style={getMaskIconStyle(iconAnchor)}
                    aria-hidden
                  />
                  promote as anchor
                </button>
              );
            })()}
          </div>
        </div>

        {selected.error ? (
          <div className="alert alert--danger">
            <span className="alert__icon">!</span>
            {selected.error}
          </div>
        ) : selected.run && (
          detailMode === 'tree' && selected.parsed !== null ? (
            <JsonTreeView data={selected.parsed} />
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
