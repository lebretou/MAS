import React from 'react';
import type { AnalyzedRun, ComparisonReference, FieldOption, ProjectionItem } from '../../../hooks/useRunAnalysis';
import SimilarityScatterplot from './SimilarityScatterplot';
import FieldDistributionView from './FieldDistributionView';
import { collectFieldValues } from '../../../hooks/useRunAnalysis';
import { useEmbeddingProjection } from '../../../hooks/useEmbeddingProjection';

interface Props {
  analyzed: AnalyzedRun[];
  reference: ComparisonReference | null;
  projectionItems: ProjectionItem[];
  fieldOptions: FieldOption[];
  failureCount: number;
  selectedRun: string | null;
  onSelectRun: (index: string | null) => void;
  detailContent: React.ReactNode;
}

const ResultsComparison: React.FC<Props> = ({
  analyzed,
  reference,
  projectionItems,
  fieldOptions,
  failureCount,
  selectedRun,
  onSelectRun,
  detailContent,
}) => {
  const [selectedFieldPath, setSelectedFieldPath] = React.useState('');
  const [stringMode, setStringMode] = React.useState<'semantic' | 'exact'>('semantic');
  const formatLatency = (latencyMs: number | null | undefined) => (
    typeof latencyMs === 'number' ? `${latencyMs.toFixed(1)}ms` : null
  );
  const getRunVersionLabel = React.useCallback((versionId: string | null | undefined, fallbackLabel: string) => (
    versionId || fallbackLabel
  ), []);
  const selectedField = React.useMemo(
    () => fieldOptions.find((field) => field.path === selectedFieldPath) ?? null,
    [fieldOptions, selectedFieldPath],
  );
  React.useEffect(() => {
    if (selectedFieldPath && !fieldOptions.some((field) => field.path === selectedFieldPath)) {
      setSelectedFieldPath('');
    }
  }, [fieldOptions, selectedFieldPath]);
  const fieldValues = React.useMemo(
    () => selectedField ? collectFieldValues(analyzed, selectedField.path) : [],
    [analyzed, selectedField],
  );
  const projectionSourceItems = React.useMemo(() => {
    if (!selectedField || selectedField.type !== 'string' || stringMode !== 'semantic') {
      return projectionItems;
    }

    return fieldValues
      .filter((entry): entry is typeof entry & { value: string } => typeof entry.value === 'string' && entry.value.trim().length > 0)
      .map((entry) => ({
        id: `field:${selectedField.path}:${entry.selectionId}`,
        kind: 'run' as const,
        output: entry.value,
        selectionId: entry.selectionId,
        groupId: entry.groupId,
        groupLabel: entry.groupLabel,
        groupVersionId: entry.groupVersionId,
        groupTone: entry.groupTone,
        label: entry.label,
        isFailed: false,
      }));
  }, [fieldValues, projectionItems, selectedField, stringMode]);
  const projection = useEmbeddingProjection(projectionSourceItems);
  const isScatterMode = !selectedField || (selectedField.type === 'string' && stringMode === 'semantic');
  const plottedRunCount = React.useMemo(
    () => projectionSourceItems.filter((item) => item.kind === 'run').length,
    [projectionSourceItems],
  );

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
    <div className="results__layout">
      <div className="results__visual">
        <div className="results__analysis-toolbar">
          <div className="field">
            <label className="field__label" htmlFor="playground-analysis-field">
              View
            </label>
            <select
              id="playground-analysis-field"
              className="select"
              value={selectedFieldPath}
              onChange={(event) => setSelectedFieldPath(event.target.value)}
            >
              <option value="">All outputs</option>
              {fieldOptions.map((field) => (
                <option key={field.path} value={field.path}>
                  {field.label}
                </option>
              ))}
            </select>
          </div>
          {selectedField?.type === 'string' && (
            <div className="field">
              <label className="field__label" htmlFor="playground-analysis-string-mode">
                String mode
              </label>
              <select
                id="playground-analysis-string-mode"
                className="select"
                value={stringMode}
                onChange={(event) => setStringMode(event.target.value as 'semantic' | 'exact')}
              >
                <option value="semantic">Semantic map</option>
                <option value="exact">Exact match</option>
              </select>
            </div>
          )}
        </div>

        {failureCount > 0 && (
          <div className="alert alert--danger">
            <span className="alert__icon">!</span>
            {failureCount} run{failureCount > 1 ? 's' : ''} failed to produce output.
          </div>
        )}

        {isScatterMode && projection.points.length >= 2 ? (
          <SimilarityScatterplot
            title={selectedField ? `${selectedField.label} semantic map` : 'Output map'}
            hint={selectedField ? 'embedding-based field projection' : 'embedding-based projection with cosine similarity and pca'}
            summary={`${plottedRunCount} plotted run${plottedRunCount === 1 ? '' : 's'}${failureCount > 0 ? `, ${failureCount} failed` : ''}.`}
            points={projection.points}
            selectedIndex={selectedRun}
            onSelectRun={onSelectRun}
            anchorLabel={reference?.kind === 'anchor' && !selectedField ? reference.label : null}
          />
        ) : isScatterMode && projection.loading ? (
          <div className="card">
            <div className="empty-state create-run__empty-body">
              <div className="empty-state__title">Projecting outputs...</div>
              <div className="empty-state__desc">
                Embeddings are being generated for the current analysis selection.
              </div>
            </div>
          </div>
        ) : isScatterMode && projection.error ? (
          <div className="card">
            <div className="empty-state create-run__empty-body">
              <div className="empty-state__title">Projection unavailable</div>
              <div className="empty-state__desc">
                Embedding analysis failed for this selection. Try running the projection again.
              </div>
            </div>
          </div>
        ) : !isScatterMode && selectedField ? (
          <FieldDistributionView
            field={selectedField}
            values={fieldValues}
          />
        ) : (
          <div className="card">
            <div className="empty-state create-run__empty-body">
              <div className="empty-state__title">Not enough runs to plot</div>
              <div className="empty-state__desc">
                Render at least two plotted entries by running more prompts or selecting a field with data.
              </div>
            </div>
          </div>
        )}
      </div>

      <aside className="results__sidebar">
        <div className="results__sidebar-section">
          <div className="results__sidebar-head">
            <span className="section-label">Runs</span>
            <span className="field__hint">select one to inspect its output</span>
          </div>
          <div className="results__run-list">
            {analyzed.map((run) => {
              const isSelected = run.selectionId === selectedRun;
              const isFailed = run.state === 'failed';
              const isReady = run.state === 'ready';
              const needsAttention = run.state === 'non_json' || run.state === 'schema_invalid';
              const latencyLabel = formatLatency(run.run?.latency_ms);
              const runVersionLabel = getRunVersionLabel(run.groupVersionId ?? run.run?.version_id, run.groupLabel);

              let cardClass = 'run-card results__run-card';
              if (isSelected) cardClass += ' run-card--selected';
              if (run.groupTone === 'compare') cardClass += ' results__run-card--compare-target';
              if (isFailed) cardClass += ' run-card--failure';
              else if (needsAttention) cardClass += ' run-card--deviation';

              return (
                <div
                  key={run.selectionId}
                  className={cardClass}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelectRun(isSelected ? null : run.selectionId)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelectRun(isSelected ? null : run.selectionId);
                    }
                  }}
                >
                  <div className="run-card__header">
                    <span className="run-card__title">{runVersionLabel} · Run {run.index + 1}</span>
                    <div className="results__badge-group">
                      {isReady && <span className="badge badge--success">success</span>}
                      {isFailed && <span className="badge badge--danger">fail</span>}
                      {reference?.kind === 'anchor' && reference.anchorSelectionId === run.selectionId && (
                        <span className="badge badge--primary">anchor</span>
                      )}
                      {run.state === 'non_json' && <span className="badge badge--warning">non-json</span>}
                      {run.state === 'schema_invalid' && <span className="badge badge--warning">schema</span>}
                    </div>
                  </div>
                  {run.error && (
                    <div className="results__error-text">{run.error}</div>
                  )}
                  <div className="run-card__meta">
                    {latencyLabel && <span>{latencyLabel}</span>}
                    {run.run?.total_tokens && <span>{run.run.total_tokens} tokens</span>}
                    {run.run?.tool_calls?.length ? <span>{run.run.tool_calls.length} tool call{run.run.tool_calls.length > 1 ? 's' : ''}</span> : null}
                    {reference?.kind === 'anchor' && run.anchorSimilarity !== null && (
                      <span>{Math.round(run.anchorSimilarity * 100)}% to anchor</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="results__sidebar-section results__sidebar-section--detail">
          <div className="results__sidebar-head">
            <span className="section-label">Output detail</span>
          </div>
          {detailContent}
        </div>
      </aside>
    </div>
  );
};

export default ResultsComparison;
