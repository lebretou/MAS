import React from 'react';
import type { ConsensusSchema, FieldDeviation } from '../../../utils/schemaAggregation';

interface Props {
  consensus: ConsensusSchema;
  runDeviations: FieldDeviation[][];
  onCellClick?: (runIndex: number, path: string) => void;
  title?: string;
  hint?: string;
}

type CellStatus = 'ok' | 'miss' | 'type' | 'extra' | 'na';

const DeviationHeatmap: React.FC<Props> = ({
  consensus,
  runDeviations,
  onCellClick,
  title = 'Deviation Heatmap',
  hint = 'Fields vs. Runs',
}) => {
  const majorityThreshold = consensus.totalRuns / 2;
  const significantFields = consensus.fields
    .filter(f => f.count >= majorityThreshold)
    .map(f => f.path)
    .concat(
      runDeviations
        .flatMap(devs => devs.filter(d => d.type === 'extra').map(d => d.path))
    );

  const allFields = Array.from(new Set(significantFields));
  const displayFields = allFields.filter(f => !f.includes('['));

  if (displayFields.length === 0) return null;

  const getCellStatus = (runIndex: number, path: string): CellStatus => {
    const devs = runDeviations[runIndex];
    if (!devs) return 'na';
    const dev = devs.find(d => d.path === path);
    if (!dev) return 'ok';
    switch (dev.type) {
      case 'missing': return 'miss';
      case 'type_mismatch': return 'type';
      case 'extra': return 'extra';
      default: return 'ok';
    }
  };

  const cellClass = (status: CellStatus): string => {
    switch (status) {
      case 'ok': return 'heatmap__ok';
      case 'miss': return 'heatmap__miss';
      case 'type': return 'heatmap__type';
      case 'extra': return 'heatmap__extra';
      case 'na': return 'heatmap__na';
    }
  };

  const cellLabel = (status: CellStatus): string => {
    switch (status) {
      case 'ok': return 'ok';
      case 'miss': return 'MISS';
      case 'type': return 'TYPE';
      case 'extra': return 'EXTRA';
      case 'na': return '-';
    }
  };

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">{title}</h3>
        <span className="field__hint heatmap__hint">{hint}</span>
      </div>
      <div className="card__body heatmap__body">
        <table className="heatmap">
          <thead>
            <tr>
              <th>Field</th>
              {runDeviations.map((_, i) => (
                <th key={i}>Run {i + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayFields.map(field => (
              <tr key={field}>
                <td className="table__mono">{field}</td>
                {runDeviations.map((_, runIdx) => {
                  const status = getCellStatus(runIdx, field);
                  return (
                    <td
                      key={runIdx}
                      className={cellClass(status)}
                      role="button"
                      tabIndex={0}
                      onClick={() => onCellClick?.(runIdx, field)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onCellClick?.(runIdx, field); }}
                    >
                      {cellLabel(status)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DeviationHeatmap;
