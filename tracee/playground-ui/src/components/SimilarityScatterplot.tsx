import React from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LabelList,
} from 'recharts';
import type { RunClassification } from '../utils/schemaAggregation';

interface RunPoint {
  x: number;
  y: number;
  index: number;
  classification: RunClassification;
  similarity: number;
}

interface Props {
  points: RunPoint[];
  selectedIndex: number | null;
  onSelectRun: (index: number) => void;
}

/**
 * Compute dynamic color thresholds from the data distribution.
 * Blue = cluster core (>= median), Cyan = moderate, Red = outlier (< median - 1.5*stddev).
 *
 * When the spread is small (stddev < 0.05), the group is uniformly clustered
 * and no runs should be flagged as outliers regardless of absolute scores.
 */
function computeSimilarityTiers(points: RunPoint[]): { blueThreshold: number; cyanThreshold: number } {
  const sims = points
    .filter(p => p.classification !== 'failure')
    .map(p => p.similarity)
    .sort((a, b) => a - b);

  if (sims.length === 0) return { blueThreshold: 0.85, cyanThreshold: 0.6 };

  const minSim = sims[0];
  const median = sims[Math.floor(sims.length / 2)];
  const mean = sims.reduce((a, b) => a + b, 0) / sims.length;
  const variance = sims.reduce((sum, s) => sum + (s - mean) ** 2, 0) / sims.length;
  const stddev = Math.sqrt(variance);

  // If the spread is tiny, all runs are effectively the same cluster.
  // Set thresholds below the minimum so nothing gets flagged.
  if (stddev < 0.05) {
    return {
      blueThreshold: minSim,
      cyanThreshold: minSim - 0.01,
    };
  }

  return {
    blueThreshold: median,
    cyanThreshold: median - 1.5 * stddev,
  };
}

function getSimilarityColor(
  similarity: number,
  classification: RunClassification,
  blueThreshold: number,
  cyanThreshold: number,
): string {
  if (classification === 'failure') return '#dc2626';
  if (similarity >= blueThreshold) return '#1d4ed8';
  if (similarity >= cyanThreshold) return '#06b6d4'; 
  return '#dc2626';                                    
}

const SimilarityScatterplot: React.FC<Props> = ({ points, selectedIndex, onSelectRun }) => {
  if (points.length === 0) return null;

  const { blueThreshold, cyanThreshold } = computeSimilarityTiers(points);

  const data = points.map(p => ({
    ...p,
    label: String(p.index + 1),
    fill: getSimilarityColor(p.similarity, p.classification, blueThreshold, cyanThreshold),
  }));

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title scatter__title">Similarity Scatterplot</h3>
        <span className="field__hint scatter__hint">Clustered = similar, distant = outlier</span>
      </div>
      <div className="card__body scatter__chart-body">
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 10, left: 10 }}>
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, 1]}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={(props) => {
                const payload = props.payload as unknown as ReadonlyArray<{ payload: RunPoint & { label: string } }> | undefined;
                if (!payload || payload.length === 0) return null;
                const point = payload[0].payload as RunPoint & { label: string };
                return (
                  <div className="scatter__tooltip">
                    <div className="scatter__tooltip-title">Run {point.label}</div>
                    <div className="scatter__tooltip-detail">
                      Avg similarity: {(point.similarity * 100).toFixed(0)}%
                    </div>
                    <div className="scatter__tooltip-classification">
                      {point.classification.replace(/_/g, ' ')}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter
              data={data}
              onClick={(entry) => {
                const point = entry as { index?: number } | undefined;
                if (typeof point?.index === 'number') {
                  onSelectRun(point.index);
                }
              }}
            >
              {data.map((point) => {
                const isSelected = point.index === selectedIndex;
                return (
                  <Cell
                    key={point.index}
                    fill={point.fill}
                    stroke={isSelected ? '#1a1a2e' : '#fff'}
                    strokeWidth={isSelected ? 2.5 : 1}
                    r={isSelected ? 8 : 6}
                    style={{ cursor: 'pointer' }}
                  />
                );
              })}
              <LabelList
                dataKey="label"
                position="top"
                offset={8}
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  fill: '#6b7280',
                  fontFamily: '"JetBrains Mono", monospace',
                }}
              />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <div className="scatter__legend">
          <span><span className="scatter__legend-dot--blue">●</span> Similar (cluster)</span>
          <span><span className="scatter__legend-dot--cyan">●</span> Moderate</span>
          <span><span className="scatter__legend-dot--red">●</span> Outlier</span>
        </div>
      </div>
    </div>
  );
};

export default SimilarityScatterplot;
