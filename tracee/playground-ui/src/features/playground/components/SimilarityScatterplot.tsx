import React from 'react';
import * as d3 from 'd3';
import type { ScatterPoint } from '../../../hooks/useRunAnalysis';
import VisualizationLegend, { type VisualizationLegendEntry } from './VisualizationLegend';
import { formatVisualizationGroupLabel, spreadScatterLayoutPoints } from './visualizationUtils';

interface Props {
  points: ScatterPoint[];
  selectedIndex: string | null;
  onSelectRun: (index: string | null) => void;
  title?: string;
  hint?: string;
  summary?: string;
}

interface HoveredPoint {
  point: ScatterPoint;
  left: number;
  top: number;
}

const CHART_HEIGHT = 420;
const CHART_PADDING = 32;

const SimilarityScatterplot: React.FC<Props> = ({
  points,
  selectedIndex,
  onSelectRun,
  title = 'Output map',
  hint = 'embedding-based projection',
  summary,
}) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = React.useState(720);
  const [hoveredPoint, setHoveredPoint] = React.useState<HoveredPoint | null>(null);
  const legendItems = React.useMemo<VisualizationLegendEntry[]>(() => {
    const toneEntries = new Map<string, VisualizationLegendEntry>();
    points.forEach((point) => {
      if (point.isAnchor) {
        return;
      }
      if (!toneEntries.has(point.groupId)) {
        toneEntries.set(point.groupId, {
          id: point.groupId,
          tone: point.groupTone,
          label: formatVisualizationGroupLabel(point.groupLabel, point.groupVersionId),
        });
      }
    });

    const entries = Array.from(toneEntries.values()).sort((left, right) => {
      if (left.tone === right.tone) {
        return left.label.localeCompare(right.label);
      }
      return left.tone === 'primary' ? -1 : 1;
    });

    if (points.some((point) => point.isAnchor)) {
      entries.push({
        id: 'anchor',
        tone: 'anchor',
        label: 'Anchor',
      });
    }

    return entries;
  }, [points]);

  React.useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return undefined;
    }

    const nextWidth = node.getBoundingClientRect().width;
    if (nextWidth) {
      setWidth(nextWidth);
    }

    const observer = new ResizeObserver((entries) => {
      const observedWidth = entries[0]?.contentRect.width;
      if (!observedWidth) {
        return;
      }

      setWidth(observedWidth);
    });

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const xScale = React.useMemo(
    () => d3.scaleLinear().domain([0, 1]).range([CHART_PADDING, Math.max(CHART_PADDING + 1, width - CHART_PADDING)]),
    [width],
  );
  const yScale = React.useMemo(
    () => d3.scaleLinear().domain([0, 1]).range([CHART_HEIGHT - CHART_PADDING, CHART_PADDING]),
    [],
  );
  const plottedPoints = React.useMemo(
    () => spreadScatterLayoutPoints(
      points.map((point) => ({
        ...point,
        cx: xScale(point.x),
        cy: yScale(point.y),
      })),
      {
        minimumDistance: 14,
        maxOffset: 20,
      },
    ),
    [points, xScale, yScale],
  );

  React.useEffect(() => {
    setHoveredPoint(null);
  }, [points]);

  if (points.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title scatter__title">{title}</h3>
        <span className="field__hint scatter__hint">{hint}</span>
      </div>
      <div className="card__body scatter__chart-body">
        {summary && <div className="field__hint scatter__summary">{summary}</div>}
        <div
          ref={containerRef}
          className="scatter__surface"
          role="group"
          aria-label="Projected output positions"
        >
          <VisualizationLegend
            items={legendItems}
            className="scatter__legend scatter__legend--overlay"
          />
          <svg
            className="scatter__svg"
            viewBox={`0 0 ${width} ${CHART_HEIGHT}`}
          >
            <rect
              x={CHART_PADDING / 2}
              y={CHART_PADDING / 2}
              width={Math.max(0, width - CHART_PADDING)}
              height={CHART_HEIGHT - CHART_PADDING}
              rx={20}
              className="scatter__frame"
              aria-hidden="true"
            />
            {plottedPoints.map((point) => {
              const { cx, cy } = point;
              const isSelected = point.selectionId === selectedIndex;
              const radius = point.isAnchor ? 9 : isSelected ? 7.5 : 6;

              return (
                <circle
                  key={point.id}
                  cx={cx}
                  cy={cy}
                  r={radius}
                  className={`scatter__point scatter__point--${point.groupTone}${point.isAnchor ? ' scatter__point--anchor' : ''}${isSelected ? ' scatter__point--selected' : ''}`}
                  role={point.selectionId ? 'button' : undefined}
                  tabIndex={point.selectionId ? 0 : undefined}
                  aria-label={point.isAnchor ? 'Anchor point' : `Select run ${point.label}`}
                  onMouseEnter={() => setHoveredPoint({
                    point,
                    left: cx,
                    top: cy,
                  })}
                  onMouseLeave={() => setHoveredPoint((current) => (
                    current?.point.id === point.id ? null : current
                  ))}
                  onClick={() => {
                    if (point.selectionId) {
                      onSelectRun(point.selectionId === selectedIndex ? null : point.selectionId);
                    }
                  }}
                  onKeyDown={(event) => {
                    if ((event.key === 'Enter' || event.key === ' ') && point.selectionId) {
                      event.preventDefault();
                      onSelectRun(point.selectionId === selectedIndex ? null : point.selectionId);
                    }
                  }}
                />
              );
            })}
          </svg>
          {hoveredPoint && (
            <div
              className="scatter__tooltip"
              style={{
                left: hoveredPoint.left,
                top: hoveredPoint.top,
              }}
            >
              <div className="scatter__tooltip-title">
                {hoveredPoint.point.isAnchor ? 'Anchor' : hoveredPoint.point.label}
              </div>
              {!hoveredPoint.point.isAnchor && (
                <div className="scatter__tooltip-detail">
                  {formatVisualizationGroupLabel(hoveredPoint.point.groupLabel, hoveredPoint.point.groupVersionId)}
                </div>
              )}
              {hoveredPoint.point.isFailed && (
                <div className="scatter__tooltip-detail">failed run</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimilarityScatterplot;
