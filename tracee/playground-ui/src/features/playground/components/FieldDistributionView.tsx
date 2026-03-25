import React from 'react';
import * as d3 from 'd3';
import type { FieldOption, FieldValueEntry } from '../../../hooks/useRunAnalysis';
import VisualizationLegend from './VisualizationLegend';
import { formatVisualizationGroupLabel } from './visualizationUtils';

interface Props {
  field: FieldOption;
  values: FieldValueEntry[];
}

interface GroupSummary {
  id: string;
  label: string;
  displayLabel: string;
  tone: 'primary' | 'compare';
  values: FieldValueEntry[];
}

const CHART_HEIGHT = 320;
const CHART_PADDING_TOP = 28;
const CHART_PADDING_RIGHT = 24;
const CHART_PADDING_BOTTOM = 52;
const CHART_PADDING_LEFT = 56;

function getGroupColor(tone: 'primary' | 'compare') {
  return tone === 'compare' ? 'var(--playground-compare-fill)' : 'var(--playground-primary-fill)';
}

function formatNumericTick(value: number) {
  return d3.format('~g')(value);
}

function getPlotBottom() {
  return CHART_HEIGHT - CHART_PADDING_BOTTOM;
}

function getCountTicks(maxCount: number) {
  return Array.from(new Set([
    0,
    ...d3.ticks(0, maxCount, Math.min(4, maxCount)),
    maxCount,
  ].map((value) => Math.round(value))))
    .sort((left, right) => left - right);
}

function getGroups(values: FieldValueEntry[]): GroupSummary[] {
  const groupMap = new Map<string, GroupSummary>();
  values.forEach((entry) => {
    const current = groupMap.get(entry.groupId);
    if (current) {
      current.values.push(entry);
      return;
    }
    groupMap.set(entry.groupId, {
      id: entry.groupId,
      label: entry.groupLabel,
      displayLabel: formatVisualizationGroupLabel(entry.groupLabel, entry.groupVersionId),
      tone: entry.groupTone,
      values: [entry],
    });
  });
  return Array.from(groupMap.values());
}

function createBins(values: number[], domain: [number, number], count = 8) {
  return d3.bin()
    .domain(domain)
    .thresholds(count)(values);
}

const FieldDistributionView: React.FC<Props> = ({ field, values }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = React.useState(720);
  const groups = React.useMemo(() => getGroups(values), [values]);
  const legendItems = React.useMemo(
    () => groups.map((group) => ({
      id: group.id,
      label: group.displayLabel,
      tone: group.tone,
    })),
    [groups],
  );

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
      if (observedWidth) {
        setWidth(observedWidth);
      }
    });

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const chartWidth = Math.max(width, 320);
  const plotBottom = getPlotBottom();
  const plotRight = chartWidth - CHART_PADDING_RIGHT;

  if (values.length === 0) {
    return (
      <div className="card">
        <div className="empty-state create-run__empty-body">
          <div className="empty-state__title">No field values yet</div>
          <div className="empty-state__desc">
            Run the prompt with structured output to inspect this field.
          </div>
        </div>
      </div>
    );
  }

  if (field.type === 'boolean') {
    const valueLegend = [
      { id: 'true', label: 'true', className: 'field-viz__value-swatch--true' },
      { id: 'false', label: 'false', className: 'field-viz__value-swatch--false' },
    ];

    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title scatter__title">{field.label}</h3>
          <span className="field__hint scatter__hint">boolean share by run group</span>
        </div>
        <div className="card__body field-viz__card-body">
          <div className="field-viz__value-legend" aria-label="Boolean value legend">
            {valueLegend.map((item) => (
              <span key={item.id} className="field-viz__value-legend-item">
                <span className={`field-viz__value-swatch ${item.className}`} aria-hidden />
                {item.label}
              </span>
            ))}
          </div>
          <div className="field-viz__boolean-list">
          {groups.map((group) => {
            const trueCount = group.values.filter((entry) => entry.value === true).length;
            const falseCount = group.values.length - trueCount;
            const trueRatio = group.values.length > 0 ? trueCount / group.values.length : 0;
            const falseRatio = group.values.length > 0 ? falseCount / group.values.length : 0;
            return (
              <div key={group.id} className="field-viz__boolean-row">
                <div className="field-viz__row-head">
                  <span className={`field-viz__group-dot field-viz__group-dot--${group.tone}`} aria-hidden />
                  <span>{group.displayLabel}</span>
                  <span className="field__hint">{trueCount} true / {falseCount} false</span>
                </div>
                <div
                  className="field-viz__boolean-bar"
                  aria-label={`${group.displayLabel}: ${trueCount} true, ${falseCount} false`}
                  role="img"
                >
                  <div
                    className="field-viz__boolean-segment field-viz__boolean-segment--true"
                    style={{ width: `${trueRatio * 100}%` }}
                  />
                  <div
                    className="field-viz__boolean-segment field-viz__boolean-segment--false"
                    style={{ width: `${falseRatio * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
          </div>
        </div>
      </div>
    );
  }

  if (field.type === 'string') {
    const categories = Array.from(
      values.reduce((counts, entry) => {
        const key = String(entry.value);
        counts.set(key, (counts.get(key) ?? 0) + 1);
        return counts;
      }, new Map<string, number>()).entries()
    )
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([category]) => category);
    const xScale = d3.scaleBand()
      .domain(categories)
      .range([CHART_PADDING_LEFT, plotRight])
      .padding(0.18);
    const innerScale = d3.scaleBand()
      .domain(groups.map((group) => group.id))
      .range([0, xScale.bandwidth()])
      .padding(0.12);
    const categoryCounts = categories.map((category) => ({
      category,
      counts: groups.map((group) => ({
        groupId: group.id,
        count: group.values.filter((entry) => String(entry.value) === category).length,
        tone: group.tone,
      })),
    }));
    const maxCount = Math.max(1, ...categoryCounts.flatMap((entry) => entry.counts.map((count) => count.count)));
    const yScale = d3.scaleLinear()
      .domain([0, maxCount])
      .range([plotBottom, CHART_PADDING_TOP]);
    const countTicks = getCountTicks(maxCount);

    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title scatter__title">{field.label}</h3>
          <span className="field__hint scatter__hint">top exact string matches</span>
        </div>
        <div ref={containerRef} className="card__body field-viz__card-body">
          <VisualizationLegend items={legendItems} />
          <div className="field-viz__surface">
          <svg className="field-viz__svg" viewBox={`0 0 ${chartWidth} ${CHART_HEIGHT}`}>
            <rect
              x={CHART_PADDING_LEFT}
              y={CHART_PADDING_TOP}
              width={Math.max(0, plotRight - CHART_PADDING_LEFT)}
              height={plotBottom - CHART_PADDING_TOP}
              rx={18}
              className="field-viz__frame"
            />
            {countTicks.map((tick) => {
              const y = yScale(tick);
              return (
                <g key={`tick-${tick}`}>
                  <line
                    x1={CHART_PADDING_LEFT}
                    x2={plotRight}
                    y1={y}
                    y2={y}
                    className="field-viz__grid-line"
                  />
                  <text
                    x={CHART_PADDING_LEFT - 10}
                    y={y + 4}
                    textAnchor="end"
                    className="field-viz__axis-label"
                  >
                    {tick}
                  </text>
                </g>
              );
            })}
            {categoryCounts.map((entry) => (
              entry.counts.map((count) => {
                const x = (xScale(entry.category) ?? CHART_PADDING_LEFT) + (innerScale(count.groupId) ?? 0);
                const y = yScale(count.count);
                const barHeight = plotBottom - y;
                return (
                  <rect
                    key={`${entry.category}-${count.groupId}`}
                    x={x}
                    y={y}
                    width={innerScale.bandwidth()}
                    height={Math.max(barHeight, 2)}
                    rx={8}
                    fill={getGroupColor(count.tone)}
                  />
                );
              })
            ))}
            {categories.map((category) => (
              <g key={category}>
                <text
                  x={(xScale(category) ?? CHART_PADDING_LEFT) + xScale.bandwidth() / 2}
                  y={CHART_HEIGHT - 14}
                  textAnchor="middle"
                  className="field-viz__axis-label"
                >
                  {category.slice(0, 16)}
                </text>
                <title>{category}</title>
              </g>
            ))}
            <line
              x1={CHART_PADDING_LEFT}
              x2={plotRight}
              y1={plotBottom}
              y2={plotBottom}
              className="field-viz__axis-line"
            />
            <text
              x={(CHART_PADDING_LEFT + plotRight) / 2}
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              className="field-viz__axis-caption"
            >
              exact value
            </text>
            <text
              x={18}
              y={(CHART_PADDING_TOP + plotBottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 18 ${(CHART_PADDING_TOP + plotBottom) / 2})`}
              className="field-viz__axis-caption"
            >
              runs
            </text>
          </svg>
          </div>
        </div>
      </div>
    );
  }

  if (field.type === 'number') {
    const numericGroups = groups.map((group) => ({
      ...group,
      numericValues: group.values
        .map((entry) => entry.value)
        .filter((value): value is number => typeof value === 'number'),
    }));
    const allValues = numericGroups.flatMap((group) => group.numericValues);
    if (allValues.length === 0) {
      return (
        <div className="card">
          <div className="empty-state create-run__empty-body">
            <div className="empty-state__title">No numeric values yet</div>
            <div className="empty-state__desc">
              This field does not have numeric samples in the current run set.
            </div>
          </div>
        </div>
      );
    }
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const domain: [number, number] = min === max ? [min - 1, max + 1] : [min, max];
    const xScale = d3.scaleLinear()
      .domain(domain)
      .range([CHART_PADDING_LEFT, plotRight]);
    const binsByGroup = numericGroups.map((group) => ({
      ...group,
      bins: createBins(group.numericValues, domain),
    }));
    const maxBinCount = Math.max(1, ...binsByGroup.flatMap((group) => group.bins.map((bin) => bin.length)));
    const yScale = d3.scaleLinear()
      .domain([0, maxBinCount])
      .range([plotBottom, CHART_PADDING_TOP]);
    const countTicks = getCountTicks(maxBinCount);
    const valueTicks = d3.ticks(domain[0], domain[1], 5);

    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title scatter__title">{field.label}</h3>
          <span className="field__hint scatter__hint">overlaid histogram for numeric values</span>
        </div>
        <div ref={containerRef} className="card__body field-viz__card-body">
          <VisualizationLegend items={legendItems} />
          <div className="field-viz__surface">
          <svg className="field-viz__svg" viewBox={`0 0 ${chartWidth} ${CHART_HEIGHT}`}>
            <rect
              x={CHART_PADDING_LEFT}
              y={CHART_PADDING_TOP}
              width={Math.max(0, plotRight - CHART_PADDING_LEFT)}
              height={plotBottom - CHART_PADDING_TOP}
              rx={18}
              className="field-viz__frame"
            />
            {countTicks.map((tick) => {
              const y = yScale(tick);
              return (
                <g key={`count-${tick}`}>
                  <line
                    x1={CHART_PADDING_LEFT}
                    x2={plotRight}
                    y1={y}
                    y2={y}
                    className="field-viz__grid-line"
                  />
                  <text
                    x={CHART_PADDING_LEFT - 10}
                    y={y + 4}
                    textAnchor="end"
                    className="field-viz__axis-label"
                  >
                    {tick}
                  </text>
                </g>
              );
            })}
            {binsByGroup.map((group) => group.bins.map((bin, index) => {
              const x0 = xScale(bin.x0 ?? domain[0]);
              const x1 = xScale(bin.x1 ?? domain[1]);
              const y = yScale(bin.length);
              return (
                <rect
                  key={`${group.id}-${index}`}
                  x={x0}
                  y={y}
                  width={Math.max(x1 - x0 - 2, 3)}
                  height={plotBottom - y}
                  rx={6}
                  fill={getGroupColor(group.tone)}
                  opacity={0.5}
                />
              );
            }))}
            <line
              x1={CHART_PADDING_LEFT}
              x2={plotRight}
              y1={plotBottom}
              y2={plotBottom}
              className="field-viz__axis-line"
            />
            {valueTicks.map((tick) => (
              <text
                key={`value-${tick}`}
                x={xScale(tick)}
                y={CHART_HEIGHT - 14}
                textAnchor="middle"
                className="field-viz__axis-label"
              >
                {formatNumericTick(tick)}
              </text>
            ))}
            <text
              x={(CHART_PADDING_LEFT + plotRight) / 2}
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              className="field-viz__axis-caption"
            >
              value
            </text>
            <text
              x={18}
              y={(CHART_PADDING_TOP + plotBottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 18 ${(CHART_PADDING_TOP + plotBottom) / 2})`}
              className="field-viz__axis-caption"
            >
              runs
            </text>
          </svg>
          </div>
        </div>
      </div>
    );
  }

  const arrayGroups = groups.map((group) => {
    const arrays = group.values
      .map((entry) => entry.value)
      .filter((value): value is unknown[] => Array.isArray(value));
    const lengths = arrays.map((value) => value.length);
    const topItems = new Map<string, number>();

    arrays.forEach((items) => {
      items.forEach((item) => {
        if (typeof item !== 'string' && typeof item !== 'number' && typeof item !== 'boolean') {
          return;
        }
        const key = String(item);
        topItems.set(key, (topItems.get(key) ?? 0) + 1);
      });
    });

    return {
      ...group,
      lengths,
      topItems: Array.from(topItems.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5),
    };
  });
  const allLengths = arrayGroups.flatMap((group) => group.lengths);
  const minLength = Math.min(...allLengths, 0);
  const maxLength = Math.max(...allLengths, 1);
  const domain: [number, number] = minLength === maxLength ? [minLength, maxLength + 1] : [minLength, maxLength];
  const xScale = d3.scaleLinear()
    .domain(domain)
    .range([CHART_PADDING_LEFT, plotRight]);
  const binsByGroup = arrayGroups.map((group) => ({
    ...group,
    bins: createBins(group.lengths, domain, Math.min(8, Math.max(3, domain[1] - domain[0] + 1))),
  }));
  const maxBinCount = Math.max(1, ...binsByGroup.flatMap((group) => group.bins.map((bin) => bin.length)));
  const yScale = d3.scaleLinear()
    .domain([0, maxBinCount])
    .range([plotBottom, CHART_PADDING_TOP]);
  const countTicks = getCountTicks(maxBinCount);
  const lengthTicks = d3.ticks(domain[0], domain[1], 5);

  return (
    <div className="field-viz__stack">
      <div className="card">
        <div className="card__header">
          <h3 className="card__title scatter__title">{field.label}</h3>
          <span className="field__hint scatter__hint">array length distribution</span>
        </div>
        <div ref={containerRef} className="card__body field-viz__card-body">
          <VisualizationLegend items={legendItems} />
          <div className="field-viz__surface">
          <svg className="field-viz__svg" viewBox={`0 0 ${chartWidth} ${CHART_HEIGHT}`}>
            <rect
              x={CHART_PADDING_LEFT}
              y={CHART_PADDING_TOP}
              width={Math.max(0, plotRight - CHART_PADDING_LEFT)}
              height={plotBottom - CHART_PADDING_TOP}
              rx={18}
              className="field-viz__frame"
            />
            {countTicks.map((tick) => {
              const y = yScale(tick);
              return (
                <g key={`length-count-${tick}`}>
                  <line
                    x1={CHART_PADDING_LEFT}
                    x2={plotRight}
                    y1={y}
                    y2={y}
                    className="field-viz__grid-line"
                  />
                  <text
                    x={CHART_PADDING_LEFT - 10}
                    y={y + 4}
                    textAnchor="end"
                    className="field-viz__axis-label"
                  >
                    {tick}
                  </text>
                </g>
              );
            })}
            {binsByGroup.map((group) => group.bins.map((bin, index) => {
              const x0 = xScale(bin.x0 ?? domain[0]);
              const x1 = xScale(bin.x1 ?? domain[1]);
              const y = yScale(bin.length);
              return (
                <rect
                  key={`${group.id}-${index}`}
                  x={x0}
                  y={y}
                  width={Math.max(x1 - x0 - 2, 3)}
                  height={plotBottom - y}
                  rx={6}
                  fill={getGroupColor(group.tone)}
                  opacity={0.5}
                />
              );
            }))}
            <line
              x1={CHART_PADDING_LEFT}
              x2={plotRight}
              y1={plotBottom}
              y2={plotBottom}
              className="field-viz__axis-line"
            />
            {lengthTicks.map((tick) => (
              <text
                key={`length-${tick}`}
                x={xScale(tick)}
                y={CHART_HEIGHT - 14}
                textAnchor="middle"
                className="field-viz__axis-label"
              >
                {formatNumericTick(tick)}
              </text>
            ))}
            <text
              x={(CHART_PADDING_LEFT + plotRight) / 2}
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              className="field-viz__axis-caption"
            >
              array length
            </text>
            <text
              x={18}
              y={(CHART_PADDING_TOP + plotBottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 18 ${(CHART_PADDING_TOP + plotBottom) / 2})`}
              className="field-viz__axis-caption"
            >
              runs
            </text>
          </svg>
          </div>
        </div>
      </div>
      <div className="card">
        <div className="card__header">
          <h3 className="card__title scatter__title">Common items</h3>
          <span className="field__hint scatter__hint">flattened across array elements</span>
        </div>
        <div className="card__body field-viz__item-grid">
          {arrayGroups.map((group) => (
            <div key={group.id} className="field-viz__item-column">
              <div className="field-viz__row-head">
                <span className={`field-viz__group-dot field-viz__group-dot--${group.tone}`} aria-hidden />
                <span>{group.displayLabel}</span>
              </div>
              {group.topItems.length > 0 ? group.topItems.map(([item, count]) => (
                <div key={`${group.id}-${item}`} className="field-viz__item-row">
                  <span className="field-viz__item-label">{item}</span>
                  <span className="field__hint">{count}</span>
                </div>
              )) : (
                <div className="field__hint">no primitive array items to summarize</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FieldDistributionView;
