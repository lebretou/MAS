import { describe, expect, it } from 'vitest';

import { formatVisualizationGroupLabel, spreadScatterLayoutPoints } from './visualizationUtils';

describe('formatVisualizationGroupLabel', () => {
  it('combines the prompt name and version id', () => {
    expect(formatVisualizationGroupLabel('draft prompt', 'draft-v5')).toBe('draft prompt/draft-v5');
  });

  it('falls back to the prompt name when the version is missing', () => {
    expect(formatVisualizationGroupLabel('draft prompt', null)).toBe('draft prompt');
  });
});

describe('spreadScatterLayoutPoints', () => {
  it('spreads points that share the same plotted position', () => {
    const spread = spreadScatterLayoutPoints([
      { id: 'a', cx: 100, cy: 100 },
      { id: 'b', cx: 100, cy: 100 },
      { id: 'c', cx: 100, cy: 100 },
    ]);

    expect(new Set(spread.map((point) => `${point.cx}:${point.cy}`)).size).toBe(3);
    expect(spread.some((point) => point.cx !== 100 || point.cy !== 100)).toBe(true);
  });

  it('keeps distant points in place', () => {
    const spread = spreadScatterLayoutPoints([
      { id: 'a', cx: 100, cy: 100 },
      { id: 'b', cx: 140, cy: 140 },
    ]);

    expect(spread).toEqual([
      { id: 'a', cx: 100, cy: 100 },
      { id: 'b', cx: 140, cy: 140 },
    ]);
  });
});
