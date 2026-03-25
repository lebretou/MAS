interface LayoutPoint {
  id: string;
  cx: number;
  cy: number;
}

interface SpreadOptions {
  minimumDistance?: number;
  maxOffset?: number;
}

export function formatVisualizationGroupLabel(groupLabel: string, groupVersionId: string | null): string {
  return groupVersionId ? `${groupLabel}/${groupVersionId}` : groupLabel;
}

function distance(left: LayoutPoint, right: LayoutPoint): number {
  return Math.hypot(left.cx - right.cx, left.cy - right.cy);
}

export function spreadScatterLayoutPoints<T extends LayoutPoint>(
  points: T[],
  options: SpreadOptions = {},
): T[] {
  const minimumDistance = options.minimumDistance ?? 12;
  const maxOffset = options.maxOffset ?? 18;
  const clusters: T[][] = [];

  points.forEach((point) => {
    const cluster = clusters.find((entry) => entry.some((candidate) => distance(point, candidate) < minimumDistance));
    if (cluster) {
      cluster.push(point);
      return;
    }

    clusters.push([point]);
  });

  return clusters.flatMap((cluster) => {
    if (cluster.length === 1) {
      return cluster;
    }

    const radius = Math.min(maxOffset, Math.max(8, Math.ceil(cluster.length / 2) * 4));
    const sortedCluster = [...cluster].sort((left, right) => left.id.localeCompare(right.id));

    return sortedCluster.map((point, index) => {
      const angle = (Math.PI * 2 * index) / sortedCluster.length;
      return {
        ...point,
        cx: point.cx + Math.cos(angle) * radius,
        cy: point.cy + Math.sin(angle) * radius,
      };
    });
  });
}
