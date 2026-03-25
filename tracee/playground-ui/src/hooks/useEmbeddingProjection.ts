import React from 'react';
import { playgroundAPI } from '../services/api';
import type { ProjectionItem, ScatterPoint } from './useRunAnalysis';

interface ProjectionState {
  loading: boolean;
  points: ScatterPoint[];
  error: string | null;
}

export function useEmbeddingProjection(items: ProjectionItem[]): ProjectionState {
  const [state, setState] = React.useState<ProjectionState>({
    loading: false,
    points: [],
    error: null,
  });
  const requestIdRef = React.useRef(0);

  React.useEffect(() => {
    const activeItems = items.filter((item) => item.output.trim());
    if (activeItems.length < 2) {
      setState({
        loading: false,
        points: [],
        error: null,
      });
      return;
    }

    let cancelled = false;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setState((current) => ({
      ...current,
      loading: true,
      error: null,
    }));

    playgroundAPI.analyzeOutputs({
      items: activeItems.map((item) => ({
        id: item.id,
        group_id: item.groupId,
        label: item.label,
        output: item.output,
      })),
    }).then((response) => {
      if (cancelled || requestIdRef.current !== requestId) {
        return;
      }

      const points = response.points.map((point) => {
        const item = activeItems.find((entry) => entry.id === point.id);
        return {
          id: point.id,
          x: point.x,
          y: point.y,
          selectionId: item?.selectionId ?? null,
          similarity: point.average_similarity,
          groupId: point.group_id,
          groupLabel: item?.groupLabel ?? point.group_id,
          groupVersionId: item?.groupVersionId ?? null,
          groupTone: item?.groupTone ?? 'primary',
          isAnchor: item?.kind === 'anchor',
          isFailed: item?.isFailed ?? false,
          label: item?.label ?? point.label,
        } satisfies ScatterPoint;
      });

      setState({
        loading: false,
        points,
        error: null,
      });
    }).catch(() => {
      if (cancelled || requestIdRef.current !== requestId) {
        return;
      }

      setState({
        loading: false,
        points: [],
        error: 'Embedding analysis failed. Try running the projection again.',
      });
    });

    return () => {
      cancelled = true;
    };
  }, [items]);

  return state;
}
