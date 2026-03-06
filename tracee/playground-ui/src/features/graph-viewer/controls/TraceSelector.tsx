import { useEffect, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { fetchTraceSummary, fetchTraces } from "../../../api/traces";
import { useLayer } from "../../../context/LayerContext";
import type { GraphEdgeData, GraphNodeData } from "../../../types/node-data";
import type { TraceMetadata, TraceSummary } from "../../../types/trace";
import iconTraces from "../../../assets/icon-traces.svg";
import { TraceMinimapPreview } from "./TraceMinimapPreview";

interface TraceSelectorProps {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
}

function formatDateTime(value?: string | null): string {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function calculateLatency(created: string, updated: string): string {
  const c = new Date(created).getTime();
  const u = new Date(updated).getTime();
  if (Number.isNaN(c) || Number.isNaN(u)) return "n/a";
  const diff = u - c;
  if (diff < 1000) return `${diff}ms`;
  return `${(diff / 1000).toFixed(2)}s`;
}

export function TraceSelector({ nodes, edges }: TraceSelectorProps) {
  const { layer, selectedTraceId, setSelectedTraceId } = useLayer();
  const [traces, setTraces] = useState<TraceMetadata[]>([]);
  const [summaries, setSummaries] = useState<Record<string, TraceSummary>>({});

  useEffect(() => {
    if (layer !== "execution") {
      setTraces([]);
      return;
    }
    let cancelled = false;
    fetchTraces()
      .then((items) => {
        if (cancelled) return;
        const sorted = [...items].sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
        );
        setTraces(sorted);
      })
      .catch(() => {
        if (!cancelled) setTraces([]);
      });

    return () => {
      cancelled = true;
    };
  }, [layer]);

  useEffect(() => {
    if (layer !== "execution") {
      setSummaries({});
      return;
    }
    if (traces.length === 0) {
      setSummaries({});
      return;
    }

    let cancelled = false;
    setSummaries({});
    for (const trace of traces) {
      fetchTraceSummary(trace.trace_id)
        .then((summary) => {
          if (cancelled) return;
          setSummaries((current) => ({ ...current, [trace.trace_id]: summary }));
        })
        .catch(() => {});
    }

    return () => {
      cancelled = true;
    };
  }, [layer, traces]);

  useEffect(() => {
    if (layer !== "execution") return;
    if (traces.length === 0) {
      if (selectedTraceId !== null) setSelectedTraceId(null);
      return;
    }
    if (!selectedTraceId || !traces.some((trace) => trace.trace_id === selectedTraceId)) {
      setSelectedTraceId(traces[0].trace_id);
    }
  }, [layer, traces, selectedTraceId, setSelectedTraceId]);

  if (layer !== "execution") return null;

  return (
    <section className="trace-selector-card">
      <header className="trace-selector-card__header">
        <h3 className="trace-selector-card__title">
          <img src={iconTraces} alt="" className="trace-selector-card__title-icon" aria-hidden />
          Execution Traces
        </h3>
      </header>
      <div className="trace-selector-card__list">
        {traces.length === 0 ? (
          <div className="trace-selector-card__empty">no traces found</div>
        ) : (
          traces.map((t) => {
            const isSelected = selectedTraceId === t.trace_id;
            return (
              <button
                key={t.trace_id}
                type="button"
                className={`trace-selector-item ${isSelected ? "is-selected" : ""}`}
                onClick={() => setSelectedTraceId(t.trace_id)}
              >
                <div className="trace-selector-item__header">
                  <span className="trace-selector-item__id" title={t.trace_id}>
                    {t.trace_id.slice(0, 8)}
                  </span>
                  <span className="trace-selector-item__events">
                    {t.event_count} events
                  </span>
                </div>
                <div className="trace-selector-item__details">
                  <span className="trace-selector-item__time">
                    {formatDateTime(t.created_at)}
                  </span>
                  <span className="trace-selector-item__latency">
                    {calculateLatency(t.created_at, t.updated_at)}
                  </span>
                </div>
                <TraceMinimapPreview nodes={nodes} edges={edges} summary={summaries[t.trace_id]} />
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}


