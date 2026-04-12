import { useEffect, useMemo, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { fetchTraceSummary, fetchTraces } from "../../../api/traces";
import { useLayer } from "../../../context/LayerContext";
import type { Layer } from "../../../context/LayerContext";
import type { GraphEdgeData, GraphNodeData, TraceOutlineItem, TraceOutlineItemKind } from "../../../types/node-data";
import type { TraceMetadata, TraceSummary } from "../../../types/trace";
import iconTraces from "../../../assets/icon-traces.svg";
import iconChain from "../../../assets/icon-chain.svg";
import cognitionIcon from "../../../assets/cognition.svg";
import iconCode from "../../../assets/icon-code.svg";
import iconCollapse from "../../../assets/icon-collapse.svg";
import iconError from "../../../assets/icon-error.svg";
import iconExpand from "../../../assets/icon-expand.svg";
import iconLlm from "../../../assets/icon-llm.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconState from "../../../assets/icon-state.svg";
import iconTool from "../../../assets/icon-tool.svg";
import { TraceMinimapPreview } from "./TraceMinimapPreview";

function isTraceLayer(layer: Layer): boolean {
  return layer === "execution" || layer === "cognition";
}

interface TraceSelectorProps {
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  graphId: string | null;
  outline: TraceOutlineItem[];
  outlineLoading: boolean;
  onOutlineSelect: (item: TraceOutlineItem) => void;
}

const OUTLINE_ICON_BY_KIND: Record<TraceOutlineItemKind, string> = {
  agent: cognitionIcon,
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  state_update: iconState,
  error: iconError,
};

const OUTLINE_COLOR_BY_KIND: Record<TraceOutlineItemKind, string> = {
  agent: "chain",
  llm_call: "llm",
  tool_call: "tool",
  rag_retrieve: "rag",
  code_exec: "code",
  subgraph_call: "chain",
  state_update: "state",
  error: "error",
};

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

function formatLatency(latencyMs?: number): string {
  if (typeof latencyMs !== "number") return "";
  if (latencyMs < 1000) return `${Math.round(latencyMs)}ms`;
  return `${(latencyMs / 1000).toFixed(1)}s`;
}

interface TraceOutlineTreeProps {
  items: TraceOutlineItem[];
  collapsedIds: Set<string>;
  onToggle: (id: string) => void;
  onSelect: (item: TraceOutlineItem) => void;
  depth?: number;
}

function TraceOutlineTree({
  items,
  collapsedIds,
  onToggle,
  onSelect,
  depth = 0,
}: TraceOutlineTreeProps) {
  return (
    <>
      {items.map((item) => {
        const isCollapsed = collapsedIds.has(item.id);
        const hasChildren = item.children.length > 0;
        const icon = OUTLINE_ICON_BY_KIND[item.kind];
        const colorClass = OUTLINE_COLOR_BY_KIND[item.kind];
        const isClickable = Boolean(item.nodeId);

        return (
          <div key={item.id} className="trace-outline__branch">
            <div
              className={`trace-outline__row trace-outline__row--${colorClass}${item.status === "error" ? " is-error" : ""}${isClickable ? " is-clickable" : ""}`}
              style={{ paddingLeft: `${12 + depth * 18}px` }}
            >
              <div className="trace-outline__row-main">
                {hasChildren ? (
                  <button
                    type="button"
                    className="trace-outline__toggle"
                    onClick={() => onToggle(item.id)}
                    aria-label={isCollapsed ? "expand branch" : "collapse branch"}
                  >
                    <img
                      src={isCollapsed ? iconExpand : iconCollapse}
                      alt=""
                      className="trace-outline__toggle-icon"
                      aria-hidden
                    />
                  </button>
                ) : (
                  <span className="trace-outline__toggle-spacer" />
                )}
                <button
                  type="button"
                  className="trace-outline__row-button"
                  onClick={() => onSelect(item)}
                  disabled={!isClickable}
                >
                  <span className={`trace-outline__icon-wrap trace-outline__icon-wrap--${colorClass}`}>
                    <img src={icon} alt="" className="trace-outline__icon" aria-hidden />
                  </span>
                  <span className="trace-outline__label">{item.label}</span>
                </button>
              </div>
              <span className="trace-outline__latency">{formatLatency(item.latencyMs)}</span>
            </div>
            {hasChildren && !isCollapsed && (
              <TraceOutlineTree
                items={item.children}
                collapsedIds={collapsedIds}
                onToggle={onToggle}
                onSelect={onSelect}
                depth={depth + 1}
              />
            )}
          </div>
        );
      })}
    </>
  );
}

export function TraceSelector({
  nodes,
  edges,
  graphId,
  outline,
  outlineLoading,
  onOutlineSelect,
}: TraceSelectorProps) {
  const { layer, selectedTraceId, setSelectedTraceId } = useLayer();
  const [traces, setTraces] = useState<TraceMetadata[]>([]);
  const [summaries, setSummaries] = useState<Record<string, TraceSummary>>({});
  const [maxHeight, setMaxHeight] = useState<number | null>(null);
  const [showTraceList, setShowTraceList] = useState(true);
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());
  const cardRef = useRef<HTMLElement | null>(null);
  const isExecutionLayer = layer === "execution";

  const selectedTrace = useMemo(
    () => traces.find((trace) => trace.trace_id === selectedTraceId) ?? null,
    [traces, selectedTraceId],
  );

  useEffect(() => {
    if (!isTraceLayer(layer)) {
      setTraces([]);
      return;
    }
    let cancelled = false;
    fetchTraces(100, 0, graphId)
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
  }, [layer, graphId]);

  useEffect(() => {
    if (!isTraceLayer(layer)) {
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
    if (!isTraceLayer(layer)) return;
    if (traces.length === 0) {
      if (selectedTraceId !== null) setSelectedTraceId(null);
      return;
    }
    if (!selectedTraceId || !traces.some((trace) => trace.trace_id === selectedTraceId)) {
      setSelectedTraceId(traces[0].trace_id);
    }
  }, [layer, traces, selectedTraceId, setSelectedTraceId]);

  useEffect(() => {
    if (!isTraceLayer(layer)) {
      setMaxHeight(null);
      return;
    }

    const element = cardRef.current;
    if (!element) return;

    const updateMaxHeight = () => {
      const { top } = element.getBoundingClientRect();
      const nextMaxHeight = Math.max(240, Math.floor(window.innerHeight - top - 16));
      setMaxHeight(nextMaxHeight);
    };

    updateMaxHeight();
    window.addEventListener("resize", updateMaxHeight);

    const observer = new ResizeObserver(() => {
      updateMaxHeight();
    });
    observer.observe(document.body);

    return () => {
      window.removeEventListener("resize", updateMaxHeight);
      observer.disconnect();
    };
  }, [layer, traces.length]);

  useEffect(() => {
    if (!isExecutionLayer) {
      setShowTraceList(true);
      return;
    }
    if (!selectedTraceId) {
      setShowTraceList(true);
      return;
    }
    setShowTraceList(false);
  }, [isExecutionLayer, selectedTraceId]);

  useEffect(() => {
    setCollapsedIds(new Set());
  }, [selectedTraceId, outline]);

  if (!isTraceLayer(layer)) return null;

  const handleTraceSelect = (traceId: string) => {
    setSelectedTraceId(traceId);
    if (isExecutionLayer) setShowTraceList(false);
  };

  const handleToggleBranch = (id: string) => {
    setCollapsedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <section
      ref={cardRef}
      className="trace-selector-card"
      style={maxHeight ? { maxHeight: `${maxHeight}px` } : undefined}
    >
      <header className="trace-selector-card__header">
        <div className="trace-selector-card__header-main">
          <h3 className="trace-selector-card__title">
            <img src={iconTraces} alt="" className="trace-selector-card__title-icon" aria-hidden />
            {isExecutionLayer && !showTraceList ? "Execution Outline" : "Execution Traces"}
          </h3>
          {isExecutionLayer && !showTraceList && selectedTrace && (
            <div className="trace-selector-card__subtitle">
              trace {selectedTrace.trace_id.slice(0, 8)}
            </div>
          )}
        </div>
        {isExecutionLayer && !showTraceList && (
          <button
            type="button"
            className="trace-selector-card__header-btn"
            onClick={() => setShowTraceList(true)}
          >
            back to traces
          </button>
        )}
      </header>
      {isExecutionLayer && !showTraceList ? (
        <div className="trace-outline">
          {outlineLoading ? (
            <div className="trace-selector-card__empty">loading execution...</div>
          ) : outline.length === 0 ? (
            <div className="trace-selector-card__empty">no operations found</div>
          ) : (
            <div className="trace-outline__list" onWheelCapture={(event) => event.stopPropagation()}>
              <div className="trace-outline__header-row">
                <span className="trace-outline__header-label">Operation</span>
                <span className="trace-outline__header-label">Latency</span>
              </div>
              <TraceOutlineTree
                items={outline}
                collapsedIds={collapsedIds}
                onToggle={handleToggleBranch}
                onSelect={onOutlineSelect}
              />
            </div>
          )}
        </div>
      ) : (
        <div className="trace-selector-card__list" onWheelCapture={(event) => event.stopPropagation()}>
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
                  onClick={() => handleTraceSelect(t.trace_id)}
                >
                  <div className="trace-selector-item__header">
                    <span className="trace-selector-item__id" title={t.trace_id}>
                      {t.trace_id.slice(0, 8)}
                    </span>
                    <div className="trace-selector-item__meta">
                      <span className="trace-selector-item__time">
                        {formatDateTime(t.created_at)}
                      </span>
                      <span className="trace-selector-item__latency">
                        {calculateLatency(t.created_at, t.updated_at)}
                      </span>
                    </div>
                  </div>
                  <TraceMinimapPreview nodes={nodes} edges={edges} summary={summaries[t.trace_id]} />
                  <div className="trace-selector-item__footer">
                    <span className="trace-selector-item__events">
                      {t.event_count} events
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      )}
    </section>
  );
}


