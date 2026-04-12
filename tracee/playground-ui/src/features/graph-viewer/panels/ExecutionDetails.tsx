import { useEffect, useRef, useState } from "react";
import type { AgentOperation } from "../../../types/node-data";
import type { GraphNodeData } from "../../../types/node-data";
import { StateDiffView } from "../../../components/StateDiffView";
import iconCode from "../../../assets/icon-code.svg";
import iconError from "../../../assets/icon-error.svg";
import iconLlm from "../../../assets/icon-llm.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconState from "../../../assets/icon-state.svg";
import iconTool from "../../../assets/icon-tool.svg";
import iconChain from "../../../assets/icon-chain.svg";
import { useSidebar } from "../../../context/SidebarContext";

const operationIconMap: Record<AgentOperation["type"], string> = {
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  state_update: iconState,
  error: iconError,
};

// design-sheet palette mapped to each operation type
const operationColorMap: Record<AgentOperation["type"], string> = {
  llm_call: "llm",
  tool_call: "tool",
  rag_retrieve: "rag",
  code_exec: "code",
  subgraph_call: "chain",
  state_update: "state",
  error: "error",
};

const MIN_SEGMENT_PX = 48;
const LABEL_FIT_THRESHOLD = 100;

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

interface Props {
  node: GraphNodeData;
}

function truncateText(value: string, maxChars = 6000): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}\n... truncated (${value.length - maxChars} chars omitted)`;
}

function sanitizeForDisplay(value: unknown, depth = 0): unknown {
  if (value == null) return value;
  if (typeof value === "string") return truncateText(value, 4000);
  if (typeof value !== "object") return value;
  if (depth >= 3) return "[max depth reached]";
  if (Array.isArray(value)) {
    const capped = value.slice(0, 30).map((item) => sanitizeForDisplay(item, depth + 1));
    if (value.length > 30) capped.push(`... ${value.length - 30} more items`);
    return capped;
  }
  const entries = Object.entries(value as Record<string, unknown>);
  const out: Record<string, unknown> = {};
  for (const [index, [key, val]] of entries.entries()) {
    if (index >= 40) {
      out.__truncated__ = `${entries.length - 40} more keys`;
      break;
    }
    out[key] = sanitizeForDisplay(val, depth + 1);
  }
  return out;
}

// unescape literal \n, \t, \r in strings for display
function unescapeForDisplay(s: string): string {
  return s
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\r/g, "\r")
    .replace(/\\"/g, '"');
}

function formatForLog(value: unknown, options?: { full?: boolean }): { empty: boolean; text: string } {
  const isFull = options?.full === true;
  if (value == null) return { empty: true, text: "" };
  if (value === "") return { empty: true, text: "" };
  if (Array.isArray(value) && value.length === 0) return { empty: true, text: "" };
  if (
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value as object).length === 0
  )
    return { empty: true, text: "" };

  if (typeof value === "string") {
    const displayValue = isFull ? value : truncateText(value, 4000);
    const trimmed = value.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(trimmed);
      } catch {
        return { empty: false, text: unescapeForDisplay(displayValue) };
      }
      return { empty: false, text: unescapeForDisplay(JSON.stringify(parsed, null, 2)) };
    }
    return { empty: false, text: unescapeForDisplay(displayValue) };
  }
  const text = unescapeForDisplay(
    isFull
      ? JSON.stringify(value, null, 2)
      : JSON.stringify(sanitizeForDisplay(value), null, 2),
  );
  return { empty: false, text };
}

function computeSegmentWidths(operations: AgentOperation[], totalPx: number): number[] {
  if (operations.length === 0) return [];
  const durations = operations.map((op) => op.latencyMs ?? 1);
  const totalDuration = durations.reduce((sum, d) => sum + d, 0);
  if (totalDuration === 0) return durations.map(() => totalPx / operations.length);

  const minTotal = MIN_SEGMENT_PX * operations.length;
  const available = Math.max(totalPx, minTotal);

  // first pass: proportional
  const raw = durations.map((d) => (d / totalDuration) * available);

  // second pass: enforce minimums and redistribute
  let deficit = 0;
  const clamped = raw.map((w) => {
    if (w < MIN_SEGMENT_PX) {
      deficit += MIN_SEGMENT_PX - w;
      return MIN_SEGMENT_PX;
    }
    return w;
  });

  if (deficit > 0) {
    const elastic = clamped.filter((w) => w > MIN_SEGMENT_PX);
    const elasticTotal = elastic.reduce((s, w) => s + w, 0);
    if (elasticTotal > 0) {
      const scale = (elasticTotal - deficit) / elasticTotal;
      return clamped.map((w) => (w > MIN_SEGMENT_PX ? w * scale : w));
    }
  }

  return clamped;
}

export function ExecutionDetails({ node }: Props) {
  const { selectedOperationId, clearSelectedOperation } = useSidebar();
  const exec = node.execution;
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);
  const timelineViewportRef = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState(0);
  const operations = exec?.operations ?? [];

  useEffect(() => {
    if (!timelineViewportRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setBarWidth(entry.contentRect.width);
    });
    ro.observe(timelineViewportRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!exec?.invoked || operations.length === 0) {
      if (selectedSegment !== null) setSelectedSegment(null);
      return;
    }
    if (!selectedSegment || operations.some((op) => op.id === selectedSegment)) return;
    setSelectedSegment(operations[0].id);
  }, [exec?.invoked, operations, selectedSegment]);

  useEffect(() => {
    if (!selectedOperationId) return;
    if (!operations.some((op) => op.id === selectedOperationId)) return;
    setSelectedSegment(selectedOperationId);
    clearSelectedOperation();
  }, [selectedOperationId, operations, clearSelectedOperation]);

  if (!exec || !exec.invoked) {
    const emptyCopy = node.playback?.frameState === "upcoming"
      ? "this agent has not been invoked yet at the selected frame."
      : "this agent was not invoked during the selected trace.";
    return (
      <section className="side-panel__section">
        <h3 className="side-panel__section-title">execution</h3>
        <div className="side-panel__card">
          <p className="side-panel__empty">{emptyCopy}</p>
        </div>
      </section>
    );
  }

  const activeSegmentId = selectedSegment || (operations.length > 0 ? operations[0].id : null);
  const activeItem = operations.find((item) => item.id === activeSegmentId);
  const widths = computeSegmentWidths(operations, barWidth);
  const timelineWidth = widths.length > 0
    ? widths.reduce((sum, width) => sum + width, 0)
    : Math.max(barWidth, operations.length * MIN_SEGMENT_PX);
  // the hovered segment for the floating icon tooltip, if it's too thin
  const tooltipItem = hoveredSegment ? operations.find((op) => op.id === hoveredSegment) : null;
  const tooltipIndex = tooltipItem ? operations.indexOf(tooltipItem) : -1;
  const tooltipWidth = tooltipIndex >= 0 ? widths[tooltipIndex] : 0;

  // cumulative durations for the time ruler
  const cumulativeMs: number[] = [];
  let runningTotal = 0;
  for (const op of operations) {
    runningTotal += op.latencyMs ?? 0;
    cumulativeMs.push(runningTotal);
  }

  return (
    <>
      <section className="side-panel__section">
        <h3 className="side-panel__section-title">execution summary</h3>
        <div className="side-panel__meta-grid">
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">status</span>
            <span
              className="side-panel__meta-value"
              style={{ color: exec.status === "error" ? "#ef4444" : "#10b981" }}
            >
              {exec.status ?? "success"}
            </span>
          </div>
          {exec.latencyMs != null && (
            <div className="side-panel__meta-card">
              <span className="side-panel__meta-key">latency</span>
              <span className="side-panel__meta-value">{Math.round(exec.latencyMs)}ms</span>
            </div>
          )}
          {exec.promptTokens != null && (
            <div className="side-panel__meta-card">
              <span className="side-panel__meta-key">prompt tokens</span>
              <span className="side-panel__meta-value">{exec.promptTokens}</span>
            </div>
          )}
          {exec.completionTokens != null && (
            <div className="side-panel__meta-card">
              <span className="side-panel__meta-key">completion tokens</span>
              <span className="side-panel__meta-value">{exec.completionTokens}</span>
            </div>
          )}
          {exec.retryCount != null && exec.retryCount > 1 && (
            <div className="side-panel__meta-card">
              <span className="side-panel__meta-key">retries</span>
              <span className="side-panel__meta-value">{exec.retryCount - 1}</span>
            </div>
          )}
        </div>
      </section>

      {operations.length > 0 && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">operations</h3>
          <div className="side-panel__timeline-container">
            <div className="side-panel__timeline-scroll" ref={timelineViewportRef}>
              <div className="side-panel__timeline-track" style={{ width: `${timelineWidth}px` }}>
                {/* segmented timeline bar */}
                <div className="side-panel__progbar" role="list" aria-label="operations progress">
                  {operations.map((item, index) => {
                    const w = widths[index] ?? MIN_SEGMENT_PX;
                    const isActive = item.id === activeSegmentId;
                    const isErr = item.status === "error";
                    const colorClass = operationColorMap[item.type];
                    const labelFits = w >= LABEL_FIT_THRESHOLD;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        role="listitem"
                        className={`side-panel__progbar-seg seg--${colorClass}${isActive ? " is-active" : ""}${isErr ? " is-error" : ""}`}
                        style={{ width: `${w}px` }}
                        onClick={() => setSelectedSegment(item.id)}
                        onMouseEnter={() => setHoveredSegment(item.id)}
                        onMouseLeave={() => setHoveredSegment(null)}
                        title={`${item.label}${item.latencyMs != null ? ` · ${Math.round(item.latencyMs)}ms` : ""}`}
                        aria-label={item.label}
                      >
                        <span className="side-panel__progbar-step">{index + 1}</span>
                        <div className="side-panel__progbar-content">
                          <img src={operationIconMap[item.type]} alt="" className="side-panel__progbar-icon" />
                          {labelFits && <span className="side-panel__progbar-label">{item.label}</span>}
                        </div>
                      </button>
                    );
                  })}

                  {/* floating label tooltip for thin segments */}
                  {tooltipItem && tooltipWidth < LABEL_FIT_THRESHOLD && (
                    <div
                      className="side-panel__progbar-tip"
                      style={{
                        left: `${widths.slice(0, tooltipIndex).reduce((s, w) => s + w, 0) + tooltipWidth / 2}px`,
                      }}
                    >
                      <span className="side-panel__progbar-tip-label">{tooltipItem.label}</span>
                    </div>
                  )}
                </div>

                {/* cumulative time ruler */}
                {cumulativeMs.length > 0 && barWidth > 0 && (
                  <div className="side-panel__time-ruler" aria-hidden="true">
                    <span className="side-panel__time-ruler-label" style={{ left: 0 }}>0ms</span>
                    {cumulativeMs.map((ms, i) => {
                      const left = widths.slice(0, i + 1).reduce((s, w) => s + w, 0);
                      return (
                        <span
                          key={operations[i].id}
                          className="side-panel__time-ruler-label"
                          style={{ left: `${left}px` }}
                        >
                          {formatMs(ms)}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* operation meta pills — below the bar */}
            {activeItem && (
              <div className="side-panel__op-meta">
                <span className="side-panel__op-pill">{activeItem.label}</span>
                <span className="side-panel__op-pill">{activeItem.type.replace("_", " ")}</span>
                {activeItem.latencyMs != null && (
                  <span className="side-panel__op-pill">{Math.round(activeItem.latencyMs)}ms</span>
                )}
                {activeItem.tokenCount != null && activeItem.tokenCount > 0 && (
                  <span className="side-panel__op-pill">{activeItem.tokenCount} tokens</span>
                )}
              </div>
            )}

            {/* operation detail — separate cards per section */}
            {activeItem && (() => {
              const isStateUpdate = activeItem.type === "state_update";
              const changedKeys = (activeItem.metadata?.changedKeys ?? []) as string[];

              if (isStateUpdate && changedKeys.length > 0) {
                return (
                  <div className="side-panel__timeline-detail">
                    {activeItem.errorMessage && (
                      <div className="side-panel__card">
                        <div className="side-panel__card-label side-panel__card-label--error">error</div>
                        <pre className="side-panel__pre">{activeItem.errorMessage}</pre>
                      </div>
                    )}
                    <div className="side-panel__card">
                      <div className="side-panel__card-label">state changes</div>
                      <StateDiffView
                        input={activeItem.input}
                        output={activeItem.output}
                        changedKeys={changedKeys}
                      />
                    </div>
                  </div>
                );
              }

              const inputLog = formatForLog(activeItem.input);
              const outputLog = formatForLog(activeItem.output, { full: true });
              const metadataLog = formatForLog(activeItem.metadata);
              return (
                <div className="side-panel__timeline-detail">
                  {activeItem.errorMessage && (
                    <div className="side-panel__card">
                      <div className="side-panel__card-label side-panel__card-label--error">error</div>
                      <pre className="side-panel__pre">{activeItem.errorMessage}</pre>
                    </div>
                  )}
                  <div className="side-panel__card">
                    <div className="side-panel__card-label">input</div>
                    {inputLog.empty ? (
                      <span className="side-panel__empty">empty</span>
                    ) : (
                      <pre className="side-panel__pre">{inputLog.text}</pre>
                    )}
                  </div>
                  <div className="side-panel__card">
                    <div className="side-panel__card-label">output</div>
                    {outputLog.empty ? (
                      <span className="side-panel__empty">empty</span>
                    ) : (
                      <pre className="side-panel__pre">{outputLog.text}</pre>
                    )}
                  </div>
                  <div className="side-panel__card">
                    <div className="side-panel__card-label">metadata</div>
                    {metadataLog.empty ? (
                      <span className="side-panel__empty">empty</span>
                    ) : (
                      <pre className="side-panel__pre">{metadataLog.text}</pre>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </section>
      )}
    </>
  );
}
