import { useEffect, useMemo, useRef, useState } from "react";
import { Panel } from "@xyflow/react";
import type { ExecutionFrame } from "../../../types/node-data";
import type { JsonSchema } from "../../../types/schema";
import iconCognition from "../../../assets/cognition.svg";
import iconTimeframe from "../../../assets/icon-timeframe.svg";
import iconStateVariable from "../../../assets/icon-statevariable.svg";
import dictIcon from "../../../assets/dict.svg";
import listIcon from "../../../assets/list.svg";
import numberIcon from "../../../assets/number.svg";
import stringIcon from "../../../assets/string.svg";

interface Props {
  frames: ExecutionFrame[];
  activeFrameIndex: number | null;
  onFrameChange: (index: number | null) => void;
  schema: JsonSchema;
  activeFrame?: ExecutionFrame | null;
}

const MIN_SEGMENT_PX = 48;
const LABEL_FIT_THRESHOLD = 100;

const TYPE_ICONS: Record<string, string> = {
  object: dictIcon,
  array: listIcon,
  number: numberIcon,
  string: stringIcon,
  boolean: stringIcon,
};

function computeSegmentWidths(frames: ExecutionFrame[], totalPx: number): number[] {
  if (frames.length === 0) return [];
  const durations = frames.map((f) => f.latencyMs || 1);
  const totalDuration = durations.reduce((sum, d) => sum + d, 0);
  if (totalDuration === 0) return durations.map(() => totalPx / frames.length);

  const minTotal = MIN_SEGMENT_PX * frames.length;
  const available = Math.max(totalPx, minTotal);
  const raw = durations.map((d) => (d / totalDuration) * available);

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

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function inferTypeFromValue(value: unknown): string {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
}

// ── timeline (left zone) ────────────────────────────────

function TimelineZone({
  frames,
  activeFrameIndex,
  onFrameChange,
}: {
  frames: ExecutionFrame[];
  activeFrameIndex: number | null;
  onFrameChange: (index: number | null) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState(0);

  useEffect(() => {
    if (!barRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setBarWidth(entry.contentRect.width);
    });
    ro.observe(barRef.current);
    return () => ro.disconnect();
  }, []);

  // auto-scroll active frame into view
  useEffect(() => {
    if (activeFrameIndex == null || !scrollRef.current) return;
    const container = scrollRef.current;
    const segments = container.querySelectorAll<HTMLElement>(".ei-timeline__seg");
    const seg = segments[activeFrameIndex];
    if (!seg) return;
    const segLeft = seg.offsetLeft;
    const segRight = segLeft + seg.offsetWidth;
    const scrollLeft = container.scrollLeft;
    const visible = container.clientWidth;
    if (segLeft < scrollLeft) {
      container.scrollTo({ left: Math.max(0, segLeft - 16), behavior: "smooth" });
    } else if (segRight > scrollLeft + visible) {
      container.scrollTo({ left: segRight - visible + 16, behavior: "smooth" });
    }
  }, [activeFrameIndex]);

  const totalNeeded = frames.length * MIN_SEGMENT_PX;
  const needsScroll = totalNeeded > barWidth && barWidth > 0;
  const effectiveWidth = needsScroll ? totalNeeded : barWidth;
  const widths = computeSegmentWidths(frames, effectiveWidth);

  const cumulativeMs: number[] = [];
  let running = 0;
  for (const f of frames) {
    running += f.latencyMs || 0;
    cumulativeMs.push(running);
  }

  const activeFrame = activeFrameIndex != null ? frames[activeFrameIndex] : null;

  return (
    <div className="ei-timeline">
      <div className="ei-timeline__header">
        <div className="ei-timeline__title-group">
          <div className="ei-timeline__title-row">
            <img src={iconTimeframe} alt="" className="ei-timeline__title-icon" aria-hidden />
            <span className="ei-timeline__title">State Timeline</span>
          </div>
          <span className="ei-timeline__subtitle">
            {activeFrame && activeFrameIndex != null
              ? `Step ${activeFrameIndex + 1} / ${frames.length} · ${activeFrame.label} · ${formatMs(activeFrame.latencyMs)}`
              : `${frames.length} steps · ${formatMs(running)}`}
          </span>
        </div>
        <div className="ei-timeline__controls">
          <button
            type="button"
            className={`ei-timeline__btn${activeFrameIndex == null ? " is-active" : ""}`}
            onClick={() => onFrameChange(null)}
            aria-label="show all frames"
          >
            All
          </button>
          <button
            type="button"
            className="ei-timeline__btn"
            onClick={() =>
              onFrameChange(activeFrameIndex == null ? frames.length - 1 : Math.max(0, activeFrameIndex - 1))
            }
            disabled={frames.length === 0}
            aria-label="previous frame"
          >
            ◀
          </button>
          <button
            type="button"
            className="ei-timeline__btn"
            onClick={() =>
              onFrameChange(
                activeFrameIndex == null ? null : Math.min(frames.length - 1, activeFrameIndex + 1),
              )
            }
            disabled={activeFrameIndex == null || activeFrameIndex >= frames.length - 1}
            aria-label="next frame"
          >
            ▶
          </button>
        </div>
      </div>

      <div
        className={`ei-timeline__scroll-area${needsScroll ? " is-scrollable" : ""}`}
        ref={scrollRef}
      >
        <div
          className="ei-timeline__bar"
          ref={barRef}
          role="list"
          style={needsScroll ? { width: `${effectiveWidth}px` } : undefined}
        >
          {frames.map((frame, index) => {
            const w = widths[index] ?? MIN_SEGMENT_PX;
            const isReached = activeFrameIndex == null || index <= activeFrameIndex;
            const isCurrent = activeFrameIndex != null && index === activeFrameIndex;
            const labelFits = w >= LABEL_FIT_THRESHOLD;
            return (
              <button
                key={frame.eventId}
                type="button"
                role="listitem"
                className={`ei-timeline__seg seg--chain${isReached ? " is-active" : ""}${isCurrent ? " is-current" : ""}`}
                style={{ width: `${w}px` }}
                onClick={() => onFrameChange(index)}
                title={`${frame.label}${frame.latencyMs ? ` · ${formatMs(frame.latencyMs)}` : ""}`}
                aria-label={frame.label}
              >
                <span className="ei-timeline__step-badge">{index + 1}</span>
                <div className="ei-timeline__seg-content">
                  <img src={iconCognition} alt="" className="ei-timeline__seg-icon" />
                  {labelFits && <span className="ei-timeline__seg-label">{frame.label}</span>}
                </div>
              </button>
            );
          })}
        </div>

        {cumulativeMs.length > 0 && effectiveWidth > 0 && (
          <div className="ei-timeline__ruler" aria-hidden="true">
            <span className="ei-timeline__ruler-label" style={{ left: 0 }}>
              0ms
            </span>
            {cumulativeMs.map((ms, i) => {
              const left = widths.slice(0, i + 1).reduce((s, w) => s + w, 0);
              return (
                <span
                  key={frames[i].eventId}
                  className="ei-timeline__ruler-label"
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
  );
}

// ── compact state changes (right zone) ──────────────────
// only shows keys that changed in the active frame

function StateChangesZone({
  schema,
  activeFrame,
}: {
  schema: JsonSchema;
  activeFrame?: ExecutionFrame | null;
}) {
  const properties = schema.properties || {};
  const displayFrame = activeFrame ?? null;
  const changedKeys = displayFrame?.changedKeys ?? [];
  const canShowChanges = Boolean(displayFrame && displayFrame.index > 0);
  const keysToShow = canShowChanges ? changedKeys : [];

  // for frame 0 (initial state), show all keys that got populated
  const initialKeys = useMemo(() => {
    if (!displayFrame || displayFrame.index !== 0) return [];
    const snap = displayFrame.stateSnapshot;
    return Object.keys(snap).filter((k) => {
      const v = snap[k];
      return v !== undefined && v !== "" && v != null &&
        !(Array.isArray(v) && v.length === 0) &&
        !(typeof v === "object" && v !== null && !Array.isArray(v) && Object.keys(v).length === 0);
    }).sort();
  }, [displayFrame]);

  const displayKeys = canShowChanges ? keysToShow : initialKeys;

  return (
    <div className="ei-changes">
      <div className="ei-changes__header">
        <img src={iconStateVariable} alt="" className="ei-changes__icon" aria-hidden />
        <span className="ei-changes__title">
          {displayFrame ? `Step ${displayFrame.index + 1}` : "State"}
        </span>
        {displayKeys.length > 0 && (
          <span className="ei-changes__count">{displayKeys.length} updated</span>
        )}
      </div>

      {displayFrame == null ? (
        <div className="ei-changes__empty">
          select a step to see state changes
        </div>
      ) : displayKeys.length === 0 ? (
        <div className="ei-changes__empty">
          no state changes
        </div>
      ) : (
        <div className="ei-changes__list">
          {displayKeys.map((key) => {
            const propSchema = properties[key];
            const typeStr = Array.isArray(propSchema?.type)
              ? propSchema.type[0]
              : propSchema?.type || inferTypeFromValue(displayFrame.stateSnapshot[key]);
            const iconSrc = TYPE_ICONS[typeStr] || stringIcon;

            return (
              <div key={key} className="ei-changes__row">
                <img src={iconSrc} alt="" className="ei-changes__type-icon" />
                <span className="ei-changes__key">{key}</span>
                <span className="ei-changes__type">{typeStr}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── main composite ──────────────────────────────────────

export function ExecutionInspector({
  frames,
  activeFrameIndex,
  onFrameChange,
  schema,
  activeFrame,
}: Props) {
  return (
    <Panel position="bottom-center" className="execution-inspector">
      <TimelineZone
        frames={frames}
        activeFrameIndex={activeFrameIndex}
        onFrameChange={onFrameChange}
      />
      <div className="ei-divider" />
      <StateChangesZone
        schema={schema}
        activeFrame={activeFrame}
      />
    </Panel>
  );
}
