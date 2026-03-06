import { useEffect, useRef, useState } from "react";
import type { ExecutionFrame } from "../../../types/node-data";
import iconCognition from "../../../assets/cognition.svg";
import iconTimeframe from "../../../assets/icon-timeframe.svg";

interface Props {
  frames: ExecutionFrame[];
  activeFrameIndex: number | null;
  onChange: (index: number | null) => void;
}

const MIN_SEGMENT_PX = 48;
const LABEL_FIT_THRESHOLD = 100;

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

export function FrameScrubber({ frames, activeFrameIndex, onChange }: Props) {
  const activeFrame = activeFrameIndex == null ? null : frames[activeFrameIndex];
  const barRef = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState(0);
  const [hoveredFrame, setHoveredFrame] = useState<number | null>(null);

  useEffect(() => {
    if (!barRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setBarWidth(entry.contentRect.width);
    });
    ro.observe(barRef.current);
    return () => ro.disconnect();
  }, []);

  const widths = computeSegmentWidths(frames, barWidth);
  const tooltipFrameIndex = hoveredFrame != null ? hoveredFrame : -1;
  const tooltipFrame = tooltipFrameIndex >= 0 ? frames[tooltipFrameIndex] : null;
  const tooltipWidth = tooltipFrameIndex >= 0 ? widths[tooltipFrameIndex] : 0;

  const cumulativeMs: number[] = [];
  let runningTotal = 0;
  for (const f of frames) {
    runningTotal += f.latencyMs || 0;
    cumulativeMs.push(runningTotal);
  }

  return (
    <div className="frame-scrubber" aria-label="execution frame scrubber">
      <div className="frame-scrubber__header">
        <div className="frame-scrubber__title-group">
          <div className="frame-scrubber__title-row">
            <img src={iconTimeframe} alt="" className="frame-scrubber__title-icon" aria-hidden />
            <span className="frame-scrubber__title">Time frame</span>
          </div>
          <span className="frame-scrubber__subtitle">
            {activeFrame ? `${activeFrameIndex + 1} / ${frames.length} · ${activeFrame.label}` : "Showing all"}
          </span>
        </div>
        <div className="frame-scrubber__controls">
          <button
            type="button"
            className={`frame-scrubber__button${activeFrameIndex == null ? " is-active" : ""}`}
            onClick={() => onChange(null)}
            aria-label="show all frames"
          >
            Show all
          </button>
          <button
            type="button"
            className="frame-scrubber__button"
            onClick={() => onChange(activeFrameIndex == null ? frames.length - 1 : Math.max(0, activeFrameIndex - 1))}
            disabled={frames.length === 0}
            aria-label="previous frame"
          >
            Prev
          </button>
          <button
            type="button"
            className="frame-scrubber__button"
            onClick={() => onChange(activeFrameIndex == null ? null : Math.min(frames.length - 1, activeFrameIndex + 1))}
            disabled={activeFrameIndex == null || activeFrameIndex >= frames.length - 1}
            aria-label="next frame"
          >
            Next
          </button>
        </div>
      </div>

      <div className="frame-scrubber__timeline-container" style={{ position: "relative", marginTop: 24, marginBottom: 24 }}>
        <div className="side-panel__progbar" ref={barRef} role="list" aria-label="frames progress">
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
                className={`side-panel__progbar-seg seg--chain${isReached ? " is-active" : ""}${isCurrent ? " is-current" : ""}`}
                style={{ width: `${w}px` }}
                onClick={() => onChange(index)}
                onMouseEnter={() => setHoveredFrame(index)}
                onMouseLeave={() => setHoveredFrame(null)}
                title={`${frame.label}${frame.latencyMs ? ` · ${Math.round(frame.latencyMs)}ms` : ""}`}
                aria-label={frame.label}
              >
                <span className="side-panel__progbar-step">{index + 1}</span>
                <div className="side-panel__progbar-content">
                  <img src={iconCognition} alt="" className="side-panel__progbar-icon" />
                  {labelFits && <span className="side-panel__progbar-label">{frame.label}</span>}
                </div>
              </button>
            );
          })}

          {tooltipFrame && tooltipWidth < LABEL_FIT_THRESHOLD && (
            <div
              className="side-panel__progbar-tip"
              style={{
                left: `${widths.slice(0, tooltipFrameIndex).reduce((s, w) => s + w, 0) + tooltipWidth / 2}px`,
              }}
            >
              <span className="side-panel__progbar-tip-label">{tooltipFrame.label}</span>
            </div>
          )}
        </div>

        {cumulativeMs.length > 0 && barWidth > 0 && (
          <div className="side-panel__time-ruler" aria-hidden="true" style={{ marginTop: 8 }}>
            <span className="side-panel__time-ruler-label" style={{ left: 0 }}>0ms</span>
            {cumulativeMs.map((ms, i) => {
              const left = widths.slice(0, i + 1).reduce((s, w) => s + w, 0);
              return (
                <span
                  key={frames[i].eventId}
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
  );
}
