import { useEffect, useState } from "react";
import { Panel } from "@xyflow/react";
import { fetchTraces } from "../../../api/traces";
import { useLayer } from "../../../context/LayerContext";
import type { TraceMetadata } from "../../../types/trace";

export function TraceSelector() {
  const { layer, selectedTraceId, setSelectedTraceId } = useLayer();
  const [traces, setTraces] = useState<TraceMetadata[]>([]);

  useEffect(() => {
    if (layer === "execution") {
      fetchTraces().then(setTraces).catch(() => {});
    }
  }, [layer]);

  if (layer !== "execution") return null;

  return (
    <Panel position="top-center" className="trace-selector-panel">
      <select
        className="trace-selector"
        value={selectedTraceId ?? ""}
        onChange={(e) => setSelectedTraceId(e.target.value || null)}
      >
        <option value="">select a trace...</option>
        {traces.map((t) => (
          <option key={t.trace_id} value={t.trace_id}>
            {t.trace_id.slice(0, 8)}... ({t.event_count} events)
          </option>
        ))}
      </select>
    </Panel>
  );
}
