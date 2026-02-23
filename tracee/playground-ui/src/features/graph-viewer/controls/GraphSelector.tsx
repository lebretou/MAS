import { useEffect, useState } from "react";
import { Panel } from "@xyflow/react";
import { fetchGraphs } from "../../../api/graphs";
import type { GraphTopology } from "../../../types/graph";

interface Props {
  selectedGraphId: string | null;
  onSelect: (graphId: string) => void;
}

export function GraphSelector({ selectedGraphId, onSelect }: Props) {
  const [graphs, setGraphs] = useState<GraphTopology[]>([]);

  useEffect(() => {
    fetchGraphs().then(setGraphs).catch(() => {});
  }, []);

  if (graphs.length <= 1) return null;

  return (
    <Panel position="top-center" className="graph-selector-panel">
      <select
        className="graph-selector"
        value={selectedGraphId ?? ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        {graphs.map((g) => (
          <option key={g.graph_id} value={g.graph_id}>
            {g.name}
          </option>
        ))}
      </select>
    </Panel>
  );
}
