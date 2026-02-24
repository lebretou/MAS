import type { GraphTopology } from "../../../types/graph";

interface Props {
  graph: GraphTopology | null;
}

function formatDateTime(value?: string | null): string {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function GraphInfoPanel({ graph }: Props) {
  if (!graph) return null;

  return (
    <section className="graph-info-card" aria-label="Graph information">
      <header className="graph-info-card__header">
        <h3 className="graph-info-card__title">Graph Details</h3>
      </header>
      <div className="graph-info-card__content">
        <div className="graph-info-card__row">
          <span className="graph-info-card__key">graph id</span>
          <span className="graph-info-card__value" title={graph.graph_id}>
            {graph.graph_id}
          </span>
        </div>
        <div className="graph-info-card__row">
          <span className="graph-info-card__key">name</span>
          <span className="graph-info-card__value" title={graph.name}>
            {graph.name}
          </span>
        </div>
        <div className="graph-info-card__row">
          <span className="graph-info-card__key">created at</span>
          <span className="graph-info-card__value">{formatDateTime(graph.created_at)}</span>
        </div>
        <div className="graph-info-card__row">
          <span className="graph-info-card__key">updated at</span>
          <span className="graph-info-card__value">{formatDateTime(graph.updated_at)}</span>
        </div>
      </div>
    </section>
  );
}

