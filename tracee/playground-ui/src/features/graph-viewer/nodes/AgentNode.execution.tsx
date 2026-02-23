import type { GraphNodeData } from "../../../types/node-data";

interface Props {
  data: GraphNodeData;
}

export function ExecutionContent({ data }: Props) {
  const exec = data.execution;

  if (!exec || !exec.invoked) {
    return (
      <div className="agent-node__body">
        <div className="agent-node__row">
          <span className="agent-node__key">status</span>
          <span className="agent-node__value" style={{ color: "#9ca3af" }}>not invoked</span>
        </div>
      </div>
    );
  }

  return (
    <div className="agent-node__body">
      <div className="agent-node__row">
        <span className="agent-node__key">status</span>
        <span
          className="agent-node__value"
          style={{ color: exec.status === "error" ? "#ef4444" : "#10b981" }}
        >
          {exec.status ?? "success"}
        </span>
      </div>
      {exec.latencyMs != null && (
        <div className="agent-node__row">
          <span className="agent-node__key">latency</span>
          <span className="agent-node__value">{Math.round(exec.latencyMs)}ms</span>
        </div>
      )}
      {(exec.promptTokens != null || exec.completionTokens != null) && (
        <div className="agent-node__row">
          <span className="agent-node__key">tokens</span>
          <span className="agent-node__value">
            {exec.promptTokens ?? 0} / {exec.completionTokens ?? 0}
          </span>
        </div>
      )}
    </div>
  );
}
