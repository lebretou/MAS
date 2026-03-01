import type { GraphNodeData } from "../../../types/node-data";
import iconCode from "../../../assets/icon-code.svg";
import iconError from "../../../assets/icon-error.svg";
import iconLlm from "../../../assets/icon-llm.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconRetry from "../../../assets/icon-retry.svg";
import iconTool from "../../../assets/icon-tool.svg";
import iconChain from "../../../assets/icon-chain.svg";

interface Props {
  data: GraphNodeData;
}

const operationIconMap = {
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  error: iconError,
} as const;

function truncateLabel(label: string): string {
  if (label.length <= 16) return label;
  return `${label.slice(0, 13)}...`;
}

export function ExecutionContent({ data }: Props) {
  const exec = data.execution;
  const hasOperations = Boolean(exec?.operations && exec.operations.length > 0);

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
    <>
      <div className={`agent-node__body${hasOperations ? " agent-node__body--with-ops" : ""}`}>
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
        {exec.retryCount != null && exec.retryCount > 1 && (
          <div className="agent-node__row">
            <span className="agent-node__key">retries</span>
            <span className="agent-node__value agent-node__retry-value">
              <img src={iconRetry} alt="" className="agent-node__retry-icon" />
              {exec.retryCount - 1}
            </span>
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

      {hasOperations && (
        <div className="agent-node__components agent-node__components--ops">
          <div className="agent-node__components-header">
            <span className="agent-node__components-label">OPERATIONS</span>
          </div>
          <div className="agent-node__ops">
            {exec.operations?.slice(0, 5).map((operation) => (
              <span
                key={operation.id}
                className={`agent-node__op-chip${operation.status === "error" ? " agent-node__op-chip--error" : ""}`}
                title={operation.label}
              >
                <img src={operationIconMap[operation.type]} alt="" className="agent-node__op-icon" />
                <span>{truncateLabel(operation.label)}</span>
              </span>
            ))}
            {(exec.operations?.length ?? 0) > 5 && (
              <span className="agent-node__op-more">+{(exec.operations?.length ?? 0) - 5}</span>
            )}
          </div>
        </div>
      )}
    </>
  );
}
