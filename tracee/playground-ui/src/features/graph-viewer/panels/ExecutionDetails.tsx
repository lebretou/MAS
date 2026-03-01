import type { GraphNodeData } from "../../../types/node-data";
import iconCode from "../../../assets/icon-code.svg";
import iconError from "../../../assets/icon-error.svg";
import iconLlm from "../../../assets/icon-llm.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconTool from "../../../assets/icon-tool.svg";
import iconChain from "../../../assets/icon-chain.svg";

const operationIconMap = {
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  error: iconError,
} as const;

interface Props {
  node: GraphNodeData;
}

export function ExecutionDetails({ node }: Props) {
  const exec = node.execution;

  if (!exec || !exec.invoked) {
    return (
      <section className="side-panel__section">
        <h3 className="side-panel__section-title">execution</h3>
        <div className="side-panel__card">
          <p className="side-panel__empty">this agent was not invoked during the selected trace.</p>
        </div>
      </section>
    );
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

      {exec.operations && exec.operations.length > 0 && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">operations</h3>
          <div className="side-panel__ops-list">
            {exec.operations.map((operation) => (
              <div
                key={operation.id}
                className={`side-panel__op-row${operation.status === "error" ? " is-error" : ""}`}
              >
                <img src={operationIconMap[operation.type]} alt="" className="side-panel__op-icon" />
                <div className="side-panel__op-main">
                  <div className="side-panel__op-label">{operation.label}</div>
                  <div className="side-panel__op-meta">
                    <span className="side-panel__op-pill">{operation.type.replace("_", " ")}</span>
                    {operation.latencyMs != null && (
                      <span className="side-panel__op-pill">{Math.round(operation.latencyMs)}ms</span>
                    )}
                    {operation.tokenCount != null && operation.tokenCount > 0 && (
                      <span className="side-panel__op-pill">{operation.tokenCount} tokens</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {exec.llmInput && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">LLM input</h3>
          <div className="side-panel__card">
            <pre className="side-panel__pre">{exec.llmInput}</pre>
          </div>
        </section>
      )}

      {exec.llmOutput && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">LLM output</h3>
          <div className="side-panel__card">
            <pre className="side-panel__pre">{exec.llmOutput}</pre>
          </div>
        </section>
      )}
    </>
  );
}
