import type { GraphNodeData } from "../../../types/node-data";

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
        </div>
      </section>

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
