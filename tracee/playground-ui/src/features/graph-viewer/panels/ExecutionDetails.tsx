import { useEffect, useState } from "react";
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

function formatUnknown(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(sanitizeForDisplay(value), null, 2);
}

export function ExecutionDetails({ node }: Props) {
  const exec = node.execution;
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);
  const operations = exec?.operations ?? [];

  useEffect(() => {
    if (!exec?.invoked || operations.length === 0) {
      if (selectedSegment !== null) setSelectedSegment(null);
      return;
    }
    if (!selectedSegment || operations.some((op) => op.id === selectedSegment)) return;
    setSelectedSegment(operations[0].id);
  }, [exec?.invoked, operations, selectedSegment]);

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

  const activeSegmentId = selectedSegment || (operations.length > 0 ? operations[0].id : null);
  const activeItem = operations.find((item) => item.id === activeSegmentId);

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

      {exec.llmInput && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">LLM input</h3>
          <div className="side-panel__card">
            <pre className="side-panel__pre">{exec.llmInput}</pre>
          </div>
        </section>
      )}

      {operations.length > 0 && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">operations</h3>
          <div className="side-panel__timeline-container">
            <div className="side-panel__ops-progress" role="list" aria-label="operations progress">
              {operations.map((item, index) => (
                <div key={item.id} className="side-panel__ops-segment-wrap" role="listitem">
                  <button
                    type="button"
                    className={`side-panel__ops-segment${item.id === activeSegmentId ? " is-active" : ""}${item.status === "error" ? " is-error" : ""}`}
                    onClick={() => setSelectedSegment(item.id)}
                    title={item.label}
                    aria-label={item.label}
                  >
                    <img src={operationIconMap[item.type]} alt="" className="side-panel__timeline-icon" />
                  </button>
                  {index < operations.length - 1 && <div className="side-panel__ops-connector" />}
                </div>
              ))}
            </div>

            <div className="side-panel__timeline-detail">
              {activeItem && (
                <div className="side-panel__card">
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
                  {activeItem.errorMessage && (
                    <pre className="side-panel__pre">{activeItem.errorMessage}</pre>
                  )}
                  {activeItem.input != null && (
                    <>
                      <div className="side-panel__text">input</div>
                      <pre className="side-panel__pre">{formatUnknown(activeItem.input)}</pre>
                    </>
                  )}
                  {activeItem.output != null && (
                    <>
                      <div className="side-panel__text">output</div>
                      <pre className="side-panel__pre">{formatUnknown(activeItem.output)}</pre>
                    </>
                  )}
                  {activeItem.metadata && (
                    <>
                      <div className="side-panel__text">metadata</div>
                      <pre className="side-panel__pre">{formatUnknown(activeItem.metadata)}</pre>
                    </>
                  )}
                </div>
              )}
            </div>
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
