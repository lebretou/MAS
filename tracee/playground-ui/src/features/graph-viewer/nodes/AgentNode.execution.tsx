import { Fragment } from "react";
import type { AgentOperationType, GraphNodeData } from "../../../types/node-data";
import iconCode from "../../../assets/icon-code.svg";
import iconError from "../../../assets/icon-error.svg";
import iconLlm from "../../../assets/icon-llm.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconRetry from "../../../assets/icon-retry.svg";
import iconState from "../../../assets/icon-state.svg";
import iconTool from "../../../assets/icon-tool.svg";
import iconChain from "../../../assets/icon-chain.svg";
import { SchemaValidationIndicator } from "./SchemaValidationIndicator";
import { hasOutputSchemaProperties } from "../../../utils/schema-validation";

interface Props {
  data: GraphNodeData;
}

const operationIconMap = {
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  state_update: iconState,
  error: iconError,
} as const;

const operationLabelMap: Record<AgentOperationType, string> = {
  llm_call: "LLM",
  tool_call: "Tool",
  rag_retrieve: "RAG",
  code_exec: "Code",
  subgraph_call: "Sub",
  state_update: "State",
  error: "Error",
};

// same type → color class as ExecutionDetails progress bar for consistent styling
const operationColorMap: Record<AgentOperationType, string> = {
  llm_call: "llm",
  tool_call: "tool",
  rag_retrieve: "rag",
  code_exec: "code",
  subgraph_call: "chain",
  state_update: "state",
  error: "error",
};

export function ExecutionContent({ data }: Props) {
  const exec = data.execution;
  const hasOperations = Boolean(exec?.operations && exec.operations.length > 0);
  const hasSchemaValidation = Boolean(
    data.outputSchema
      && hasOutputSchemaProperties(data.outputSchema)
      && (exec?.llmOutputValue != null || (exec?.events?.length ?? 0) > 0),
  );
  const totalOperations = exec?.operations?.length ?? 0;
  const hasHiddenOperations = totalOperations > 4;
  const visibleOperations = exec?.operations?.slice(0, hasHiddenOperations ? 3 : 4) ?? [];
  const hiddenOperations = totalOperations - visibleOperations.length;

  if (!exec || !exec.invoked) {
    const frameState = data.playback?.frameState;
    const statusLabel = frameState === "upcoming" ? "not yet invoked" : "not invoked";
    return (
      <div className="agent-node__body">
        <div className="agent-node__row">
          <span className="agent-node__key">status</span>
          <span className="agent-node__value" style={{ color: "#9ca3af" }}>{statusLabel}</span>
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

      {(hasOperations || hasSchemaValidation) && (
        <div className="agent-node__components agent-node__components--ops">
          {hasOperations && (
            <>
              <div className="agent-node__components-header">
                <span className="agent-node__components-label">OPERATIONS</span>
              </div>
              <div className="agent-node__timeline">
                {visibleOperations.map((operation, index, arr) => (
                  <Fragment key={operation.id}>
                    <span
                      className={`agent-node__timeline-node agent-node__timeline-node--${operationColorMap[operation.type]}${operation.status === "error" ? " agent-node__timeline-node--error" : ""}`}
                      title={operation.label}
                    >
                      <img src={operationIconMap[operation.type]} alt="" className="agent-node__timeline-icon" />
                      <span className="agent-node__timeline-label">{operationLabelMap[operation.type]}</span>
                    </span>
                    {index < arr.length - 1 && <div className="agent-node__timeline-line" />}
                  </Fragment>
                ))}
                {hiddenOperations > 0 && (
                  <>
                    <div className="agent-node__timeline-line" />
                    <span className="agent-node__op-more">+{hiddenOperations}</span>
                  </>
                )}
              </div>
            </>
          )}
          <SchemaValidationIndicator
            outputSchema={data.outputSchema}
            outputValue={exec?.llmOutputValue}
            events={exec?.events}
          />
        </div>
      )}
    </>
  );
}
