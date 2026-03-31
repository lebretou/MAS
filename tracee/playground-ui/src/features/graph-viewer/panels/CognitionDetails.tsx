import { useState } from "react";
import type { AgentOperation, GraphNodeData } from "../../../types/node-data";
import { CognitionText } from "../../../components/CognitionText";
import iconLlm from "../../../assets/icon-llm.svg";
import iconTool from "../../../assets/icon-tool.svg";
import iconRag from "../../../assets/icon-rag.svg";
import iconCode from "../../../assets/icon-code.svg";
import iconChain from "../../../assets/icon-chain.svg";
import iconState from "../../../assets/icon-state.svg";
import iconError from "../../../assets/icon-error.svg";

interface Props {
  node: GraphNodeData;
}

const operationIconMap: Record<AgentOperation["type"], string> = {
  llm_call: iconLlm,
  tool_call: iconTool,
  rag_retrieve: iconRag,
  code_exec: iconCode,
  subgraph_call: iconChain,
  state_update: iconState,
  error: iconError,
};

function formatValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try { return JSON.stringify(JSON.parse(trimmed), null, 2); } catch { /* fall through */ }
    }
    return value.replace(/\\n/g, "\n").replace(/\\t/g, "\t");
  }
  return JSON.stringify(value, null, 2);
}

function findOperationByChip(
  chipType: string,
  chipValue: string,
  operations: AgentOperation[],
): AgentOperation | undefined {
  if (chipType === "tool") {
    return operations.find(
      (op) => (op.type === "tool_call" || op.type === "rag_retrieve" || op.type === "code_exec")
        && op.label.toLowerCase().includes(chipValue.toLowerCase()),
    );
  }
  if (chipType === "state") {
    return operations.find(
      (op) => op.type === "state_update"
        && op.metadata?.changedKeys
        && (op.metadata.changedKeys as string[]).some((k: string) => k.toLowerCase().includes(chipValue.toLowerCase())),
    );
  }
  return undefined;
}

export function CognitionDetails({ node }: Props) {
  const cog = node.cognition;
  const exec = node.execution;
  const operations = exec?.operations ?? [];
  const [expandedOp, setExpandedOp] = useState<AgentOperation | null>(null);

  if (!cog) {
    return (
      <section className="side-panel__section">
        <h3 className="side-panel__section-title">cognition</h3>
        <div className="side-panel__card">
          <p className="side-panel__empty">
            no cognition analysis available. switch to the cognition layer and run analysis.
          </p>
        </div>
      </section>
    );
  }

  const handleChipClick = (chipType: string, chipValue: string) => {
    const op = findOperationByChip(chipType, chipValue, operations);
    if (op) {
      setExpandedOp((prev) => prev?.id === op.id ? null : op);
    }
  };

  return (
    <>
      {cog.handoff_description && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">incoming handoff</h3>
          <div className="cognition-detail__handoff">
            <CognitionText text={cog.handoff_description} />
          </div>
        </section>
      )}

      <section className="side-panel__section">
        <h3 className="side-panel__section-title">description</h3>
        <div className="side-panel__card">
          <div className="cognition-detail__description">
            <CognitionText
              text={cog.description}
              onToolClick={(value) => handleChipClick("tool", value)}
              onStateClick={(value) => handleChipClick("state", value)}
            />
          </div>
        </div>
      </section>

      {expandedOp && (
        <section className="side-panel__section">
          <h3 className="side-panel__section-title">
            <img src={operationIconMap[expandedOp.type]} alt="" style={{ width: 14, height: 14, marginRight: 4, verticalAlign: "middle" }} />
            {expandedOp.label}
            <button
              className="cognition-detail__close-op"
              onClick={() => setExpandedOp(null)}
              aria-label="close"
            >
              &times;
            </button>
          </h3>
          <div className="side-panel__card">
            {expandedOp.input != null && (
              <>
                <div className="side-panel__card-label">input</div>
                <pre className="side-panel__pre">{formatValue(expandedOp.input)}</pre>
              </>
            )}
            {expandedOp.output != null && (
              <>
                <div className="side-panel__card-label" style={{ marginTop: 8 }}>output</div>
                <pre className="side-panel__pre">{formatValue(expandedOp.output)}</pre>
              </>
            )}
            {expandedOp.errorMessage && (
              <>
                <div className="side-panel__card-label side-panel__card-label--error" style={{ marginTop: 8 }}>error</div>
                <pre className="side-panel__pre">{expandedOp.errorMessage}</pre>
              </>
            )}
          </div>
        </section>
      )}
    </>
  );
}
