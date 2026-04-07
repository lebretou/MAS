import { useEffect, useRef, useState } from "react";
import type { AgentOperation, GraphNodeData } from "../../../types/node-data";
import { CognitionText } from "../../../components/CognitionText";
import { StateDiffView } from "../../../components/StateDiffView";
import { useSidebar } from "../../../context/SidebarContext";
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

function unescapeForDisplay(s: string): string {
  return s
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\r/g, "\r")
    .replace(/\\"/g, '"');
}

function formatValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try { return unescapeForDisplay(JSON.stringify(JSON.parse(trimmed), null, 2)); } catch { /* fall through */ }
    }
    return unescapeForDisplay(value);
  }
  return unescapeForDisplay(JSON.stringify(value, null, 2));
}

function isEmptyLog(value: unknown): boolean {
  if (value == null || value === "") return true;
  if (Array.isArray(value) && value.length === 0) return true;
  if (
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value as object).length === 0
  )
    return true;
  return false;
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
  const [unmatchedChip, setUnmatchedChip] = useState<string | null>(null);
  const unmatchedTimer = useRef<ReturnType<typeof setTimeout>>();
  const { chipExpansion, clearChipExpansion } = useSidebar();

  // auto-expand when navigating from summary panel chip click
  useEffect(() => {
    if (!chipExpansion || operations.length === 0) return;
    const op = findOperationByChip(chipExpansion.type, chipExpansion.value, operations);
    if (op) {
      setExpandedOp(op);
      clearChipExpansion();
    }
  }, [chipExpansion, operations, clearChipExpansion]);

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
      setUnmatchedChip(null);
      setExpandedOp((prev) => prev?.id === op.id ? null : op);
    } else {
      clearTimeout(unmatchedTimer.current);
      setUnmatchedChip(chipValue);
      unmatchedTimer.current = setTimeout(() => setUnmatchedChip(null), 2500);
    }
  };

  const isStateUpdate = expandedOp?.type === "state_update";
  const changedKeys = (expandedOp?.metadata?.changedKeys ?? []) as string[];

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
          {unmatchedChip && (
            <div className="cognition-detail__unmatched">
              no matching operation found for "{unmatchedChip}"
            </div>
          )}
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
          <div className="side-panel__timeline-detail">
            {isStateUpdate && changedKeys.length > 0 ? (
              <>
                {expandedOp.errorMessage && (
                  <div className="side-panel__card">
                    <div className="side-panel__card-label side-panel__card-label--error">error</div>
                    <pre className="side-panel__pre">{expandedOp.errorMessage}</pre>
                  </div>
                )}
                <div className="side-panel__card">
                  <div className="side-panel__card-label">state changes</div>
                  <StateDiffView
                    input={expandedOp.input}
                    output={expandedOp.output}
                    changedKeys={changedKeys}
                  />
                </div>
              </>
            ) : (
              <>
                {expandedOp.errorMessage && (
                  <div className="side-panel__card">
                    <div className="side-panel__card-label side-panel__card-label--error">error</div>
                    <pre className="side-panel__pre">{expandedOp.errorMessage}</pre>
                  </div>
                )}
                <div className="side-panel__card">
                  <div className="side-panel__card-label">input</div>
                  {isEmptyLog(expandedOp.input) ? (
                    <span className="side-panel__empty">empty</span>
                  ) : (
                    <pre className="side-panel__pre">{formatValue(expandedOp.input)}</pre>
                  )}
                </div>
                <div className="side-panel__card">
                  <div className="side-panel__card-label">output</div>
                  {isEmptyLog(expandedOp.output) ? (
                    <span className="side-panel__empty">empty</span>
                  ) : (
                    <pre className="side-panel__pre">{formatValue(expandedOp.output)}</pre>
                  )}
                </div>
              </>
            )}
          </div>
        </section>
      )}
    </>
  );
}
