import { useState, useMemo } from "react";
import type { GraphNodeData } from "../../../types/node-data";
import { componentColors } from "../constants";
import { SchemaTable } from "./SchemaTable";

interface Props {
  node: GraphNodeData;
}

export function IntentDetails({ node }: Props) {
  const [mode, setMode] = useState<"components" | "raw">("components");
  const [schemaMode, setSchemaMode] = useState<"rendered" | "json">("rendered");
  const [activeComponentIdx, setActiveComponentIdx] = useState(0);
  const components = node.components ?? [];
  const metadata = node.metadata;
  const safeActiveIdx = Math.min(activeComponentIdx, Math.max(0, components.length - 1));
  const activeComponent = components[safeActiveIdx];

  const rawPrompt = useMemo(() => {
    if (components.length === 0) return "";
    return components
      .map((c) => {
        const status = c.enabled ? "" : " [disabled]";
        return `## ${c.type}${status}\n${c.content}`;
      })
      .join("\n\n");
  }, [components]);

  const outputSchemaJson = useMemo(() => {
    if (!node.outputSchema) return "";
    return JSON.stringify(node.outputSchema, null, 2);
  }, [node.outputSchema]);

  return (
    <>
      <section className="side-panel__section">
        <h3 className="side-panel__section-title">configuration</h3>
        <div className="side-panel__meta-grid">
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">prompt id</span>
            <span className="side-panel__meta-value prompt-link" title={node.promptId}>
              {node.promptId ?? "n/a"}
            </span>
          </div>
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">version</span>
            <span className="side-panel__meta-value" title={node.promptVersionId}>
              {node.promptVersionId ?? "n/a"}
            </span>
          </div>
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">model</span>
            <span className="side-panel__meta-value" title={metadata?.model}>
              {metadata?.model ?? "n/a"}
            </span>
          </div>
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">temperature</span>
            <span className="side-panel__meta-value">
              {metadata?.temperature ?? "n/a"}
            </span>
          </div>
          <div className="side-panel__meta-card">
            <span className="side-panel__meta-key">tools</span>
            <span className="side-panel__meta-value">
              {metadata?.hasTools ? "enabled" : "none"}
            </span>
          </div>
        </div>
      </section>

      <section className="side-panel__section">
        <div className="side-panel__section-head">
          <h3 className="side-panel__section-title">prompt components</h3>
          <div className="side-panel__mode-toggle">
            <button
              type="button"
              className={`side-panel__mode-btn ${mode === "components" ? "is-active" : ""}`}
              onClick={() => setMode("components")}
            >
              components
            </button>
            <button
              type="button"
              className={`side-panel__mode-btn ${mode === "raw" ? "is-active" : ""}`}
              onClick={() => setMode("raw")}
            >
              raw prompt
            </button>
          </div>
        </div>

        {mode === "components" ? (
          <div className="side-panel__components-layout">
            <div className="side-panel__component-nav">
              {components.map((component, idx) => (
                <button
                  key={`${component.type}-${idx}`}
                  type="button"
                  className={`side-panel__component-tab ${idx === safeActiveIdx ? "is-active" : ""} ${component.enabled ? "" : "is-disabled"}`}
                  onClick={() => setActiveComponentIdx(idx)}
                  style={
                    idx === safeActiveIdx && component.enabled
                      ? {
                          backgroundColor: `${componentColors[component.type]}1A`,
                          color: "#0f172a",
                        }
                      : undefined
                  }
                >
                  <span
                    className="side-panel__component-tab-dot"
                    style={{
                      background: component.enabled
                        ? componentColors[component.type]
                        : "#d1d5db",
                    }}
                  />
                  {component.type}
                </button>
              ))}
            </div>
            <div className="side-panel__card side-panel__component-content">
              {activeComponent ? (
                <>
                  <div className="side-panel__component-head">
                    <span className="side-panel__component-type">{activeComponent.type}</span>
                    {!activeComponent.enabled && (
                      <span className="side-panel__component-off">off</span>
                    )}
                  </div>
                  <p className="side-panel__text">{activeComponent.content}</p>
                </>
              ) : (
                <p className="side-panel__empty">no prompt components.</p>
              )}
            </div>
          </div>
        ) : (
          <div className="side-panel__card">
            {rawPrompt ? (
              <pre className="side-panel__pre">{rawPrompt}</pre>
            ) : (
              <p className="side-panel__empty">no prompt components.</p>
            )}
          </div>
        )}
      </section>

      <section className="side-panel__section">
        <div className="side-panel__section-head">
          <h3 className="side-panel__section-title">output schema</h3>
          <div className="side-panel__mode-toggle">
            <button
              type="button"
              className={`side-panel__mode-btn ${schemaMode === "rendered" ? "is-active" : ""}`}
              onClick={() => setSchemaMode("rendered")}
            >
              Table
            </button>
            <button
              type="button"
              className={`side-panel__mode-btn ${schemaMode === "json" ? "is-active" : ""}`}
              onClick={() => setSchemaMode("json")}
            >
              Raw JSON
            </button>
          </div>
        </div>
        <div className="side-panel__card">
          {node.outputSchema ? (
            schemaMode === "rendered" ? (
              <SchemaTable schema={node.outputSchema} />
            ) : (
              <pre className="side-panel__pre">{outputSchemaJson}</pre>
            )
          ) : (
            <p className="side-panel__empty">no output schema.</p>
          )}
        </div>
      </section>
    </>
  );
}
