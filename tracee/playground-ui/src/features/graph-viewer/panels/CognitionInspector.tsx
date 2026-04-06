import { Panel } from "@xyflow/react";
import type { TraceCognition } from "../../../types/cognition";
import { CognitionText } from "../../../components/CognitionText";
import { useSidebar } from "../../../context/SidebarContext";
import iconSummary from "../../../assets/icon-summarypanel.svg";

interface Props {
  cognition: TraceCognition | null;
  loading: boolean;
  analyzing: boolean;
  onAnalyze: () => void;
}

export function CognitionInspector({ cognition, loading, analyzing, onAnalyze }: Props) {
  const { openSidebar } = useSidebar();

  const openAgentSidebar = (agentId: string, chipExpansion?: { type: string; value: string }) => {
    if (!cognition) return;
    const cog = cognition.node_cognitions[agentId];
    if (cog) {
      openSidebar(agentId, { label: agentId, nodeType: "agent", cognition: cog }, chipExpansion);
    }
  };

  // find the agent whose description contains a given chip tag
  const findOwningAgent = (chipType: string, chipValue: string): string | null => {
    if (!cognition) return null;
    const pattern = `{${chipType}:${chipValue}}`;
    for (const [agentId, cog] of Object.entries(cognition.node_cognitions)) {
      if (cog.description.includes(pattern)) return agentId;
    }
    return null;
  };

  const handleAgentClick = (agentId: string) => openAgentSidebar(agentId);

  const handleToolClick = (value: string) => {
    const agentId = findOwningAgent("tool", value);
    if (agentId) openAgentSidebar(agentId, { type: "tool", value });
  };

  const handleStateClick = (value: string) => {
    const agentId = findOwningAgent("state", value);
    if (agentId) openAgentSidebar(agentId, { type: "state", value });
  };

  if (loading) {
    return (
      <Panel position="bottom-center" className="cognition-inspector">
        <div className="cognition-inspector__empty">
          loading cognition...
        </div>
      </Panel>
    );
  }

  if (!cognition) {
    return (
      <Panel position="bottom-center" className="cognition-inspector">
        <div className="cognition-inspector__empty">
          <div style={{ textAlign: "center" }}>
            <div style={{ marginBottom: 8, color: "#6b7280" }}>no cognition analysis yet</div>
            <button
              className="cognition-inspector__analyze-btn"
              onClick={onAnalyze}
              disabled={analyzing}
            >
              {analyzing ? "analyzing..." : "run analysis"}
            </button>
          </div>
        </div>
      </Panel>
    );
  }

  return (
    <Panel position="bottom-center" className="cognition-inspector">
      <div className="cognition-inspector__narrative">
        <div className="cognition-inspector__header">
          <img src={iconSummary} alt="" className="cognition-inspector__title-icon" />
          <span className="cognition-inspector__title">Trace Summary</span>
        </div>
        <div className="cognition-inspector__body">
          <CognitionText
            text={cognition.narrative}
            onAgentClick={handleAgentClick}
            onToolClick={handleToolClick}
            onStateClick={handleStateClick}
          />
        </div>
      </div>
    </Panel>
  );
}
