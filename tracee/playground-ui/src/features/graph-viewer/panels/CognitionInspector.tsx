import { Panel } from "@xyflow/react";
import type { TraceCognition } from "../../../types/cognition";
import { CognitionText } from "../../../components/CognitionText";
import { useSidebar } from "../../../context/SidebarContext";

interface Props {
  cognition: TraceCognition | null;
  loading: boolean;
  analyzing: boolean;
  onAnalyze: () => void;
}

export function CognitionInspector({ cognition, loading, analyzing, onAnalyze }: Props) {
  const { openSidebar } = useSidebar();

  const handleAgentClick = (agentId: string) => {
    if (!cognition) return;
    const cog = cognition.node_cognitions[agentId];
    if (cog) {
      openSidebar(agentId, {
        label: agentId,
        nodeType: "agent",
        cognition: cog,
      });
    }
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

  const nodeEntries = Object.values(cognition.node_cognitions);

  return (
    <Panel position="bottom-center" className="cognition-inspector">
      <div className="cognition-inspector__narrative">
        <CognitionText text={cognition.narrative} onAgentClick={handleAgentClick} />
      </div>
      <div className="cognition-inspector__divider" />
      <div className="cognition-inspector__nodes">
        {nodeEntries.map((cog) => (
          <div
            key={cog.agent_id}
            className="cognition-inspector__node-row"
            onClick={() => handleAgentClick(cog.agent_id)}
          >
            <span className="cognition-inspector__node-name">{cog.agent_id}</span>
          </div>
        ))}
        <button
          className="cognition-inspector__rerun-btn"
          onClick={onAnalyze}
          disabled={analyzing}
        >
          {analyzing ? "analyzing..." : "re-run"}
        </button>
      </div>
    </Panel>
  );
}
