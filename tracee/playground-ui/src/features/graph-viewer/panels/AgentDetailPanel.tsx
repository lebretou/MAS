import { useEffect } from "react";
import cognitionIcon from "../../../assets/cognition.svg";
import { useSidebar } from "../../../context/SidebarContext";
import { useLayer } from "../../../context/LayerContext";
import { IntentDetails } from "./IntentDetails";
import { ExecutionDetails } from "./ExecutionDetails";

interface Props {
  onRequestClose: () => void;
}

export function AgentDetailPanel({ onRequestClose }: Props) {
  const { selectedNode } = useSidebar();
  const { layer } = useLayer();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selectedNode) onRequestClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onRequestClose, selectedNode]);

  if (!selectedNode) return null;

  return (
    <>
      <div className="side-panel__backdrop" onClick={onRequestClose} />
      <div className="side-panel" onClick={(e) => e.stopPropagation()}>
        <div className="side-panel__header">
          <div className="side-panel__header-main">
            <div className="side-panel__icon-wrapper">
              <img src={cognitionIcon} alt="" className="side-panel__icon" />
            </div>
            <div className="side-panel__header-text">
              <h2 className="side-panel__title">{selectedNode.label}</h2>
              <p className="side-panel__subtitle">
                {layer === "intent" ? "agent details" : "execution details"}
              </p>
            </div>
          </div>
          <button className="side-panel__close" onClick={onRequestClose} aria-label="close panel">
            &times;
          </button>
        </div>
        <div className="side-panel__content">
          {layer === "intent" ? (
            <IntentDetails node={selectedNode} />
          ) : (
            <ExecutionDetails node={selectedNode} />
          )}
        </div>
      </div>
    </>
  );
}
