import { Panel } from "@xyflow/react";
import intentIcon from "../../../assets/intent.svg";
import executionIcon from "../../../assets/execution.svg";
import { useLayer } from "../../../context/LayerContext";

export function LayerToggle() {
  const { layer, setLayer } = useLayer();

  return (
    <Panel position="top-left" className="layer-toggle-panel">
      <div className="layer-toggle">
        <div
          className={`layer-toggle__indicator ${layer === "execution" ? "is-right" : ""}`}
        />
        <button
          type="button"
          className={`layer-toggle__btn ${layer === "intent" ? "is-active" : ""}`}
          onClick={() => setLayer("intent")}
          aria-label="Intent Layer"
        >
          <img src={intentIcon} alt="Intent" className="layer-toggle__icon" />
        </button>
        <button
          type="button"
          className={`layer-toggle__btn ${layer === "execution" ? "is-active" : ""}`}
          onClick={() => setLayer("execution")}
          aria-label="Execution Layer"
        >
          <img src={executionIcon} alt="Execution" className="layer-toggle__icon" />
        </button>
      </div>
    </Panel>
  );
}
