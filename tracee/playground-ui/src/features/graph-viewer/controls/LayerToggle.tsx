import intentIcon from "../../../assets/intent.svg";
import executionIcon from "../../../assets/execution.svg";
import cognitionIcon from "../../../assets/icon-cognition-layer-button.svg";
import { useLayer } from "../../../context/LayerContext";

const INDICATOR_POSITION = {
  intent: "",
  execution: "is-center",
  cognition: "is-right",
} as const;

export function LayerToggle() {
  const { layer, setLayer } = useLayer();

  return (
    <div className="layer-toggle layer-toggle--three">
      <div
        className={`layer-toggle__indicator ${INDICATOR_POSITION[layer]}`}
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
      <button
        type="button"
        className={`layer-toggle__btn ${layer === "cognition" ? "is-active" : ""}`}
        onClick={() => setLayer("cognition")}
        aria-label="Cognition Layer"
      >
        <img src={cognitionIcon} alt="Cognition" className="layer-toggle__icon" />
      </button>
    </div>
  );
}
