import CreateRun from "./components/CreateRun";
import iconPlayground from "../../assets/icon-playground.svg";

export function PlaygroundPage() {
  return (
    <div className="page-container flex-col playground-page">
      <div className="flex-col">
        <div className="playground-page__title-row">
          <img src={iconPlayground} alt="" className="playground-page__title-icon" aria-hidden />
          <h2>Playground</h2>
        </div>
        <p className="field__hint">
          Run prompt experiments and inspect the output side by side.
        </p>
      </div>
      <CreateRun />
    </div>
  );
}
