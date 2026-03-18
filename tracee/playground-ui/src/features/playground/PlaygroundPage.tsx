import { useState } from "react";
import CreateRun from "../../components/CreateRun";
import PromptsList from "../../components/PromptsList";

type PlaygroundTab = "create" | "prompts";

export function PlaygroundPage() {
  const [tab, setTab] = useState<PlaygroundTab>("create");

  return (
    <div className="page-container flex-col">
      <div className="flex-col">
        <h2>Playground</h2>
        <p className="field__hint">
          Run prompt experiments and inspect saved prompt versions.
        </p>
      </div>

      <div className="tab-bar">
        <button
          className={`tab-bar__item ${tab === "create" ? "is-active" : ""}`}
          onClick={() => setTab("create")}
          type="button"
        >
          Run Playground
        </button>
        <button
          className={`tab-bar__item ${tab === "prompts" ? "is-active" : ""}`}
          onClick={() => setTab("prompts")}
          type="button"
        >
          Prompt Library
        </button>
      </div>

      {tab === "create" ? <CreateRun /> : <PromptsList />}
    </div>
  );
}
