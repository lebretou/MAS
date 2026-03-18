import PromptsList from "./components/PromptsList";

export function PromptsPage() {
  return (
    <div className="page-container flex-col playground-page">
      <div className="flex-col">
        <h2>Prompts</h2>
        <p className="field__hint">
          Browse saved prompts and inspect their version history.
        </p>
      </div>
      <PromptsList />
    </div>
  );
}
