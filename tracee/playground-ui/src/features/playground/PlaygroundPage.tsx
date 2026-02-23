import { useState } from "react";
import { createRun } from "../../api/playground";
import { createPrompt, createVersion } from "../../api/prompts";
import type { PlaygroundRun } from "../../types/playground";
import type { PromptComponent } from "../../types/prompt";

export function PlaygroundPage() {
  const [promptText, setPromptText] = useState("");
  const [inputVars, setInputVars] = useState("{}");
  const [model, setModel] = useState("gpt-4");
  const [provider, setProvider] = useState("openai");
  const [temperature, setTemperature] = useState(0.7);

  const [result, setResult] = useState<PlaygroundRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    const inputVariables: Record<string, string> = JSON.parse(inputVars);
    const timestamp = Date.now();
    const promptId = `prompt_${timestamp}`;

    await createPrompt({
      prompt_id: promptId,
      name: `Playground Prompt ${timestamp}`,
      description: "Created from playground",
    });

    const components: PromptComponent[] = [
      { type: "role", content: promptText, enabled: true },
    ];

    await createVersion(promptId, {
      name: "v1",
      components,
      variables: inputVariables,
    });

    const runResult = await createRun({
      prompt_id: promptId,
      version_id: "v1",
      input_variables: inputVariables,
      model,
      provider,
      temperature,
    });
    setResult(runResult);
    setLoading(false);
  };

  return (
    <div style={{ padding: 24, maxWidth: 720 }}>
      <h2 style={{ marginBottom: 16 }}>Playground</h2>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div>
          <label style={{ display: "block", marginBottom: 4, fontSize: 13, color: "#6b7280" }}>Prompt</label>
          <textarea
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            rows={8}
            placeholder={'Include {{variable_name}} for input variables'}
            required
            style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #e5e7eb", fontFamily: "inherit" }}
          />
        </div>

        <div>
          <label style={{ display: "block", marginBottom: 4, fontSize: 13, color: "#6b7280" }}>Input variables (JSON)</label>
          <textarea
            value={inputVars}
            onChange={(e) => setInputVars(e.target.value)}
            rows={3}
            required
            style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #e5e7eb", fontFamily: "monospace" }}
          />
        </div>

        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: 4, fontSize: 13, color: "#6b7280" }}>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #e5e7eb" }}>
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-4o-mini">GPT-4o Mini</option>
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", marginBottom: 4, fontSize: 13, color: "#6b7280" }}>Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid #e5e7eb" }}>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </div>
        </div>

        <div>
          <label style={{ display: "block", marginBottom: 4, fontSize: 13, color: "#6b7280" }}>Temperature: {temperature}</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 20px",
            borderRadius: 8,
            border: "none",
            background: "#219ebc",
            color: "white",
            fontWeight: 600,
            cursor: loading ? "wait" : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "Running..." : "Run"}
        </button>
      </form>

      {error && <p style={{ color: "#ef4444", marginTop: 12 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 8 }}>Output</h3>
          <pre style={{
            background: "#f8f9fb",
            padding: 16,
            borderRadius: 12,
            border: "1px solid #e5e7eb",
            whiteSpace: "pre-wrap",
            fontSize: 13,
          }}>
            {result.output}
          </pre>
        </div>
      )}
    </div>
  );
}
