import React, { useState } from 'react';
import { playgroundAPI, promptAPI } from '../services/api';
import { PlaygroundRun, PlaygroundRunCreate } from '../types/playground';
import { PromptComponent } from '../types/prompt';

const CreateRun: React.FC = () => {
  const [promptText, setPromptText] = useState<string>('');
  const [inputVars, setInputVars] = useState<string>('{}');
  const [model, setModel] = useState<string>('gpt-4');
  const [provider, setProvider] = useState<string>('openai');
  const [temperature, setTemperature] = useState<number>(0.7);
  
  const [result, setResult] = useState<PlaygroundRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const inputVariables: Record<string, string> = JSON.parse(inputVars);
      
      const timestamp = Date.now();
      const promptId = `prompt_${timestamp}`;
      
      await promptAPI.createPrompt({
        prompt_id: promptId,
        name: `Playground Prompt ${timestamp}`,
        description: "Created from playground",
      });
      
      const components: PromptComponent[] = [
        {
          type: "role", // Using "role" as default component type
          content: promptText,
          enabled: true,
        }
      ];
      
      await promptAPI.createVersion(promptId, {
        name: "v1",
        components: components,
        variables: inputVariables,
      });
      
      const requestData: PlaygroundRunCreate = {
        prompt_id: promptId,
        version_id: "v1",
        input_variables: inputVariables,
        model,
        provider,
        temperature,
      };
      
      const runResult = await playgroundAPI.createRun(requestData);
      setResult(runResult);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to create run');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Create Playground Run</h2>
      
      <form onSubmit={handleSubmit}>
        <div>
          <label>Prompt:</label>
          <textarea
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            rows={8}
            placeholder='To use input variables, include {{variable_name}}'
            required
          />
        </div>

        <div>
          <label>Input Variables:</label>
          <textarea
            value={inputVars}
            onChange={(e) => setInputVars(e.target.value)}
            rows={4}
            required
          />
        </div>

        <div>
          <label>Model:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="gpt-4">GPT-4</option>
          </select>
        </div>

        <div>
          <label>Provider:</label>
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            <option value="openai">OpenAI</option>
          </select>
        </div>

        <div>
          <label>Temperature: {temperature}</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Executing...' : 'Execute Run'}
        </button>
      </form>

      {error && <div style={{ color: 'red' }}>{error}</div>}
      
      {result && (
        <div>
          <h3>Result</h3>
          <h4>Output (LLM response)</h4>
          <pre>{result.output}</pre>
        </div>
      )}
    </div>
  );
};

export default CreateRun;