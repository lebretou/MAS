import React from 'react';
import { PromptComponent, SchemaProperty } from '../types/prompt';
import SchemaBuilder, { toJsonSchema } from './SchemaBuilder';
import PromptComponentEditor from './PromptComponentEditor';
import { useRunExecution } from '../hooks/useRunExecution';
import { PlaygroundRun } from '../types/playground';

interface Props {
  onRunComplete: (results: Array<PlaygroundRun | null>, errors: Array<string | null>) => void;
}

const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ['gpt-4', 'gpt-3.5-turbo'],
  anthropic: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
};

const PromptForm: React.FC<Props> = ({ onRunComplete }) => {
  const [promptComponents, setPromptComponents] = React.useState<PromptComponent[]>([
    { type: 'role', content: '', enabled: true },
  ]);
  const [inputVars, setInputVars] = React.useState('{}');
  const [provider, setProvider] = React.useState('openai');
  const [model, setModel] = React.useState('gpt-4');
  const [temperature, setTemperature] = React.useState(0.7);
  const [numRuns, setNumRuns] = React.useState(1);

  const [schemaEnabled, setSchemaEnabled] = React.useState(false);
  const [schemaProperties, setSchemaProperties] = React.useState<SchemaProperty[]>([]);

  const [jsonError, setJsonError] = React.useState<string | null>(null);

  const {
    loading, setupError, showSavePrompt,
    savedSchemaVersionId, saveError,
    execute, saveSchema, dismissSavePrompt,
  } = useRunExecution(onRunComplete);

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    const models = PROVIDER_MODELS[newProvider];
    if (models && models.length > 0) {
      setModel(models[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setJsonError(null);

    let inputVariables: Record<string, string>;
    try {
      inputVariables = JSON.parse(inputVars);
    } catch {
      setJsonError('Input Variables must be valid JSON (e.g., {"key": "value"})');
      return;
    }

    const components = promptComponents.filter(c => c.enabled && c.content.trim());
    const outputSchema =
      schemaEnabled && schemaProperties.length > 0
        ? toJsonSchema(schemaProperties)
        : null;

    execute({
      components,
      inputVariables,
      model,
      provider,
      temperature,
      numRuns,
      outputSchema,
    });
  };

  return (
    <div>
      <div className="card">
        <div className="card__body">
          <form onSubmit={handleSubmit} className="flex-col create-run__form">
            {/* Prompt Components */}
            <div className="field">
              <label className="field__label">Prompt Components</label>
              <PromptComponentEditor
                components={promptComponents}
                onChange={setPromptComponents}
              />
            </div>

            {/* Input Variables */}
            <div className="field">
              <label className="field__label">Input Variables</label>
              <textarea
                className={`textarea textarea--code${jsonError ? ' input--error' : ''}`}
                value={inputVars}
                onChange={(e) => { setInputVars(e.target.value); setJsonError(null); }}
                rows={3}
                required
              />
              {jsonError && <span className="field__error">{jsonError}</span>}
            </div>

            {/* Provider + Model */}
            <div className="form-grid">
              <div className="field">
                <label className="field__label">Provider</label>
                <select
                  className="select"
                  value={provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                >
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                </select>
              </div>

              <div className="field">
                <label className="field__label">Model</label>
                <select
                  className="select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  {(PROVIDER_MODELS[provider] ?? []).map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Temperature + Runs */}
            <div className="form-grid">
              <div className="field">
                <label className="field__label">Temperature: {temperature}</label>
                <input
                  className="input"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>

              <div className="field">
                <label className="field__label">Number of Runs (max 10)</label>
                <input
                  className="input"
                  type="number"
                  min="1"
                  max="10"
                  value={numRuns}
                  onChange={(e) =>
                    setNumRuns(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))
                  }
                />
              </div>
            </div>

            {/* Schema toggle */}
            <div>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={schemaEnabled}
                  onChange={e => setSchemaEnabled(e.target.checked)}
                />
                Enable Output Schema
              </label>
              {schemaEnabled && (
                <SchemaBuilder
                  properties={schemaProperties}
                  onChange={setSchemaProperties}
                />
              )}
            </div>

            <hr className="divider" />

            <div className="flex-row create-run__actions">
              <button
                type="submit"
                className="btn btn--primary"
                disabled={loading}
              >
                {loading && <span className="spinner create-run__spinner" />}
                {loading ? 'Executing...' : 'Execute Run'}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Setup error */}
      {setupError && (
        <div className="alert alert--danger create-run__mt">
          <span className="alert__icon">!</span>
          {setupError}
        </div>
      )}

      {/* JSON validation error */}
      {jsonError && !setupError && (
        <div className="alert alert--danger create-run__mt">
          <span className="alert__icon">!</span>
          {jsonError}
        </div>
      )}

      {/* Save schema prompt */}
      {showSavePrompt && (
        <div className="card create-run__mt">
          <div className="card__body">
            <p className="create-run__save-text">
              Save schema to the current prompt version?
            </p>
            <p className="field__hint">
              This will create a new version with the schema attached.
            </p>
            {saveError && (
              <div className="alert alert--danger">
                <span className="alert__icon">!</span>
                {saveError}
              </div>
            )}
            <div className="flex-row create-run__save-actions">
              <button className="btn btn--primary btn--sm" type="button" onClick={saveSchema}>
                Yes, save
              </button>
              <button className="btn btn--ghost btn--sm" type="button" onClick={dismissSavePrompt}>
                No, skip
              </button>
            </div>
          </div>
        </div>
      )}

      {savedSchemaVersionId && (
        <div className="alert alert--success create-run__mt">
          <span className="alert__icon">ok</span>
          Schema saved — new version ID: <strong>{savedSchemaVersionId}</strong>
        </div>
      )}
    </div>
  );
};

export default PromptForm;
