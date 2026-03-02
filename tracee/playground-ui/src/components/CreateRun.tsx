import Ajv from 'ajv';
import React, { useState } from 'react';
import { playgroundAPI, promptAPI } from '../services/api';
import { PlaygroundRun, PlaygroundRunCreate } from '../types/playground';
import { PromptComponent, SchemaProperty } from '../types/prompt';
import SchemaBuilder, { toJsonSchema } from './SchemaBuilder';

const ajv = new Ajv();

const CreateRun: React.FC = () => {
  const [promptText, setPromptText] = useState<string>('');
  const [inputVars, setInputVars] = useState<string>('{}');
  const [model, setModel] = useState<string>('gpt-4');
  const [provider, setProvider] = useState<string>('openai');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [numRuns, setNumRuns] = useState<number>(1);

  const [schemaEnabled, setSchemaEnabled] = useState<boolean>(false);
  const [schemaProperties, setSchemaProperties] = useState<SchemaProperty[]>([]);

  const [results, setResults] = useState<Array<PlaygroundRun | null>>([]);
  const [runErrors, setRunErrors] = useState<Array<string | null>>([]);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Save-to-version prompt state
  const [showSavePrompt, setShowSavePrompt] = useState<boolean>(false);
  const [savedSchemaVersionId, setSavedSchemaVersionId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [lastRunContext, setLastRunContext] = useState<{
    promptId: string;
    components: PromptComponent[];
    inputVariables: Record<string, string>;
    outputSchema: Record<string, unknown> | null;
  } | null>(null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSetupError(null);
    setResults([]);
    setRunErrors([]);
    setShowSavePrompt(false);
    setSavedSchemaVersionId(null);
    setSaveError(null);
    setLoading(true);

    try {
      // Phase 1: parse input variables
      const inputVariables: Record<string, string> = JSON.parse(inputVars);

      // Phase 2: build output schema if enabled
      const outputSchema =
        schemaEnabled && schemaProperties.length > 0
          ? toJsonSchema(schemaProperties)
          : null;

      // Phase 3: create prompt + version once (shared across all runs)
      const timestamp = Date.now();
      const promptId = `prompt_${timestamp}`;

      await promptAPI.createPrompt({
        prompt_id: promptId,
        name: `Playground Prompt ${timestamp}`,
        description: 'Created from playground',
      });

      const components: PromptComponent[] = [
        {
          type: 'role',
          content: promptText,
          enabled: true,
        },
      ];

      const createdVersion = await promptAPI.createVersion(promptId, {
        name: 'v1',
        components,
        variables: inputVariables,
      });

      // Phase 4: fire N runs concurrently
      const requestData: PlaygroundRunCreate = {
        prompt_id: promptId,
        version_id: createdVersion.version_id,
        input_variables: inputVariables,
        model,
        provider,
        temperature,
        output_schema: outputSchema,
      };

      const settled = await Promise.allSettled(
        Array.from({ length: numRuns }, () => playgroundAPI.createRun(requestData))
      );

      setResults(settled.map(r => (r.status === 'fulfilled' ? r.value : null)));
      setRunErrors(
        settled.map(r =>
          r.status === 'rejected'
            ? r.reason?.response?.data?.detail || r.reason?.message || 'Run failed'
            : null
        )
      );

      // Phase 5: if schema was used, offer to save it
      if (outputSchema) {
        setLastRunContext({ promptId, components, inputVariables, outputSchema });
        setShowSavePrompt(true);
      }
    } catch (err: any) {
      // Only reached if Phase 1, 2, or 3 throws
      setSetupError(
        'Setup failed: ' + (err.response?.data?.detail || err.message || 'Unknown error')
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSchema = async () => {
    if (!lastRunContext) return;
    setSaveError(null);
    try {
      const newVersion = await promptAPI.createVersion(lastRunContext.promptId, {
        name: `schema-${Date.now()}`,
        components: lastRunContext.components,
        variables: lastRunContext.inputVariables,
        output_schema: lastRunContext.outputSchema,
      });
      setSavedSchemaVersionId(newVersion.version_id);
      setShowSavePrompt(false);
    } catch (err: any) {
      setSaveError(
        'Failed to save: ' + (err.response?.data?.detail || err.message || 'Unknown error')
      );
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
            placeholder="To use input variables, include {{variable_name}}"
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

        <div>
          <label>Number of Runs (max 5):</label>
          <input
            type="number"
            min="1"
            max="5"
            value={numRuns}
            onChange={(e) =>
              setNumRuns(Math.max(1, Math.min(5, parseInt(e.target.value) || 1)))
            }
          />
        </div>

        <div style={{ marginTop: '12px' }}>
          <label>
            <input
              type="checkbox"
              checked={schemaEnabled}
              onChange={e => setSchemaEnabled(e.target.checked)}
              style={{ marginRight: '6px' }}
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

        <button type="submit" disabled={loading} style={{ marginTop: '12px' }}>
          {loading ? 'Executing...' : 'Execute Run'}
        </button>
      </form>

      {setupError && <div style={{ color: 'red' }}>{setupError}</div>}

      {/* Save-to-version prompt */}
      {showSavePrompt && (
        <div style={savePromptStyle}>
          <p style={{ margin: '0 0 8px 0', fontWeight: 600 }}>
            Save schema to the current prompt version?
          </p>
          <p style={{ margin: '0 0 12px 0', fontSize: '0.9rem', color: '#555' }}>
            This will create a new version with the schema attached.
            The version will have the same components.
          </p>
          {saveError && <div style={{ color: 'red', marginBottom: '8px' }}>{saveError}</div>}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button type="button" onClick={handleSaveSchema}>Yes, save</button>
            <button type="button" onClick={() => setShowSavePrompt(false)}>No, skip</button>
          </div>
        </div>
      )}

      {savedSchemaVersionId && (
        <div style={{ marginTop: '8px', color: 'green' }}>
          Schema saved — new version ID: <strong>{savedSchemaVersionId}</strong>
        </div>
      )}

      {results.length > 0 && (
        <div>
          <h3>Results</h3>
          {results.map((run, i) => (
            <div key={i} style={{ marginBottom: '16px' }}>
              <h4>Run {i + 1}</h4>
              {runErrors[i] ? (
                <div style={{ color: 'red' }}>{runErrors[i]}</div>
              ) : run ? (
                <RunOutput run={run} />
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface RunOutputProps {
  run: PlaygroundRun;
}

const RunOutput: React.FC<RunOutputProps> = ({ run }) => {
  if (!run.output_schema) {
    return (
      <div>
        <strong>Output</strong>
        <pre>{run.output}</pre>
      </div>
    );
  }

  let parsed: unknown = null;
  let parseFailed = false;
  let validationErrors: { message?: string; instancePath?: string }[] = [];

  try {
    parsed = JSON.parse(run.output);
    const validate = ajv.compile(run.output_schema as object);
    validate(parsed);
    validationErrors = validate.errors ?? [];
  } catch {
    parseFailed = true;
  }

  return (
    <div>
      <strong>Structured Output</strong>
      {validationErrors.length > 0 && (
        <div style={validationErrorStyle}>
          ⚠ Schema validation issues:
          <ul style={{ margin: '4px 0 0 0', paddingLeft: '20px' }}>
            {validationErrors.map((e, i) => (
              <li key={i}>
                {e.instancePath ? `'${e.instancePath.replace(/^\//, '')}' ` : ''}
                {e.message}
              </li>
            ))}
          </ul>
        </div>
      )}
      {parseFailed ? (
        <>
          <span style={{ color: '#888', fontSize: '0.85rem' }}>(parse failed — raw)</span>
          <pre>{run.output}</pre>
        </>
      ) : (
        <pre>{JSON.stringify(parsed, null, 2)}</pre>
      )}
    </div>
  );
};

const savePromptStyle: React.CSSProperties = {
  marginTop: '16px',
  padding: '16px',
  border: '1px solid #ccc',
  borderRadius: '4px',
  background: '#f9f9f9',
};

const validationErrorStyle: React.CSSProperties = {
  marginTop: '8px',
  marginBottom: '8px',
  padding: '8px 12px',
  background: '#fff3cd',
  border: '1px solid #ffc107',
  borderRadius: '4px',
  fontSize: '0.9rem',
};

export default CreateRun;
