import { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import type { PromptComponent, PromptTool } from '../types/prompt';
import type { PlaygroundRun, PlaygroundRunCreate } from '../types/playground';
import { playgroundAPI, promptAPI } from '../services/api';

function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (!item || typeof item !== 'object') return String(item);
          const entry = item as { loc?: unknown[]; msg?: string };
          const path = Array.isArray(entry.loc) ? entry.loc.join('.') : '';
          return path ? `${path}: ${entry.msg ?? 'invalid value'}` : (entry.msg ?? 'invalid value');
        })
        .join('; ');
    }
    return detail ?? err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Unknown error';
}

interface RunExecutionParams {
  components: PromptComponent[];
  tools: PromptTool[];
  inputVariables: Record<string, string>;
  model: string;
  provider: string;
  temperature: number;
  numRuns: number;
  outputSchema: Record<string, unknown> | null;
}

interface RunExecutionState {
  loading: boolean;
  setupError: string | null;
}

interface CachedPromptVersion {
  promptId: string;
  versionId: string;
}

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(',')}]`;
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, nestedValue]) => `${JSON.stringify(key)}:${stableSerialize(nestedValue)}`);

    return `{${entries.join(',')}}`;
  }

  return JSON.stringify(value);
}

function getPromptSignature(params: RunExecutionParams): string {
  return stableSerialize({
    components: params.components,
    tools: params.tools,
    outputSchema: params.outputSchema,
  });
}

export function useRunExecution(
  onRunComplete: (results: Array<PlaygroundRun | null>, errors: Array<string | null>) => void,
) {
  const [state, setState] = useState<RunExecutionState>({
    loading: false,
    setupError: null,
  });
  const cachedPromptVersionsRef = useRef<Map<string, CachedPromptVersion>>(new Map());

  const execute = useCallback(async (params: RunExecutionParams) => {
    setState({
      loading: true,
      setupError: null,
    });

    try {
      const signature = getPromptSignature(params);
      const cachedPromptVersion = cachedPromptVersionsRef.current.get(signature);

      let promptId = '';
      let versionId = '';

      if (cachedPromptVersion) {
        promptId = cachedPromptVersion.promptId;
        versionId = cachedPromptVersion.versionId;
      } else {
        const timestamp = Date.now();
        promptId = `prompt_${timestamp}`;

        await promptAPI.createPrompt({
          prompt_id: promptId,
          name: `Playground Prompt ${timestamp}`,
          description: 'Created from playground',
        });

        const createdVersion = await promptAPI.createVersion(promptId, {
          name: 'v1',
          components: params.components,
          variables: null,
          output_schema: params.outputSchema,
          tools: params.tools,
        });

        versionId = createdVersion.version_id;
        cachedPromptVersionsRef.current.set(signature, {
          promptId,
          versionId,
        });
      }

      const requestData: PlaygroundRunCreate = {
        prompt_id: promptId,
        version_id: versionId,
        input_variables: params.inputVariables,
        model: params.model,
        provider: params.provider,
        temperature: params.temperature,
        output_schema: params.outputSchema,
      };

      const settled = await Promise.allSettled(
        Array.from({ length: params.numRuns }, () => playgroundAPI.createRun(requestData))
      );

      const results = settled.map(r => (r.status === 'fulfilled' ? r.value : null));
      const errors = settled.map(r =>
        r.status === 'rejected' ? getErrorMessage(r.reason) : null
      );

      onRunComplete(results, errors);
      setState(prev => ({ ...prev, loading: false }));
    } catch (err: unknown) {
      setState(prev => ({
        ...prev,
        loading: false,
        setupError: 'Setup failed: ' + getErrorMessage(err),
      }));
    }
  }, [onRunComplete]);

  return { ...state, execute };
}
