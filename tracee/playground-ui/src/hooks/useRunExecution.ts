import { useState, useCallback } from 'react';
import axios from 'axios';
import { PromptComponent } from '../types/prompt';
import { PlaygroundRun, PlaygroundRunCreate } from '../types/playground';
import { playgroundAPI, promptAPI } from '../services/api';

function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    return err.response?.data?.detail ?? err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Unknown error';
}

interface RunExecutionParams {
  components: PromptComponent[];
  inputVariables: Record<string, string>;
  model: string;
  provider: string;
  temperature: number;
  numRuns: number;
  outputSchema: Record<string, unknown> | null;
}

interface SaveContext {
  promptId: string;
  components: PromptComponent[];
  inputVariables: Record<string, string>;
  outputSchema: Record<string, unknown> | null;
}

interface RunExecutionState {
  loading: boolean;
  setupError: string | null;
  showSavePrompt: boolean;
  savedSchemaVersionId: string | null;
  saveError: string | null;
}

export function useRunExecution(
  onRunComplete: (results: Array<PlaygroundRun | null>, errors: Array<string | null>) => void,
) {
  const [state, setState] = useState<RunExecutionState>({
    loading: false,
    setupError: null,
    showSavePrompt: false,
    savedSchemaVersionId: null,
    saveError: null,
  });
  const [lastRunContext, setLastRunContext] = useState<SaveContext | null>(null);

  const execute = useCallback(async (params: RunExecutionParams) => {
    setState({
      loading: true,
      setupError: null,
      showSavePrompt: false,
      savedSchemaVersionId: null,
      saveError: null,
    });

    try {
      const timestamp = Date.now();
      const promptId = `prompt_${timestamp}`;

      await promptAPI.createPrompt({
        prompt_id: promptId,
        name: `Playground Prompt ${timestamp}`,
        description: 'Created from playground',
      });

      const createdVersion = await promptAPI.createVersion(promptId, {
        name: 'v1',
        components: params.components,
        variables: params.inputVariables,
        output_schema: params.outputSchema,
      });

      const requestData: PlaygroundRunCreate = {
        prompt_id: promptId,
        version_id: createdVersion.version_id,
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

      if (params.outputSchema) {
        setLastRunContext({
          promptId,
          components: params.components,
          inputVariables: params.inputVariables,
          outputSchema: params.outputSchema,
        });
        setState(prev => ({ ...prev, loading: false, showSavePrompt: true }));
      } else {
        setState(prev => ({ ...prev, loading: false }));
      }
    } catch (err: unknown) {
      setState(prev => ({
        ...prev,
        loading: false,
        setupError: 'Setup failed: ' + getErrorMessage(err),
      }));
    }
  }, [onRunComplete]);

  const saveSchema = useCallback(async () => {
    if (!lastRunContext) return;
    setState(prev => ({ ...prev, saveError: null }));
    try {
      const newVersion = await promptAPI.createVersion(lastRunContext.promptId, {
        name: `schema-${Date.now()}`,
        components: lastRunContext.components,
        variables: lastRunContext.inputVariables,
        output_schema: lastRunContext.outputSchema,
      });
      setState(prev => ({
        ...prev,
        savedSchemaVersionId: newVersion.version_id,
        showSavePrompt: false,
      }));
    } catch (err: unknown) {
      setState(prev => ({
        ...prev,
        saveError: 'Failed to save: ' + getErrorMessage(err),
      }));
    }
  }, [lastRunContext]);

  const dismissSavePrompt = useCallback(() => {
    setState(prev => ({ ...prev, showSavePrompt: false }));
  }, []);

  return { ...state, execute, saveSchema, dismissSavePrompt };
}
