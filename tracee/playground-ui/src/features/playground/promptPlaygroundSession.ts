export type PromptWorkflow = 'new' | 'existing';

interface StoredPromptSessionOptions {
  promptWorkflow: PromptWorkflow;
  selectedPromptId: string;
  selectedVersionId?: string;
}

interface StoredPromptSession {
  promptWorkflow: PromptWorkflow;
  selectedPromptId: string;
  selectedVersionId: string;
}

const PLAYGROUND_PROMPT_WORKFLOW_STORAGE_KEY = 'tracee:playground:prompt-workflow';
const PLAYGROUND_SELECTED_PROMPT_STORAGE_KEY = 'tracee:playground:selected-prompt-id';
const PLAYGROUND_SELECTED_VERSION_STORAGE_KEY = 'tracee:playground:selected-version-id';

export function readStoredPromptWorkflow(): PromptWorkflow {
  if (typeof window === 'undefined') {
    return 'new';
  }

  return window.sessionStorage.getItem(PLAYGROUND_PROMPT_WORKFLOW_STORAGE_KEY) === 'existing'
    ? 'existing'
    : 'new';
}

export function readStoredPromptSession(): StoredPromptSession {
  const promptWorkflow = readStoredPromptWorkflow();

  if (typeof window === 'undefined') {
    return {
      promptWorkflow,
      selectedPromptId: '',
      selectedVersionId: '',
    };
  }

  if (promptWorkflow !== 'existing') {
    return {
      promptWorkflow,
      selectedPromptId: '',
      selectedVersionId: '',
    };
  }

  return {
    promptWorkflow,
    selectedPromptId: window.sessionStorage.getItem(PLAYGROUND_SELECTED_PROMPT_STORAGE_KEY) ?? '',
    selectedVersionId: window.sessionStorage.getItem(PLAYGROUND_SELECTED_VERSION_STORAGE_KEY) ?? '',
  };
}

export function writeStoredPromptSession({
  promptWorkflow,
  selectedPromptId,
  selectedVersionId,
}: StoredPromptSessionOptions) {
  if (typeof window === 'undefined') {
    return;
  }

  window.sessionStorage.setItem(PLAYGROUND_PROMPT_WORKFLOW_STORAGE_KEY, promptWorkflow);

  if (promptWorkflow !== 'existing' || !selectedPromptId) {
    window.sessionStorage.removeItem(PLAYGROUND_SELECTED_PROMPT_STORAGE_KEY);
    window.sessionStorage.removeItem(PLAYGROUND_SELECTED_VERSION_STORAGE_KEY);
    return;
  }

  window.sessionStorage.setItem(PLAYGROUND_SELECTED_PROMPT_STORAGE_KEY, selectedPromptId);

  if (selectedVersionId === undefined) {
    return;
  }

  if (!selectedVersionId) {
    window.sessionStorage.removeItem(PLAYGROUND_SELECTED_VERSION_STORAGE_KEY);
    return;
  }

  window.sessionStorage.setItem(PLAYGROUND_SELECTED_VERSION_STORAGE_KEY, selectedVersionId);
}

export function clearStoredPromptVersionSelection() {
  if (typeof window === 'undefined') {
    return;
  }

  window.sessionStorage.removeItem(PLAYGROUND_SELECTED_VERSION_STORAGE_KEY);
}
