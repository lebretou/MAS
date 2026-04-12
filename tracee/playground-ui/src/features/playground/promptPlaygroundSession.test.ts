import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearStoredPromptVersionSelection,
  readStoredPromptSession,
  writeStoredPromptSession,
} from './promptPlaygroundSession';

function createSessionStorageMock() {
  const store = new Map<string, string>();

  return {
    getItem(key: string) {
      return store.has(key) ? store.get(key) ?? null : null;
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
    removeItem(key: string) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
  };
}

describe('promptPlaygroundSession', () => {
  beforeEach(() => {
    vi.stubGlobal('window', {
      sessionStorage: createSessionStorageMock(),
    } as unknown as Window);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('stores and reads an existing prompt selection with a specific version', () => {
    writeStoredPromptSession({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });

    expect(readStoredPromptSession()).toEqual({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });
  });

  it('clears only the requested version when consuming the handoff', () => {
    writeStoredPromptSession({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });

    clearStoredPromptVersionSelection();

    expect(readStoredPromptSession()).toEqual({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: '',
    });
  });

  it('preserves the requested version when syncing prompt selection without changing it', () => {
    writeStoredPromptSession({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });

    writeStoredPromptSession({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
    });

    expect(readStoredPromptSession()).toEqual({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });
  });

  it('clears prompt selection when switching back to a new prompt workflow', () => {
    writeStoredPromptSession({
      promptWorkflow: 'existing',
      selectedPromptId: 'prompt-123',
      selectedVersionId: 'v7',
    });

    writeStoredPromptSession({
      promptWorkflow: 'new',
      selectedPromptId: '',
    });

    expect(readStoredPromptSession()).toEqual({
      promptWorkflow: 'new',
      selectedPromptId: '',
      selectedVersionId: '',
    });
  });
});
