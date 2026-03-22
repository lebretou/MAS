import React, { useState, useCallback, useEffect } from 'react';
import type { PlaygroundRun } from '../../../types/playground';
import { useRunAnalysis } from '../../../hooks/useRunAnalysis';
import type { AnchorPoint } from '../../../hooks/useRunAnalysis';
import PromptForm from './PromptForm';
import ResultsComparison from './ResultsComparison';
import RunDetailView from './RunDetailView';

const PLAYGROUND_RESULTS_CACHE_KEY = 'tracee:playground:last-results';

interface CachedPlaygroundResults {
  version: 1;
  results: Array<PlaygroundRun | null>;
  runErrors: Array<string | null>;
  anchor: AnchorPoint | null;
}

function readCachedPlaygroundResults(): CachedPlaygroundResults | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const rawValue = window.sessionStorage.getItem(PLAYGROUND_RESULTS_CACHE_KEY);

    if (!rawValue) {
      return null;
    }

    const parsed = JSON.parse(rawValue) as CachedPlaygroundResults;

    if (parsed?.version !== 1 || !Array.isArray(parsed.results) || !Array.isArray(parsed.runErrors)) {
      return null;
    }

    return {
      version: 1,
      results: parsed.results,
      runErrors: parsed.runErrors,
      anchor: parsed.anchor ?? null,
    };
  } catch {
    return null;
  }
}

const CreateRun: React.FC = () => {
  const [cachedResults] = useState<CachedPlaygroundResults | null>(() => readCachedPlaygroundResults());
  const [workspaceMode, setWorkspaceMode] = useState<'author' | 'analysis'>('author');
  const [results, setResults] = useState<Array<PlaygroundRun | null>>(cachedResults?.results ?? []);
  const [runErrors, setRunErrors] = useState<Array<string | null>>(cachedResults?.runErrors ?? []);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [anchor, setAnchor] = useState<AnchorPoint | null>(cachedResults?.anchor ?? null);

  const analysis = useRunAnalysis(results, runErrors, anchor);

  const handleRunComplete = useCallback((
    newResults: Array<PlaygroundRun | null>,
    newErrors: Array<string | null>,
  ) => {
    setResults(newResults);
    setRunErrors(newErrors);
    setSelectedRun(null);
    setWorkspaceMode('analysis');
    setAnchor((currentAnchor) => {
      if (currentAnchor?.source !== 'run') {
        return currentAnchor;
      }

      return {
        ...currentAnchor,
        label: 'Promoted anchor',
        source: 'example',
        runIndex: null,
      };
    });
  }, []);

  const handleSelectRun = useCallback((index: number) => {
    setSelectedRun(index);
  }, []);

  const handleBack = useCallback(() => {
    setSelectedRun(null);
  }, []);

  const handleBackToEdit = useCallback(() => {
    setSelectedRun(null);
    setWorkspaceMode('author');
  }, []);

  const handleViewResults = useCallback(() => {
    setSelectedRun(null);
    setWorkspaceMode('analysis');
  }, []);

  const handleAnchorChange = useCallback((value: string) => {
    if (!value.trim()) {
      setAnchor(null);
      return;
    }

    setAnchor({
      output: value,
      label: 'Example anchor',
      source: 'example',
      runIndex: null,
    });
  }, []);

  const handleClearAnchor = useCallback(() => {
    setAnchor(null);
  }, []);

  const handlePromoteRunToAnchor = useCallback((index: number) => {
    const run = results[index];
    if (!run) {
      return;
    }

    setAnchor({
      output: run.output,
      label: `Anchor from run ${index + 1}`,
      source: 'run',
      runIndex: index,
    });
  }, [results]);

  const hasResults = results.some((result) => result !== null) || runErrors.some((error) => Boolean(error));
  const resultCount = Math.max(results.length, runErrors.length);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      if (!hasResults) {
        window.sessionStorage.removeItem(PLAYGROUND_RESULTS_CACHE_KEY);
        return;
      }

      window.sessionStorage.setItem(
        PLAYGROUND_RESULTS_CACHE_KEY,
        JSON.stringify({
          version: 1,
          results,
          runErrors,
          anchor,
        } satisfies CachedPlaygroundResults),
      );
    } catch {
      window.sessionStorage.removeItem(PLAYGROUND_RESULTS_CACHE_KEY);
    }
  }, [anchor, hasResults, results, runErrors]);

  return (
    <PromptForm
      mode={workspaceMode}
      hasResults={hasResults}
      resultCount={resultCount}
      onBackToEdit={handleBackToEdit}
      onViewResults={handleViewResults}
      analysisContent={(
        <div className="playground-analysis">
          <ResultsComparison
            analyzed={analysis.analyzed}
            reference={analysis.reference}
            referenceSchema={analysis.referenceSchema}
            referenceSchemaKind={analysis.referenceSchemaKind}
            scatterPoints={analysis.scatterPoints}
            counts={analysis.counts}
            selectedRun={selectedRun}
            onSelectRun={handleSelectRun}
            onPromoteRun={handlePromoteRunToAnchor}
          />
          {selectedRun !== null && (
            <div className="playground-analysis__detail">
              <span className="section-label">Run Detail</span>
              <div className="playground-analysis__detail-body">
                <RunDetailView
                  analyzed={analysis.analyzed}
                  selectedRun={selectedRun}
                  reference={analysis.reference}
                  onBack={handleBack}
                  onPromoteRun={handlePromoteRunToAnchor}
                />
              </div>
            </div>
          )}
        </div>
      )}
      onRunComplete={handleRunComplete}
      anchorOutput={anchor?.output ?? ''}
      anchorLabel={anchor?.label ?? null}
      onAnchorChange={handleAnchorChange}
      onClearAnchor={handleClearAnchor}
    />
  );
};

export default CreateRun;
