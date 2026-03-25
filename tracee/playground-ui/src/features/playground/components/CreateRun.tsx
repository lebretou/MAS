import React, { useState, useCallback, useEffect } from 'react';
import type { PlaygroundAnalysisGroup, PlaygroundRun } from '../../../types/playground';
import { useRunAnalysis } from '../../../hooks/useRunAnalysis';
import type { AnchorPoint } from '../../../hooks/useRunAnalysis';
import PromptForm from './PromptForm';
import ResultsComparison from './ResultsComparison';
import RunDetailView from './RunDetailView';

const PLAYGROUND_RESULTS_CACHE_KEY = 'tracee:playground:last-results';

interface CachedPlaygroundResults {
  version: 2;
  groups: PlaygroundAnalysisGroup[];
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

    const parsed = JSON.parse(rawValue) as CachedPlaygroundResults & {
      results?: Array<PlaygroundRun | null>;
      runErrors?: Array<string | null>;
    };

    if (parsed?.version === 2 && Array.isArray(parsed.groups)) {
      return {
        version: 2,
        groups: parsed.groups,
        anchor: parsed.anchor ?? null,
      };
    }

    if (Array.isArray(parsed.results) && Array.isArray(parsed.runErrors)) {
      return {
        version: 2,
        groups: [{
          id: 'primary',
          label: 'Current prompt',
          tone: 'primary',
          promptId: null,
          versionId: null,
          results: parsed.results,
          runErrors: parsed.runErrors,
        }],
        anchor: parsed.anchor ?? null,
      };
    }

    return null;
  } catch {
    return null;
  }
}

const CreateRun: React.FC = () => {
  const [cachedResults] = useState<CachedPlaygroundResults | null>(() => readCachedPlaygroundResults());
  const [workspaceMode, setWorkspaceMode] = useState<'author' | 'analysis'>('author');
  const [analysisGroups, setAnalysisGroups] = useState<PlaygroundAnalysisGroup[]>(cachedResults?.groups ?? []);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [anchor, setAnchor] = useState<AnchorPoint | null>(cachedResults?.anchor ?? null);

  const analysis = useRunAnalysis(analysisGroups, anchor);

  const handleRunComplete = useCallback((
    groups: PlaygroundAnalysisGroup[],
  ) => {
    setAnalysisGroups(groups);
    setSelectedRun((currentSelectedRun) => {
      if (!currentSelectedRun) {
        return null;
      }

      const nextSelectionIds = new Set(groups.flatMap((group) => (
        group.results.map((_, index) => `${group.id}:${index}`)
      )));
      return nextSelectionIds.has(currentSelectedRun) ? currentSelectedRun : null;
    });
    setWorkspaceMode('analysis');
    setAnchor((currentAnchor) => {
      if (groups.length !== 1 || currentAnchor?.source !== 'run') {
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

  const handleSelectRun = useCallback((index: string | null) => {
    setSelectedRun(index);
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
    const primaryGroup = analysisGroups.find((group) => group.tone === 'primary') ?? analysisGroups[0];
    const run = primaryGroup?.results[index];
    if (!run) {
      return;
    }

    setAnchor({
      output: run.output,
      label: `Anchor from run ${index + 1}`,
      source: 'run',
      runIndex: index,
    });
  }, [analysisGroups]);

  const hasResults = analysisGroups.some((group) => (
    group.results.some((result) => result !== null) || group.runErrors.some((error) => Boolean(error))
  ));
  const resultCount = analysisGroups.reduce(
    (count, group) => count + Math.max(group.results.length, group.runErrors.length),
    0,
  );

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
          version: 2,
          groups: analysisGroups,
          anchor,
        } satisfies CachedPlaygroundResults),
      );
    } catch {
      window.sessionStorage.removeItem(PLAYGROUND_RESULTS_CACHE_KEY);
    }
  }, [analysisGroups, anchor, hasResults]);

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
            projectionItems={analysis.projectionItems}
            fieldOptions={analysis.fieldOptions}
            failureCount={analysis.failureCount}
            selectedRun={selectedRun}
            onSelectRun={handleSelectRun}
            detailContent={(
              <RunDetailView
                analyzed={analysis.analyzed}
                selectedRun={selectedRun}
                reference={analysis.reference}
                onPromoteRun={handlePromoteRunToAnchor}
              />
            )}
          />
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
