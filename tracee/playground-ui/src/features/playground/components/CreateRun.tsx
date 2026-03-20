import React, { useState, useCallback } from 'react';
import type { PlaygroundRun } from '../../../types/playground';
import { useRunAnalysis } from '../../../hooks/useRunAnalysis';
import type { AnchorPoint } from '../../../hooks/useRunAnalysis';
import PromptForm from './PromptForm';
import ResultsComparison from './ResultsComparison';
import RunDetailView from './RunDetailView';

const CreateRun: React.FC = () => {
  const [workspaceMode, setWorkspaceMode] = useState<'author' | 'analysis'>('author');
  const [results, setResults] = useState<Array<PlaygroundRun | null>>([]);
  const [runErrors, setRunErrors] = useState<Array<string | null>>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [anchor, setAnchor] = useState<AnchorPoint | null>(null);

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

  return (
    <PromptForm
      mode={workspaceMode}
      hasResults={hasResults}
      onBackToEdit={handleBackToEdit}
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
              <div style={{ marginTop: 12 }}>
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
