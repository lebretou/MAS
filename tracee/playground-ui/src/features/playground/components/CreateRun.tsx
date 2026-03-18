import React, { useState, useCallback } from 'react';
import type { PlaygroundRun } from '../../../types/playground';
import { useRunAnalysis } from '../../../hooks/useRunAnalysis';
import type { AnchorPoint } from '../../../hooks/useRunAnalysis';
import PromptForm from './PromptForm';
import ResultsComparison from './ResultsComparison';
import RunDetailView from './RunDetailView';

const CreateRun: React.FC = () => {
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

  const showDetails = selectedRun !== null;

  return (
    <div className={`split-layout ${showDetails ? 'split-layout--with-detail' : 'split-layout--workspace'}`}>
      <div className="split-layout__panel">
        <span className="section-label">Prompt Configuration</span>
        <div style={{ marginTop: 12 }}>
          <PromptForm
            onRunComplete={handleRunComplete}
            anchorOutput={anchor?.output ?? ''}
            anchorLabel={anchor?.label ?? null}
            onAnchorChange={handleAnchorChange}
            onClearAnchor={handleClearAnchor}
          />
        </div>
      </div>

      <div className="split-layout__panel">
        <span className="section-label">Visualization + Results</span>
        <div style={{ marginTop: 12 }}>
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
        </div>
      </div>

      {showDetails && (
        <div className="split-layout__panel split-layout__panel--detail">
          <RunDetailView
            analyzed={analysis.analyzed}
            selectedRun={selectedRun}
            reference={analysis.reference}
            onBack={handleBack}
            onPromoteRun={handlePromoteRunToAnchor}
          />
        </div>
      )}
    </div>
  );
};

export default CreateRun;
