import React, { useState, useCallback } from 'react';
import { PlaygroundRun } from '../types/playground';
import { useRunAnalysis } from '../hooks/useRunAnalysis';
import PromptForm from './PromptForm';
import ResultsComparison from './ResultsComparison';
import RunDetailView from './RunDetailView';

type LayoutMode = 'prompt-results' | 'results-detail';

const CreateRun: React.FC = () => {
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('prompt-results');
  const [results, setResults] = useState<Array<PlaygroundRun | null>>([]);
  const [runErrors, setRunErrors] = useState<Array<string | null>>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);

  const analysis = useRunAnalysis(results, runErrors);

  const handleRunComplete = useCallback((
    newResults: Array<PlaygroundRun | null>,
    newErrors: Array<string | null>,
  ) => {
    setResults(newResults);
    setRunErrors(newErrors);
    setSelectedRun(null);
    setLayoutMode('prompt-results');
  }, []);

  const handleSelectRun = useCallback((index: number) => {
    setSelectedRun(index);
    setLayoutMode('results-detail');
  }, []);

  const handleBack = useCallback(() => {
    setLayoutMode('prompt-results');
  }, []);

  const promptVisible = layoutMode === 'prompt-results';
  const showDetails = layoutMode === 'results-detail';

  return (
    <div className="split-layout split-layout--split">
      {/* Left panel: PromptForm (always mounted, hidden via display:none in detail mode) */}
      <div
        className="split-layout__panel"
        style={promptVisible ? undefined : { display: 'none' }}
      >
        <span className="section-label">Prompt Configuration</span>
        <div style={{ marginTop: 12 }}>
          <PromptForm onRunComplete={handleRunComplete} />
        </div>
      </div>

      {/* Visualizations panel — always visible */}
      <div className="split-layout__panel">
        <span className="section-label">Visualization + Results</span>
        <div style={{ marginTop: 12 }}>
          <ResultsComparison
            analyzed={analysis.analyzed}
            consensus={analysis.consensus}
            scatterPoints={analysis.scatterPoints}
            counts={analysis.counts}
            selectedRun={selectedRun}
            onSelectRun={handleSelectRun}
          />
        </div>
      </div>

      {/* Detail panel — replaces prompt panel on the right */}
      {showDetails && (
        <div className="split-layout__panel">
          <RunDetailView
            analyzed={analysis.analyzed}
            selectedRun={selectedRun}
            consensusOutputIndex={analysis.consensusOutputIndex}
            onSelectRun={setSelectedRun}
            onBack={handleBack}
          />
        </div>
      )}
    </div>
  );
};

export default CreateRun;
