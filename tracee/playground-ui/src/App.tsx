import React, { useState } from 'react';
import CreateRun from './components/CreateRun';
import PromptsList from './components/PromptsList';

type View = 'prompts' | 'create';

const App: React.FC = () => {
  const [view, setView] = useState<View>('create');

  return (
    <div>
      <nav className="app-shell__nav">
        <span className="app-shell__brand">tracee</span>
        <div className="app-shell__links">
          <button
            className={`app-shell__link ${view === 'prompts' ? 'is-active' : ''}`}
            onClick={() => setView('prompts')}
          >
            Prompts
          </button>
          <button
            className={`app-shell__link ${view === 'create' ? 'is-active' : ''}`}
            onClick={() => setView('create')}
          >
            Playground
          </button>
        </div>
      </nav>

      <div className="page-container">
        {view === 'prompts' && <PromptsList />}
        {view === 'create' && <CreateRun />}
      </div>
    </div>
  );
};

export default App;
