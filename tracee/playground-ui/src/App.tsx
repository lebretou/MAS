import React, { useState } from 'react';
import CreateRun from './components/CreateRun';
import PromptsList from './components/PromptsList';

type View = 'list' | 'create' | 'prompts';

const App: React.FC = () => {
  const [view, setView] = useState<View>('list');

  return (
    <div>
      <nav style={{ backgroundColor: '#333', padding: '10px' }}>
        <button
          onClick={() => setView('prompts')}
          style={{
            padding: '8px 16px',
            backgroundColor: view === 'prompts' ? '#555' : '#222',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            marginRight: '8px',
          }}
        >
          Prompts
        </button>
        <button
          onClick={() => setView('create')}
          style={{
            padding: '8px 16px',
            backgroundColor: view === 'create' ? '#555' : '#222',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Create Run
        </button>
      </nav>

      {view === 'prompts' && <PromptsList />}
      {view === 'create' && <CreateRun />}
    </div>
  );
};

export default App;