import React, { useState } from 'react';
import CreateRun from './components/CreateRun';

type View = 'list' | 'create';

const App: React.FC = () => {
  const [view, setView] = useState<View>('list');

  return (
    <div>
      <nav style={{ backgroundColor: '#333', padding: '10px' }}>
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

      {view === 'create' && <CreateRun />}
    </div>
  );
};

export default App;