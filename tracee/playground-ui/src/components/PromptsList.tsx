import React, { useEffect, useState } from 'react';
import { promptAPI } from '../services/api';
import { PromptListItem, PromptWithVersions } from '../types/prompt';

type ExpandedState = PromptWithVersions | 'loading' | 'error';

const fmtDate = (iso: string) => iso.slice(0, 10);

const PromptsList: React.FC = () => {
  const [prompts, setPrompts] = useState<PromptListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, ExpandedState>>({});

  useEffect(() => {
    promptAPI.getAllPrompts()
      .then(setPrompts)
      .catch(() => setError('Failed to load prompts.'))
      .finally(() => setLoading(false));
  }, []);

  const toggleRow = async (promptId: string) => {
    if (expanded[promptId] !== undefined) {
      // Collapse
      setExpanded(prev => {
        const next = { ...prev };
        delete next[promptId];
        return next;
      });
      return;
    }

    // Expand — fetch if not cached
    setExpanded(prev => ({ ...prev, [promptId]: 'loading' }));
    try {
      const data = await promptAPI.getPrompt(promptId);
      setExpanded(prev => ({ ...prev, [promptId]: data }));
    } catch {
      setExpanded(prev => ({ ...prev, [promptId]: 'error' }));
    }
  };

  if (loading) return <p>Loading…</p>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;
  if (prompts.length === 0) return <p>No prompts found. Create a run to generate one.</p>;

  return (
    <div style={{ padding: '16px' }}>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Versions</th>
            <th>Latest</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {prompts.map(prompt => {
            const isExpanded = expanded[prompt.prompt_id] !== undefined;
            const expandedData = expanded[prompt.prompt_id];

            return (
              <React.Fragment key={prompt.prompt_id}>
                <tr onClick={() => toggleRow(prompt.prompt_id)} style={{ cursor: 'pointer' }}>
                  <td>{isExpanded ? '▼' : '▶'} {prompt.name}</td>
                  <td>{prompt.version_count}</td>
                  <td>{prompt.latest_version_id}</td>
                  <td>{fmtDate(prompt.created_at)}</td>
                </tr>

                {isExpanded && (
                  <tr>
                    <td colSpan={4} style={{ paddingLeft: '24px' }}>
                      {expandedData === 'loading' && <span>Loading versions…</span>}
                      {expandedData === 'error' && (
                        <span style={{ color: 'red' }}>Failed to load versions.</span>
                      )}
                      {expandedData !== 'loading' && expandedData !== 'error' && expandedData && (
                        <table>
                          <thead>
                            <tr>
                              <th>Version ID</th>
                            </tr>
                          </thead>
                          <tbody>
                            {expandedData.versions.map(version => (
                              <tr key={version.version_id}>
                                <td>{version.version_id}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default PromptsList;
