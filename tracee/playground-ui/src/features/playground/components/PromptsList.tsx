import React, { useEffect, useState } from 'react';
import { promptAPI } from '../../../services/api';
import type { PromptListItem, PromptWithVersions } from '../../../types/prompt';
import PromptVersionTree from './PromptVersionTree';

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
      setExpanded(prev => {
        const next = { ...prev };
        delete next[promptId];
        return next;
      });
      return;
    }

    setExpanded(prev => ({ ...prev, [promptId]: 'loading' }));
    try {
      const data = await promptAPI.getPrompt(promptId);
      setExpanded(prev => ({ ...prev, [promptId]: data }));
    } catch {
      setExpanded(prev => ({ ...prev, [promptId]: 'error' }));
    }
  };

  if (loading) {
    return (
      <div className="empty-state">
        <div className="spinner spinner--lg" />
        <div className="empty-state__title">Loading prompts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert--danger">
        <span className="alert__icon">!</span>
        {error}
      </div>
    );
  }

  if (prompts.length === 0) {
    return (
      <div className="card">
        <div className="empty-state">
          <div className="empty-state__icon">&#9744;</div>
          <div className="empty-state__title">No prompts yet</div>
          <div className="empty-state__desc">Create a run to generate your first prompt.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-col prompts__container">
      <div className="card">
        <div className="card__body prompts__card-body">
          <table className="table">
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
                    <tr
                      role="button"
                      tabIndex={0}
                      aria-expanded={isExpanded}
                      onClick={() => toggleRow(prompt.prompt_id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          toggleRow(prompt.prompt_id);
                        }
                      }}
                      className="prompts__row"
                    >
                      <td>
                        <span className="prompts__expand-icon">
                          {isExpanded ? '▼' : '▶'}
                        </span>
                        {prompt.name}
                      </td>
                      <td>
                        <span className="badge badge--primary">{prompt.version_count}</span>
                      </td>
                      <td>
                        <span className="table__mono">{prompt.latest_version_id}</span>
                      </td>
                      <td>{fmtDate(prompt.created_at)}</td>
                    </tr>

                    {isExpanded && (
                      <tr>
                        <td colSpan={4} className="prompts__expanded-cell">
                          {expandedData === 'loading' && (
                            <div className="flex-row prompts__loading-row">
                              <span className="spinner" />
                              <span className="prompts__loading-text">Loading versions...</span>
                            </div>
                          )}
                          {expandedData === 'error' && (
                            <span className="prompts__error-text">Failed to load versions.</span>
                          )}
                          {expandedData !== 'loading' && expandedData !== 'error' && expandedData && (
                            <PromptVersionTree
                              promptId={expandedData.prompt.prompt_id}
                              versions={expandedData.versions}
                            />
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
      </div>
    </div>
  );
};

export default PromptsList;
