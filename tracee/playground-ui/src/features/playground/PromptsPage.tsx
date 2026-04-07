import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { promptAPI } from '../../services/api';
import type { PromptListItem, PromptVersion, PromptWithVersions } from '../../types/prompt';
import {
  getPromptComponentDisplayName,
  normalizePromptComponents,
  resolvePromptMessages,
  serializePromptMessages,
} from './promptEditor';
import PromptVersionTree from './components/PromptVersionTree';
import PromptResolvedView from './components/PromptResolvedView';
import PromptDiffWorkspace from './components/PromptDiffWorkspace';
import PromptDetailPanel from './components/PromptDetailPanel';
import iconCompare from '../../assets/icon-compare.svg';
import iconPlayground from '../../assets/icon-playground.svg';
import iconTrash from '../../assets/icon-trash.svg';

type SortKey = 'name' | 'updated_at' | 'version_count';
type SortDir = 'asc' | 'desc';
type DetailView = 'overview' | 'resolved' | 'diff';

interface CompareTarget {
  promptId: string;
  promptName: string;
  version: PromptVersion;
}

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

export function PromptsPage() {
  const [prompts, setPrompts] = useState<PromptListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PromptWithVersions | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('updated_at');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const [deleteTarget, setDeleteTarget] = useState<PromptListItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [activeVersionId, setActiveVersionId] = useState<string | null>(null);
  const [compareTarget, setCompareTarget] = useState<CompareTarget | null>(null);
  const [detailView, setDetailView] = useState<DetailView>('overview');

  const loadPrompts = useCallback(async () => {
    setLoading(true);
    setError(null);
    const data = await promptAPI.getAllPrompts().catch(() => {
      setError('Failed to load prompts.');
      return [] as PromptListItem[];
    });
    setPrompts(data);
    setLoading(false);
  }, []);

  useEffect(() => { loadPrompts(); }, [loadPrompts]);

  const loadDetail = useCallback(async (promptId: string) => {
    setDetailLoading(true);
    const data = await promptAPI.getPrompt(promptId).catch(() => null);
    setDetail(data);
    setDetailLoading(false);
    setActiveVersionId(null);
    setCompareTarget(null);
    setDetailView('overview');
  }, []);

  const handleSelect = (promptId: string) => {
    if (selectedId === promptId) return;
    setSelectedId(promptId);
    loadDetail(promptId);
  };

  const handleDeselect = () => {
    setSelectedId(null);
    setDetail(null);
    setActiveVersionId(null);
    setCompareTarget(null);
    setDetailView('overview');
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    await promptAPI.deletePrompt(deleteTarget.prompt_id).catch(() => null);
    setDeleting(false);
    setDeleteTarget(null);
    if (selectedId === deleteTarget.prompt_id) {
      handleDeselect();
    }
    loadPrompts();
  };

  const handleMetadataUpdated = () => {
    loadPrompts();
    if (selectedId) loadDetail(selectedId);
  };

  const handleLoadVersion = (version: PromptVersion) => {
    setActiveVersionId(version.version_id);
  };

  const handleToggleCompare = (version: PromptVersion) => {
    if (!detail) return;
    const isSame = compareTarget?.version.version_id === version.version_id;
    if (isSame) {
      setCompareTarget(null);
      if (detailView === 'diff') setDetailView('overview');
    } else {
      setCompareTarget({
        promptId: detail.prompt.prompt_id,
        promptName: detail.prompt.name,
        version,
      });
      setDetailView('diff');
    }
  };

  // derived: latest version
  const latestVersion = useMemo(() => {
    if (!detail || detail.versions.length === 0) return null;
    return detail.versions.reduce((a, b) => a.created_at > b.created_at ? a : b);
  }, [detail]);

  const effectiveVersionId = activeVersionId ?? latestVersion?.version_id ?? null;
  const effectiveVersion = useMemo(() => {
    if (!detail || !effectiveVersionId) return null;
    return detail.versions.find(v => v.version_id === effectiveVersionId) ?? null;
  }, [detail, effectiveVersionId]);

  // resolved text for the active version (used by both resolved view and diff)
  const currentResolvedPrompt = useMemo(() => {
    if (!effectiveVersion) return '';
    return serializePromptMessages(
      resolvePromptMessages(
        normalizePromptComponents(effectiveVersion.components),
        effectiveVersion.variables ?? {},
      )
    );
  }, [effectiveVersion]);

  // resolved text for compare target
  const compareResolvedPrompt = useMemo(() => {
    if (!compareTarget) return '';
    return serializePromptMessages(
      resolvePromptMessages(
        normalizePromptComponents(compareTarget.version.components),
        compareTarget.version.variables ?? {},
      )
    );
  }, [compareTarget]);

  // version tree compare targets
  const versionTreeCompareTargets = useMemo(() => {
    if (!compareTarget || !detail) return [];
    return [{
      promptId: compareTarget.promptId,
      versionId: compareTarget.version.version_id,
    }];
  }, [compareTarget, detail]);

  // filtering and sorting
  const filtered = prompts.filter(p => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      p.name.toLowerCase().includes(q)
      || (p.description ?? '').toLowerCase().includes(q)
      || p.prompt_id.toLowerCase().includes(q)
    );
  });

  const sorted = [...filtered].sort((a, b) => {
    let cmp = 0;
    if (sortKey === 'name') cmp = a.name.localeCompare(b.name);
    else if (sortKey === 'updated_at') cmp = a.updated_at.localeCompare(b.updated_at);
    else if (sortKey === 'version_count') cmp = a.version_count - b.version_count;
    return sortDir === 'desc' ? -cmp : cmp;
  });

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir(key === 'name' ? 'asc' : 'desc'); }
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return null;
    return <span className="prompts-sort-icon">{sortDir === 'asc' ? '↑' : '↓'}</span>;
  };

  const hasDetail = selectedId && detail;

  return (
    <div className="page-container flex-col playground-page prompts-page">
      <div className="prompts-page__header">
        <div className="flex-col">
          <h2>Prompts</h2>
          <p className="field__hint">
            Browse, inspect, and manage your saved prompts.
          </p>
        </div>
      </div>

      {error && (
        <div className="alert alert--danger">
          <span className="alert__icon">!</span>
          {error}
          <button type="button" className="btn btn--sm btn--ghost" onClick={loadPrompts}>
            retry
          </button>
        </div>
      )}

      <div className={`prompts-page__layout${hasDetail ? ' prompts-page__layout--with-detail' : ''}`}>
        {/* left: prompt list (always narrow) */}
        <div className="prompts-page__list-panel">
          <div className="prompts-toolbar">
            <div className="prompts-toolbar__search">
              <input
                type="text"
                className="input prompts-toolbar__input"
                placeholder="Search prompts..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                aria-label="search prompts"
              />
              {search && (
                <button
                  type="button"
                  className="prompts-toolbar__clear"
                  onClick={() => setSearch('')}
                  aria-label="clear search"
                >
                  ×
                </button>
              )}
            </div>
            <span className="prompts-toolbar__count">
              {filtered.length}
            </span>
          </div>

          {loading ? (
            <div className="prompts-page__loading">
              <div className="spinner spinner--lg" />
              <span>Loading...</span>
            </div>
          ) : prompts.length === 0 ? (
            <div className="prompts-page__empty prompts-page__empty--compact">
              <div className="prompts-page__empty-title">No prompts yet</div>
              <div className="prompts-page__empty-desc">
                Create a run in the Playground to generate your first prompt.
              </div>
              <a href="/playground" className="btn btn--primary btn--sm prompts-page__link-btn">
                Go to Playground
              </a>
            </div>
          ) : filtered.length === 0 ? (
            <div className="prompts-page__empty prompts-page__empty--compact">
              <div className="prompts-page__empty-title">No matches</div>
              <div className="prompts-page__empty-desc">
                No prompts match "{search}".
              </div>
            </div>
          ) : (
            <div className="prompts-list">
              <div className="prompts-list__header">
                <button type="button" className="prompts-list__sort-btn" onClick={() => toggleSort('name')}>
                  Name {sortIndicator('name')}
                </button>
                <button type="button" className="prompts-list__sort-btn prompts-list__sort-btn--right" onClick={() => toggleSort('version_count')}>
                  V {sortIndicator('version_count')}
                </button>
                <button type="button" className="prompts-list__sort-btn prompts-list__sort-btn--right" onClick={() => toggleSort('updated_at')}>
                  Updated {sortIndicator('updated_at')}
                </button>
              </div>

              {sorted.map(prompt => (
                <div
                  key={prompt.prompt_id}
                  className={`prompts-list__item${selectedId === prompt.prompt_id ? ' is-selected' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleSelect(prompt.prompt_id)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSelect(prompt.prompt_id);
                    }
                  }}
                >
                  <div className="prompts-list__item-main">
                    <div className="prompts-list__item-name">{prompt.name}</div>
                    {prompt.description && (
                      <div className="prompts-list__item-desc">{prompt.description}</div>
                    )}
                  </div>
                  <div className="prompts-list__item-meta">
                    <span className="badge badge--neutral">{prompt.version_count}</span>
                    <span className="prompts-list__item-date">{fmtDate(prompt.updated_at)}</span>
                  </div>
                  <button
                    type="button"
                    className="prompts-list__delete-btn"
                    title="delete prompt"
                    aria-label={`delete ${prompt.name}`}
                    onClick={e => { e.stopPropagation(); setDeleteTarget(prompt); }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* middle: version tree (when selected) */}
        {hasDetail ? (
          <div className="prompts-page__tree-panel">
            <div className="card">
              <div className="card__header">
                <div className="card__title">Version history</div>
                <span className="badge badge--neutral">{detail.versions.length}</span>
              </div>
              <div className="card__body">
                {detailLoading ? (
                  <div className="prompts-page__loading">
                    <div className="spinner" />
                    <span>Loading...</span>
                  </div>
                ) : (
                  <PromptVersionTree
                    promptId={detail.prompt.prompt_id}
                    versions={detail.versions}
                    activeVersionId={effectiveVersionId}
                    compareTargets={versionTreeCompareTargets}
                    onLoadVersion={handleLoadVersion}
                    onToggleCompare={handleToggleCompare}
                  />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="prompts-page__placeholder">
            <div className="prompts-page__placeholder-content">
              <div className="prompts-page__placeholder-icon">←</div>
              <div className="prompts-page__placeholder-title">Select a prompt</div>
              <div className="prompts-page__placeholder-desc">
                Choose a prompt from the list to view its version history, preview the resolved text, and manage metadata.
              </div>
            </div>
          </div>
        )}

        {/* right: detail panel (when selected) */}
        {hasDetail && (
          <div className="prompts-page__detail-panel">
            {detailLoading ? (
              <div className="prompts-page__loading">
                <div className="spinner" />
                <span>Loading...</span>
              </div>
            ) : (
              <>
                <PromptDetailPanel
                  data={detail}
                  onClose={handleDeselect}
                  onMetadataUpdated={handleMetadataUpdated}
                />

                {/* workspace view nav — matches PromptForm's seg-control */}
                <div className="prompts-page__workspace">
                  <div className="prompts-page__workspace-header">
                    <div className="seg-control playground-workspace-nav" role="group" aria-label="Detail view">
                      <button
                        type="button"
                        className={`seg-control__btn${detailView === 'overview' ? ' is-active' : ''}`}
                        onClick={() => setDetailView('overview')}
                        aria-pressed={detailView === 'overview'}
                      >
                        Components
                      </button>
                      <button
                        type="button"
                        className={`seg-control__btn${detailView === 'resolved' ? ' is-active' : ''}`}
                        onClick={() => setDetailView('resolved')}
                        aria-pressed={detailView === 'resolved'}
                      >
                        Resolved
                      </button>
                      {compareTarget && (
                        <button
                          type="button"
                          className={`seg-control__btn${detailView === 'diff' ? ' is-active' : ''}`}
                          onClick={() => setDetailView('diff')}
                          aria-pressed={detailView === 'diff'}
                        >
                          Diff
                        </button>
                      )}
                    </div>
                    <div className="prompts-page__workspace-actions">
                      {compareTarget && (
                        <button
                          type="button"
                          className="btn btn--secondary btn--sm prompts-page__action-btn"
                          onClick={() => {
                            setCompareTarget(null);
                            if (detailView === 'diff') setDetailView('overview');
                          }}
                        >
                          <span
                            className="prompts-page__action-icon"
                            style={getMaskIconStyle(iconCompare)}
                            aria-hidden
                          />
                          <span>Clear compare</span>
                        </button>
                      )}
                      <a
                        href="/playground"
                        className="btn btn--secondary btn--sm prompts-page__action-btn prompts-page__link-btn"
                      >
                        <span
                          className="prompts-page__action-icon"
                          style={getMaskIconStyle(iconPlayground)}
                          aria-hidden
                        />
                        <span>Load in Playground</span>
                      </a>
                      <button
                        type="button"
                        className="btn btn--secondary btn--sm prompts-page__action-btn prompts-page__action-btn--danger"
                        onClick={() => {
                          const item = prompts.find(p => p.prompt_id === detail.prompt.prompt_id);
                          if (item) setDeleteTarget(item);
                        }}
                      >
                        <span
                          className="prompts-page__action-icon"
                          style={getMaskIconStyle(iconTrash)}
                          aria-hidden
                        />
                        <span>Delete</span>
                      </button>
                    </div>
                  </div>

                  <div className="prompts-page__workspace-body">
                    {detailView === 'overview' && effectiveVersion && (
                      <ComponentsView version={effectiveVersion} />
                    )}
                    {detailView === 'resolved' && (
                      <PromptResolvedView resolvedPrompt={currentResolvedPrompt} />
                    )}
                    {detailView === 'diff' && compareTarget && effectiveVersion && (
                      <PromptDiffWorkspace
                        currentDraft={{
                          label: effectiveVersion.name || detail.prompt.name,
                          promptId: detail.prompt.prompt_id,
                          versionId: effectiveVersion.version_id,
                          revisionNote: effectiveVersion.revision_note,
                          components: effectiveVersion.components,
                          variables: effectiveVersion.variables,
                          tools: effectiveVersion.tools,
                          outputSchema: effectiveVersion.output_schema,
                          resolvedPrompt: currentResolvedPrompt,
                        }}
                        target={compareTarget}
                        targetResolvedPrompt={compareResolvedPrompt}
                      />
                    )}
                    {detailView === 'diff' && !compareTarget && (
                      <div className="prompts-page__hint-note">
                        choose a saved version from the version tree to compare it with the active version.
                      </div>
                    )}
                    {!effectiveVersion && detailView !== 'diff' && (
                      <div className="prompts-page__hint-note">
                        no versions available.
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* delete confirmation dialog */}
      {deleteTarget && (
        <div className="prompts-dialog__backdrop" onClick={() => !deleting && setDeleteTarget(null)}>
          <div className="prompts-dialog" onClick={e => e.stopPropagation()}>
            <div className="prompts-dialog__title">Delete prompt</div>
            <div className="prompts-dialog__body">
              Are you sure you want to delete <strong>{deleteTarget.name}</strong> and
              all {deleteTarget.version_count} version{deleteTarget.version_count !== 1 ? 's' : ''}?
              This cannot be undone.
            </div>
            <div className="prompts-dialog__actions">
              <button
                type="button"
                className="btn btn--secondary btn--sm"
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn--danger btn--sm"
                onClick={handleDelete}
                disabled={deleting}
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// simple read-only component viewer (matches the pattern used in PromptForm's component editor view)
const ComponentsView: React.FC<{ version: PromptVersion }> = ({ version }) => {
  const components = version.components;
  const tools = version.tools ?? [];
  const variables = version.variables ?? {};
  const variableKeys = Object.keys(variables);

  return (
    <div className="prompt-detail__overview">
      {components.map((comp, i) => (
        <div
          key={comp.component_id ?? i}
          className={`card prompt-components__card${!comp.enabled ? ' prompt-components__card--disabled' : ''}`}
        >
          <div className="prompt-components__header">
            <div className="prompt-components__header-left">
              <span className="prompt-detail__component-name">{getPromptComponentDisplayName(comp)}</span>
              {comp.message_role && (
                <span className="badge badge--neutral">{comp.message_role}</span>
              )}
              {!comp.enabled && (
                <span className="badge badge--warning">disabled</span>
              )}
            </div>
          </div>
          <div className="prompt-components__body">
            <textarea
              className="textarea textarea--code"
              value={comp.content}
              readOnly
              rows={comp.content.split('\n').length}
              style={{ resize: 'none' }}
            />
          </div>
        </div>
      ))}

      {tools.length > 0 && (
        <div className="prompt-detail__section">
          <div className="section-label">Tools ({tools.length})</div>
          <div className="prompt-detail__chip-row">
            {tools.map((tool, i) => (
              <span key={tool.tool_id ?? i} className="version-tree__component-chip">{tool.name}</span>
            ))}
          </div>
        </div>
      )}

      {variableKeys.length > 0 && (
        <div className="prompt-detail__section">
          <div className="section-label">Variables ({variableKeys.length})</div>
          <div className="prompt-detail__chip-row">
            {variableKeys.map(key => (
              <code key={key} className="prompt-detail__variable-chip">{`{{${key}}}`}</code>
            ))}
          </div>
        </div>
      )}

      {version.output_schema && (
        <div className="prompt-detail__section">
          <div className="section-label">Output schema</div>
          <pre className="prompt-detail__schema-pre">
            {JSON.stringify(version.output_schema, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
