import React from 'react';
import type { PromptComponent, PromptVersion } from '../../../types/prompt';
import { computeDiff } from '../../../utils/jsonDiff';
import { normalizePromptComponents } from '../promptEditor';
import { getPromptVersionDiffSummary } from '../promptVersionDiff';

interface DiffTarget {
  promptId: string;
  promptName: string;
  version: PromptVersion;
}

interface CurrentDraft {
  label: string;
  promptId?: string | null;
  versionId: string | null;
  revisionNote?: string | null;
  components: PromptComponent[];
  resolvedPrompt: string;
}

interface Props {
  currentDraft: CurrentDraft;
  target: DiffTarget;
  targetResolvedPrompt: string;
}

const PREFIX = {
  same: '  ',
  added: '+ ',
  removed: '- ',
} as const;

const PromptDiffWorkspace: React.FC<Props> = ({
  currentDraft,
  target,
  targetResolvedPrompt,
}) => {
  const componentDelta = React.useMemo(
    () => {
      const diffSummary = getPromptVersionDiffSummary(
        {
          components: normalizePromptComponents(currentDraft.components),
          tools: [],
          outputSchema: null,
          variables: null,
        },
        {
          components: normalizePromptComponents(target.version.components),
          tools: [],
          outputSchema: null,
          variables: null,
        }
      );

      return [
        ...diffSummary.added.map((label) => ({ kind: 'added' as const, label })),
        ...diffSummary.removed.map((label) => ({ kind: 'removed' as const, label })),
        ...diffSummary.changed.map((label) => ({ kind: 'changed' as const, label })),
      ];
    },
    [currentDraft.components, target.version.components]
  );
  const resolvedDiff = React.useMemo(
    () => computeDiff(targetResolvedPrompt, currentDraft.resolvedPrompt),
    [currentDraft.resolvedPrompt, targetResolvedPrompt]
  );
  const hasResolvedChanges = resolvedDiff.some((line) => line.type !== 'same');

  return (
    <div className="prompt-diff">
      <div className="version-compare__meta-grid">
        <div className="card">
          <div className="card__body version-compare__meta-card">
            <div className="version-compare__meta-head">
              <span className="version-compare__meta-chip version-compare__meta-chip--current">current</span>
            </div>
            <div className="version-compare__version-name">{currentDraft.label}</div>
            <div className="field__hint">
              {currentDraft.promptId && currentDraft.versionId
                ? `${currentDraft.promptId} / ${currentDraft.versionId}`
                : currentDraft.versionId ?? 'unsaved draft'}
            </div>
            {currentDraft.revisionNote ? (
              <div className="version-compare__revision-note">{currentDraft.revisionNote}</div>
            ) : null}
            <div className="version-compare__stat-row">
              <span>{currentDraft.components.length} sections</span>
              <span>{currentDraft.resolvedPrompt.length} chars</span>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card__body version-compare__meta-card">
            <div className="version-compare__meta-head">
              <span className="version-compare__meta-chip version-compare__meta-chip--saved">saved version</span>
              {target.version.branch_name ? (
                <span className="badge badge--neutral">{target.version.branch_name}</span>
              ) : null}
            </div>
            <div className="version-compare__version-name">{target.version.name || target.promptName}</div>
            <div className="field__hint">{target.promptId} / {target.version.version_id}</div>
            {target.version.revision_note ? (
              <div className="version-compare__revision-note">{target.version.revision_note}</div>
            ) : null}
            <div className="version-compare__stat-row">
              <span>{target.version.components.length} sections</span>
              <span>{targetResolvedPrompt.length} chars</span>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card__body version-compare__delta-card">
          <div className="section-label">Component delta</div>
          {componentDelta.length > 0 ? (
            <div className="version-compare__delta-chips">
              {componentDelta.map((item, index) => (
                <span
                  key={`${item.kind}-${item.label}-${index}`}
                  className={`version-tree__edge-tag version-tree__edge-tag--${item.kind === 'added' ? 'added' : item.kind === 'removed' ? 'removed' : 'changed'}`}
                >
                  {item.kind === 'added' ? '+ ' : item.kind === 'removed' ? '- ' : '~ '}
                  {item.label}
                </span>
              ))}
            </div>
          ) : (
            <div className="field__hint">no structural changes between these prompt states.</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card__header diff__header">
          <div>
            <h3 className="card__title diff__title">Resolved prompt diff</h3>
            <span className="field__hint">
              current editor state vs {target.version.version_id}
            </span>
          </div>
        </div>
        <div className="card__body diff__body">
          {!hasResolvedChanges ? (
            <div className="diff__identical">No changes detected.</div>
          ) : (
            <div className="diff__legend">
              <span><span className="diff__legend-removed">-</span> {target.version.version_id}</span>
              <span><span className="diff__legend-added">+</span> current editor</span>
            </div>
          )}
          <pre className="diff__pre">
            {resolvedDiff.map((line, index) => (
              <div key={`${line.type}-${index}`} className={`diff__line diff__line--${line.type}`}>
                <span className="diff__prefix">{PREFIX[line.type]}</span>
                {line.text}
              </div>
            ))}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default PromptDiffWorkspace;
