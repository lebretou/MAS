import React from 'react';
import type { PromptComponent, PromptTool, PromptVersion } from '../../../types/prompt';
import { computeDiff } from '../../../utils/jsonDiff';
import iconOutputSchema from '../../../assets/icon-outputschema.svg';
import iconTool from '../../../assets/icon-tool.svg';
import iconVariable from '../../../assets/icon-variable.svg';
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
  variables?: Record<string, string> | null;
  tools?: PromptTool[] | null;
  outputSchema?: Record<string, unknown> | null;
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

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

function renderMetaChangeTag(
  key: string,
  label: string,
  icon: string,
) {
  return (
    <span key={key} className="version-tree__edge-tag version-tree__edge-tag--changed">
      <span
        className="version-tree__edge-tag-icon"
        style={getMaskIconStyle(icon)}
        aria-hidden
      />
      <span>~ {label}</span>
    </span>
  );
}

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
          tools: currentDraft.tools ?? [],
          outputSchema: currentDraft.outputSchema ?? null,
          variables: currentDraft.variables ?? null,
        },
        {
          components: normalizePromptComponents(target.version.components),
          tools: target.version.tools ?? [],
          outputSchema: target.version.output_schema ?? null,
          variables: target.version.variables ?? null,
        }
      );

      return [
        ...diffSummary.added.map((label) => ({ kind: 'added' as const, label })),
        ...diffSummary.removed.map((label) => ({ kind: 'removed' as const, label })),
        ...diffSummary.changed.map((label) => ({ kind: 'changed' as const, label })),
        ...(diffSummary.toolChanged ? [{ kind: 'tool' as const, label: 'tools' }] : []),
        ...(diffSummary.schemaChanged ? [{ kind: 'schema' as const, label: 'schema' }] : []),
        ...(diffSummary.variableChanged ? [{ kind: 'variable' as const, label: 'variables' }] : []),
      ];
    },
    [
      currentDraft.components,
      currentDraft.outputSchema,
      currentDraft.tools,
      currentDraft.variables,
      target.version.components,
      target.version.output_schema,
      target.version.tools,
      target.version.variables,
    ]
  );
  const resolvedDiff = React.useMemo(
    () => computeDiff(targetResolvedPrompt, currentDraft.resolvedPrompt),
    [currentDraft.resolvedPrompt, targetResolvedPrompt]
  );
  const hasResolvedChanges = resolvedDiff.some((line) => line.type !== 'same');
  const compareVersionLabel = `${target.promptId} / ${target.version.version_id}`;
  const compareVersionName = target.version.name?.trim() || target.promptName;

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
        <div className="card version-compare__meta-card-shell version-compare__meta-card-shell--compare">
          <div className="card__body version-compare__meta-card version-compare__meta-card--compare">
            <div className="version-compare__meta-head">
              <span className="version-compare__meta-chip version-compare__meta-chip--compare">comparing</span>
            </div>
            <div className="version-compare__version-name">{compareVersionLabel}</div>
            {compareVersionName ? (
              <div className="field__hint">{compareVersionName}</div>
            ) : null}
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
                item.kind === 'tool'
                  ? renderMetaChangeTag(`${item.kind}-${item.label}-${index}`, item.label, iconTool)
                  : item.kind === 'schema'
                    ? renderMetaChangeTag(`${item.kind}-${item.label}-${index}`, item.label, iconOutputSchema)
                    : item.kind === 'variable'
                      ? renderMetaChangeTag(`${item.kind}-${item.label}-${index}`, item.label, iconVariable)
                      : (
                        <span
                          key={`${item.kind}-${item.label}-${index}`}
                          className={`version-tree__edge-tag version-tree__edge-tag--${item.kind === 'added' ? 'added' : item.kind === 'removed' ? 'removed' : 'changed'}`}
                        >
                          {item.kind === 'added' ? '+ ' : item.kind === 'removed' ? '- ' : '~ '}
                          {item.label}
                        </span>
                      )
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
