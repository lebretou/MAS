import React from 'react';
import type { PromptComponent, PromptVersion } from '../../../types/prompt';
import iconLoad from '../../../assets/icon-load.svg';
import iconCompare from '../../../assets/icon-compare.svg';
import { componentColors } from '../../graph-viewer/constants';

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

interface CompareTarget {
  promptId: string;
  versionId: string;
}

interface Props {
  promptId: string;
  versions: PromptVersion[];
  activeVersionId?: string | null;
  compareTargets?: CompareTarget[];
  draftLeaf?: {
    promptId: string;
    parentVersionId: string | null;
    versionId: string;
    name: string;
    revisionNote?: string;
    components: PromptComponent[];
  } | null;
  onLoadVersion?: (version: PromptVersion) => void;
  onToggleCompare?: (version: PromptVersion) => void;
}

interface ComponentDiffSummary {
  added: string[];
  removed: string[];
  changed: string[];
}

interface RenderEntry {
  kind: 'saved' | 'draft';
  versionId: string;
  promptId: string;
  parentVersionId: string | null;
  createdAt: string;
  components: PromptComponent[];
  revisionNote?: string;
  version?: PromptVersion;
}

interface GraphRow {
  entry: RenderEntry;
  lane: number;
  parentLane: number | null;
  incoming: boolean;
  before: Array<string | null>;
  after: Array<string | null>;
}

const LANE_COLORS = [
  componentColors.task,
  componentColors.constraints,
  componentColors.outputs,
  componentColors.role,
  componentColors.io_rules,
  componentColors.inputs,
  componentColors.examples,
  componentColors.safety,
  componentColors.tool_instructions,
  componentColors.external_information,
  componentColors.goal,
];

function getComponentLabels(components: PromptComponent[]) {
  return components
    .filter((component) => component.enabled)
    .map((component, index) => ({
      key: component.component_id ?? `${component.type}-${index}`,
      label: component.type.replace(/_/g, ' '),
    }));
}

function getComponentDiffSummary(version: PromptVersion, parent: PromptVersion | null): ComponentDiffSummary {
  if (!parent) {
    return { added: [], removed: [], changed: [] };
  }

  const currentMap = new Map(
    version.components.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ])
  );
  const parentMap = new Map(
    parent.components.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ])
  );

  const added: string[] = [];
  const removed: string[] = [];
  const changed: string[] = [];

  currentMap.forEach((component, key) => {
    const previous = parentMap.get(key);
    if (!previous) {
      added.push(component.type.replace(/_/g, ' '));
      return;
    }
    if (previous.content !== component.content || previous.enabled !== component.enabled) {
      changed.push(component.type.replace(/_/g, ' '));
    }
  });

  parentMap.forEach((component, key) => {
    if (!currentMap.has(key)) {
      removed.push(component.type.replace(/_/g, ' '));
    }
  });

  return { added, removed, changed };
}

function buildEntries(
  promptId: string,
  versions: PromptVersion[],
  draftLeaf: Props['draftLeaf'],
): RenderEntry[] {
  const savedEntries: RenderEntry[] = [...versions]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .map((version) => ({
      kind: 'saved',
      versionId: version.version_id,
      promptId: version.prompt_id,
      parentVersionId: version.parent_version_id ?? null,
      createdAt: version.created_at,
      components: version.components,
      revisionNote: version.revision_note ?? undefined,
      version,
    }));

  if (!draftLeaf || draftLeaf.promptId !== promptId) {
    return savedEntries;
  }

  return [
    {
      kind: 'draft',
      versionId: draftLeaf.versionId,
      promptId: draftLeaf.promptId,
      parentVersionId: draftLeaf.parentVersionId,
      createdAt: new Date().toISOString(),
      components: draftLeaf.components,
      revisionNote: draftLeaf.revisionNote,
    },
    ...savedEntries,
  ];
}

function buildGraphRows(entries: RenderEntry[]): { rows: GraphRow[]; laneCount: number } {
  const rows: GraphRow[] = [];
  let active: Array<string | null> = [];
  let maxLane = 0;

  entries.forEach((entry) => {
    const before = [...active];
    let lane = before.indexOf(entry.versionId);
    const incoming = lane >= 0;
    const during = [...before];

    if (!incoming) {
      lane = during.findIndex((value) => value === null);
      if (lane < 0) {
        lane = during.length;
      }
      during[lane] = entry.versionId;
    }

    const after = [...during];
    let parentLane: number | null = null;

    if (entry.parentVersionId) {
      const existingParentLane = after.indexOf(entry.parentVersionId);
      if (existingParentLane >= 0 && existingParentLane !== lane) {
        parentLane = existingParentLane;
        after[lane] = null;
      } else {
        parentLane = lane;
        after[lane] = entry.parentVersionId;
      }
    } else {
      after[lane] = null;
    }

    while (after.length > 0 && after[after.length - 1] === null) {
      after.pop();
    }

    maxLane = Math.max(maxLane, lane, parentLane ?? 0, before.length - 1, after.length - 1);
    rows.push({
      entry,
      lane,
      parentLane,
      incoming,
      before,
      after,
    });
    active = after;
  });

  return {
    rows,
    laneCount: Math.max(1, maxLane + 1),
  };
}

function getLaneColor(lane: number) {
  return LANE_COLORS[lane % LANE_COLORS.length];
}

function getRgbChannels(hex: string) {
  const normalized = hex.replace('#', '');
  if (normalized.length !== 6) {
    return '59, 130, 246';
  }

  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);

  if ([r, g, b].some((channel) => Number.isNaN(channel))) {
    return '59, 130, 246';
  }

  return `${r}, ${g}, ${b}`;
}

function getBranchPath(startX: number, startY: number, endX: number, endY: number) {
  const deltaY = endY - startY;
  const entryCurveY = startY + Math.min(18, deltaY * 0.42);
  const exitCurveY = endY - Math.min(18, deltaY * 0.28);

  return `M ${startX} ${startY} C ${startX} ${entryCurveY}, ${endX} ${exitCurveY}, ${endX} ${endY}`;
}

const PromptVersionTree: React.FC<Props> = ({
  promptId,
  versions,
  activeVersionId = null,
  compareTargets = [],
  draftLeaf = null,
  onLoadVersion,
  onToggleCompare,
}) => {
  const versionMap = React.useMemo(
    () => new Map(versions.map((version) => [version.version_id, version])),
    [versions]
  );
  const entries = React.useMemo(
    () => buildEntries(promptId, versions, draftLeaf),
    [promptId, versions, draftLeaf]
  );
  const { rows, laneCount } = React.useMemo(() => buildGraphRows(entries), [entries]);
  const laneWidth = 18;
  const graphWidth = laneCount * laneWidth;
  const effectiveActiveVersionId = draftLeaf && activeVersionId === draftLeaf.parentVersionId
    ? draftLeaf.versionId
    : activeVersionId;

  const renderDiffTags = (entry: RenderEntry) => {
    if (entry.kind === 'draft') {
      return (
        <div className="version-tree__delta">
          <span className="version-tree__edge-tag version-tree__edge-tag--changed">~ unsaved draft</span>
        </div>
      );
    }

    const version = entry.version;
    if (!version?.parent_version_id) {
      return null;
    }

    const parent = versionMap.get(version.parent_version_id) ?? null;

    if (!parent) {
      return (
        <div className="version-tree__delta">
          <span className="version-tree__edge-tag">parent version unavailable</span>
        </div>
      );
    }

    const diffSummary = getComponentDiffSummary(version, parent);

    return (
      <div className="version-tree__delta">
        {diffSummary.added.length === 0 && diffSummary.removed.length === 0 && diffSummary.changed.length === 0 ? (
          <span className="version-tree__edge-tag">no component edit</span>
        ) : (
          <>
            {diffSummary.added.map((label, index) => (
              <span key={`added-${entry.versionId}-${label}-${index}`} className="version-tree__edge-tag version-tree__edge-tag--added">
                + {label}
              </span>
            ))}
            {diffSummary.removed.map((label, index) => (
              <span key={`removed-${entry.versionId}-${label}-${index}`} className="version-tree__edge-tag version-tree__edge-tag--removed">
                - {label}
              </span>
            ))}
            {diffSummary.changed.map((label, index) => (
              <span key={`changed-${entry.versionId}-${label}-${index}`} className="version-tree__edge-tag version-tree__edge-tag--changed">
                ~ {label}
              </span>
            ))}
          </>
        )}
      </div>
    );
  };

  const renderGraph = (row: GraphRow) => {
    const graphHeight = 64;
    const nodeY = 18;
    const nodeX = row.lane * laneWidth + laneWidth / 2;
    const nodeLeft = `${(nodeX / graphWidth) * 100}%`;
    const nodeTop = `${(nodeY / graphHeight) * 100}%`;
    const nodeColor = row.entry.kind === 'draft' ? '#f59e0b' : getLaneColor(row.lane);
    const nodeColorRgb = getRgbChannels(nodeColor);

    return (
      <div className="version-tree__graph-shell">
        <svg
          className="version-tree__graph-svg"
          viewBox={`0 0 ${graphWidth} ${graphHeight}`}
          preserveAspectRatio="none"
          aria-hidden
        >
          {Array.from({ length: laneCount }, (_, laneIndex) => {
            if (laneIndex === row.lane) {
              return null;
            }

            const beforeActive = row.before[laneIndex] !== undefined && row.before[laneIndex] !== null;
            const afterActive = row.after[laneIndex] !== undefined && row.after[laneIndex] !== null;

            if (!beforeActive && !afterActive) {
              return null;
            }

            const x = laneIndex * laneWidth + laneWidth / 2;
            return (
              <line
                key={`lane-${row.entry.versionId}-${laneIndex}`}
                className="version-tree__graph-line"
                style={{ stroke: getLaneColor(laneIndex) }}
                x1={x}
                y1={-2}
                x2={x}
                y2={graphHeight + 2}
              />
            );
          })}

          {row.incoming && (
            <line
              className="version-tree__graph-line"
              style={{ stroke: nodeColor }}
              x1={nodeX}
              y1={-2}
              x2={nodeX}
              y2={nodeY}
            />
          )}

          {row.parentLane !== null && row.parentLane === row.lane && (
            <line
              className="version-tree__graph-line"
              style={{ stroke: nodeColor }}
              x1={nodeX}
              y1={nodeY}
              x2={nodeX}
              y2={graphHeight + 2}
            />
          )}

          {row.parentLane !== null && row.parentLane !== row.lane && (
            <path
              className="version-tree__graph-line"
              style={{ stroke: nodeColor }}
              d={getBranchPath(
                nodeX,
                nodeY,
                row.parentLane * laneWidth + laneWidth / 2,
                graphHeight + 2
              )}
              fill="none"
            />
          )}
        </svg>
        <span
          className={`version-tree__graph-node${row.entry.kind === 'draft' ? ' is-draft' : ''}${effectiveActiveVersionId === row.entry.versionId ? ' is-active' : ''}`}
          style={
            {
              left: nodeLeft,
              top: nodeTop,
              '--version-tree-node-color': nodeColor,
              '--version-tree-node-rgb': nodeColorRgb,
            } as React.CSSProperties
          }
          aria-hidden
        />
      </div>
    );
  };

  return (
    <div className="version-tree">
      {rows.length === 0 ? (
        <div className="version-tree__empty">No versions saved yet.</div>
      ) : (
        rows.map((row) => {
          const entry = row.entry;
          const savedVersion = entry.kind === 'saved' ? entry.version ?? null : null;
          const isSavedVersion = savedVersion !== null;
          const isActive = effectiveActiveVersionId === entry.versionId;
          const compareIndex = compareTargets.findIndex(
            (target) => target.promptId === promptId && target.versionId === entry.versionId
          );
          const componentLabels = getComponentLabels(entry.components);
          const diffTags = renderDiffTags(entry);
          const diffLabel = entry.kind === 'draft' ? 'draft status' : 'changes from parent';

          return (
            <div key={entry.versionId} className="version-tree__row">
              <div className="version-tree__graph" style={{ width: graphWidth }}>
                {renderGraph(row)}
              </div>
              <div className={`version-tree__node${isActive ? ' is-active' : ''}${onLoadVersion && isSavedVersion ? ' is-clickable' : ''}${entry.kind === 'draft' ? ' version-tree__node--draft' : ''}`}>
                <div className="version-tree__node-top">
                  <div
                    className="version-tree__node-main"
                    role={onLoadVersion && isSavedVersion ? 'button' : undefined}
                    tabIndex={onLoadVersion && isSavedVersion ? 0 : undefined}
                    aria-label={onLoadVersion && isSavedVersion ? `load version ${entry.versionId}` : undefined}
                    onClick={onLoadVersion && isSavedVersion ? () => onLoadVersion(savedVersion) : undefined}
                    onKeyDown={onLoadVersion && isSavedVersion ? (event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onLoadVersion(savedVersion);
                      }
                    } : undefined}
                  >
                    <div className="version-tree__node-head">
                      <span className="version-tree__version-id">{entry.versionId}</span>
                      {entry.kind === 'draft' && (
                        <span className="badge badge--warning">draft</span>
                      )}
                      {compareIndex >= 0 && (
                        <span className="badge badge--primary">compare {compareIndex + 1}</span>
                      )}
                    </div>
                  </div>
                  <div className="version-tree__node-actions">
                    {onLoadVersion && isSavedVersion && (
                      <button
                        type="button"
                        className="btn btn--ghost btn--sm version-tree__action-btn"
                        aria-label={`load version ${entry.versionId}`}
                        title="load version"
                        onClick={() => onLoadVersion(savedVersion)}
                      >
                        <span
                          className="version-tree__action-icon"
                          style={getMaskIconStyle(iconLoad)}
                          aria-hidden
                        />
                        <span>Load</span>
                      </button>
                    )}
                    {onToggleCompare && isSavedVersion && (
                      <button
                        type="button"
                        className="btn btn--ghost btn--sm version-tree__action-btn"
                        aria-label={compareIndex >= 0 ? `remove compare for version ${entry.versionId}` : `compare version ${entry.versionId}`}
                        title={compareIndex >= 0 ? 'remove compare' : 'compare version'}
                        onClick={() => onToggleCompare(savedVersion)}
                      >
                        <span
                          className="version-tree__action-icon"
                          style={getMaskIconStyle(iconCompare)}
                          aria-hidden
                        />
                        <span>{compareIndex >= 0 ? 'Remove' : 'Compare'}</span>
                      </button>
                    )}
                  </div>
                </div>
                <div className="version-tree__meta-section">
                  <div className="section-label version-tree__meta-label">components</div>
                  <div className="version-tree__node-components">
                    {componentLabels.length > 0 ? componentLabels.map((componentLabel) => (
                      <span key={`${entry.versionId}-${componentLabel.key}`} className="version-tree__component-chip">
                        {componentLabel.label}
                      </span>
                    )) : (
                      <span className="version-tree__component-chip version-tree__component-chip--empty">
                        no active components
                      </span>
                    )}
                  </div>
                </div>
                {entry.revisionNote && (
                  <div className="version-tree__meta-section">
                    <div className="section-label version-tree__meta-label">revision note</div>
                    <div className="version-tree__node-note">{entry.revisionNote}</div>
                  </div>
                )}
                {diffTags && (
                  <div className="version-tree__meta-section">
                    <div className="section-label version-tree__meta-label">{diffLabel}</div>
                    {diffTags}
                  </div>
                )}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
};

export default PromptVersionTree;
