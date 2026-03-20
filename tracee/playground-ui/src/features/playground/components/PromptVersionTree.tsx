import React from 'react';
import type { PromptVersion } from '../../../types/prompt';

interface CompareTarget {
  promptId: string;
  versionId: string;
}

interface Props {
  promptId: string;
  promptName: string;
  versions: PromptVersion[];
  activeVersionId?: string | null;
  compareTargets?: CompareTarget[];
  onLoadVersion?: (version: PromptVersion) => void;
  onToggleCompare?: (version: PromptVersion) => void;
}

interface TreeNode {
  version: PromptVersion;
  children: TreeNode[];
}

interface ComponentDiffSummary {
  added: string[];
  removed: string[];
  changed: string[];
}

function getComponentLabels(version: PromptVersion) {
  return version.components
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

function buildTree(versions: PromptVersion[]): TreeNode[] {
  const nodeMap = new Map<string, TreeNode>();
  const ordered = [...versions].sort((a, b) => a.created_at.localeCompare(b.created_at));

  ordered.forEach((version) => {
    nodeMap.set(version.version_id, {
      version,
      children: [],
    });
  });

  const roots: TreeNode[] = [];
  ordered.forEach((version) => {
    const node = nodeMap.get(version.version_id);
    if (!node) return;

    const parentId = version.parent_version_id;
    if (parentId && nodeMap.has(parentId)) {
      nodeMap.get(parentId)?.children.push(node);
      return;
    }
    roots.push(node);
  });

  return roots;
}

const PromptVersionTree: React.FC<Props> = ({
  promptId,
  promptName,
  versions,
  activeVersionId = null,
  compareTargets = [],
  onLoadVersion,
  onToggleCompare,
}) => {
  const versionMap = React.useMemo(
    () => new Map(versions.map((version) => [version.version_id, version])),
    [versions]
  );
  const roots = React.useMemo(() => buildTree(versions), [versions]);

  const renderNode = (node: TreeNode, depth = 0) => {
    const version = node.version;
    const parent = version.parent_version_id ? versionMap.get(version.parent_version_id) ?? null : null;
    const isActive = activeVersionId === version.version_id;
    const componentLabels = getComponentLabels(version);
    const diffSummary = getComponentDiffSummary(version, parent);
    const compareIndex = compareTargets.findIndex(
      (target) => target.promptId === promptId && target.versionId === version.version_id
    );

    return (
      <div key={version.version_id} className="version-tree__branch">
        {parent && (
          <div className="version-tree__edge" style={{ marginLeft: depth * 18 + 18 }}>
            {diffSummary.added.length === 0 && diffSummary.removed.length === 0 && diffSummary.changed.length === 0 ? (
              <span className="version-tree__edge-tag">no component edit</span>
            ) : (
              <>
                {diffSummary.added.map((label) => (
                  <span key={`added-${label}`} className="version-tree__edge-tag version-tree__edge-tag--added">
                    + {label}
                  </span>
                ))}
                {diffSummary.removed.map((label) => (
                  <span key={`removed-${label}`} className="version-tree__edge-tag version-tree__edge-tag--removed">
                    - {label}
                  </span>
                ))}
                {diffSummary.changed.map((label) => (
                  <span key={`changed-${label}`} className="version-tree__edge-tag version-tree__edge-tag--changed">
                    ~ {label}
                  </span>
                ))}
              </>
            )}
          </div>
        )}
        <div
          className={`version-tree__node${isActive ? ' is-active' : ''}`}
          style={{ marginLeft: depth * 18 }}
        >
          <div className="version-tree__node-main">
            <div className="version-tree__node-head">
              <span className="version-tree__node-dot" aria-hidden />
              <span className="version-tree__version-id">{version.version_id}</span>
              {version.branch_name && (
                <span className="badge badge--neutral">{version.branch_name}</span>
              )}
              {compareIndex >= 0 && (
                <span className="badge badge--primary">compare {compareIndex + 1}</span>
              )}
            </div>
            <div className="version-tree__node-name">{version.name || promptName}</div>
            <div className="version-tree__node-components">
              {componentLabels.length > 0 ? componentLabels.map((componentLabel) => (
                <span key={`${version.version_id}-${componentLabel.key}`} className="version-tree__component-chip">
                  {componentLabel.label}
                </span>
              )) : (
                <span className="version-tree__component-chip version-tree__component-chip--empty">
                  no active components
                </span>
              )}
            </div>
            {version.revision_note && (
              <div className="version-tree__node-note">{version.revision_note}</div>
            )}
          </div>
          <div className="version-tree__node-actions">
            {onLoadVersion && (
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => onLoadVersion(version)}
              >
                load
              </button>
            )}
            {onToggleCompare && (
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => onToggleCompare(version)}
              >
                {compareIndex >= 0 ? 'remove compare' : 'compare'}
              </button>
            )}
          </div>
        </div>
        {node.children.map((child) => renderNode(child, depth + 1))}
      </div>
    );
  };

  return (
    <div className="version-tree">
      {roots.length === 0 ? (
        <div className="version-tree__empty">No versions saved yet.</div>
      ) : (
        roots.map((root) => renderNode(root))
      )}
    </div>
  );
};

export default PromptVersionTree;
