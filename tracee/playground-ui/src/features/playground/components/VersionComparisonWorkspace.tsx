import React from 'react';
import type { PromptVersion } from '../../../types/prompt';
import type { PlaygroundRun } from '../../../types/playground';
import { playgroundAPI } from '../../../services/api';
import { computeDiff } from '../../../utils/jsonDiff';

interface CompareTarget {
  promptId: string;
  promptName: string;
  version: PromptVersion;
}

interface Props {
  targets: [CompareTarget, CompareTarget];
}

interface VersionRunSet {
  runs: PlaygroundRun[];
  latestRun: PlaygroundRun | null;
  averageLatency: number | null;
}

function getRunSetKey(target: CompareTarget) {
  return `${target.promptId}:${target.version.version_id}`;
}

function formatJson(value: unknown): string {
  if (value === null || value === undefined) {
    return 'none';
  }

  return JSON.stringify(value, null, 2);
}

function formatOutput(value: string): string {
  const normalized = value.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();

  try {
    return JSON.stringify(JSON.parse(normalized), null, 2);
  } catch {
    return normalized;
  }
}

function formatPromptVersion(version: PromptVersion): string {
  const sections: string[] = [
    `prompt_id: ${version.prompt_id}`,
    `version_id: ${version.version_id}`,
    `name: ${version.name}`,
    `branch: ${version.branch_name ?? 'default'}`,
    `revision_note: ${version.revision_note ?? 'none'}`,
    '',
    'components:',
  ];

  if (version.components.length === 0) {
    sections.push('- none');
  } else {
    version.components.forEach((component, index) => {
      sections.push(`- [${index + 1}] ${component.type} (${component.enabled ? 'enabled' : 'disabled'})`);
      sections.push(component.content || '(empty)');
    });
  }

  sections.push('', 'variables:', formatJson(version.variables ?? null));
  sections.push('', 'tools:', formatJson(version.tools ?? null));
  sections.push('', 'output_schema:', formatJson(version.output_schema ?? null));

  return sections.join('\n');
}

function getComponentSummary(version: PromptVersion) {
  return version.components
    .filter((component) => component.enabled)
    .map((component, index) => ({
      key: component.component_id ?? `${component.type}-${index}`,
      label: component.type.replace(/_/g, ' '),
    }));
}

function getComponentDelta(
  baseVersion: PromptVersion,
  targetVersion: PromptVersion,
): Array<{ kind: 'added' | 'removed' | 'changed'; label: string }> {
  const baseMap = new Map(
    baseVersion.components.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ]),
  );
  const targetMap = new Map(
    targetVersion.components.map((component, index) => [
      component.component_id ?? `${component.type}:${index}`,
      component,
    ]),
  );
  const delta: Array<{ kind: 'added' | 'removed' | 'changed'; label: string }> = [];

  targetMap.forEach((component, key) => {
    const previous = baseMap.get(key);
    if (!previous) {
      delta.push({ kind: 'added', label: component.type.replace(/_/g, ' ') });
      return;
    }
    if (previous.content !== component.content || previous.enabled !== component.enabled) {
      delta.push({ kind: 'changed', label: component.type.replace(/_/g, ' ') });
    }
  });

  baseMap.forEach((component, key) => {
    if (!targetMap.has(key)) {
      delta.push({ kind: 'removed', label: component.type.replace(/_/g, ' ') });
    }
  });

  return delta;
}

function buildRunSet(runs: PlaygroundRun[]): VersionRunSet {
  const orderedRuns = [...runs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const latencyValues = orderedRuns
    .map((run) => run.latency_ms)
    .filter((latency): latency is number => typeof latency === 'number');

  return {
    runs: orderedRuns,
    latestRun: orderedRuns[0] ?? null,
    averageLatency: latencyValues.length > 0
      ? Math.round(latencyValues.reduce((sum, latency) => sum + latency, 0) / latencyValues.length)
      : null,
  };
}

function renderDiffLines(diff: ReturnType<typeof computeDiff>) {
  const hasChanges = diff.some((line) => line.type !== 'same');

  if (!hasChanges) {
    return <div className="diff__identical">No changes detected.</div>;
  }

  return (
    <pre className="diff__pre">
      {diff.map((line, index) => (
        <div key={`${line.type}-${index}`} className={`diff__line diff__line--${line.type}`}>
          <span className="diff__prefix">
            {line.type === 'same' ? '  ' : line.type === 'added' ? '+ ' : '- '}
          </span>
          {line.text}
        </div>
      ))}
    </pre>
  );
}

const VersionComparisonWorkspace: React.FC<Props> = ({ targets }) => {
  const [leftTarget, rightTarget] = targets;
  const [runSets, setRunSets] = React.useState<Record<string, VersionRunSet>>({});
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const requestIdRef = React.useRef(0);

  React.useEffect(() => {
    let cancelled = false;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const promptIds = Array.from(new Set(targets.map((target) => target.promptId)));
    setLoading(true);
    setError(null);
    setRunSets({});

    Promise.allSettled(promptIds.map((promptId) => playgroundAPI.getRunsByPrompt(promptId)))
      .then((results) => {
        if (cancelled || requestIdRef.current !== requestId) {
          return;
        }

        const failed = results.some((result) => result.status === 'rejected');
        if (failed) {
          setError('Failed to load saved runs for the selected versions.');
        }

        const runsByPrompt = new Map<string, PlaygroundRun[]>();
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            runsByPrompt.set(promptIds[index], result.value);
          }
        });

        const nextRunSets: Record<string, VersionRunSet> = {};
        targets.forEach((target) => {
          const promptRuns = runsByPrompt.get(target.promptId) ?? [];
          const versionRuns = promptRuns.filter((run) => run.version_id === target.version.version_id);
          nextRunSets[getRunSetKey(target)] = buildRunSet(versionRuns);
        });
        setRunSets(nextRunSets);
      })
      .finally(() => {
        if (!cancelled && requestIdRef.current === requestId) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    leftTarget.promptId,
    leftTarget.version.version_id,
    rightTarget.promptId,
    rightTarget.version.version_id,
  ]);

  const promptDiff = React.useMemo(
    () => computeDiff(formatPromptVersion(leftTarget.version), formatPromptVersion(rightTarget.version)),
    [leftTarget.version, rightTarget.version],
  );
  const outputDiff = React.useMemo(() => {
    const leftRun = runSets[getRunSetKey(leftTarget)]?.latestRun;
    const rightRun = runSets[getRunSetKey(rightTarget)]?.latestRun;
    if (!leftRun?.output || !rightRun?.output) {
      return null;
    }
    return computeDiff(formatOutput(leftRun.output), formatOutput(rightRun.output));
  }, [leftTarget, rightTarget, runSets]);
  const componentDelta = React.useMemo(
    () => getComponentDelta(leftTarget.version, rightTarget.version),
    [leftTarget.version, rightTarget.version],
  );

  return (
    <div className="version-compare">
      <div className="card">
        <div className="card__body version-compare__header">
          <div>
            <div className="section-label">Selected versions</div>
            <div className="field__hint">
              compare prompt structure with the latest saved runs for each selected version.
            </div>
          </div>
          <div className="version-compare__target-pills">
            <span className="badge badge--neutral">
              A: {leftTarget.promptName} / {leftTarget.version.version_id}
            </span>
            <span className="badge badge--primary">
              B: {rightTarget.promptName} / {rightTarget.version.version_id}
            </span>
          </div>
        </div>
      </div>

      <div className="version-compare__meta-grid">
        {[leftTarget, rightTarget].map((target, index) => {
          const runSet = runSets[getRunSetKey(target)];
          const componentSummary = getComponentSummary(target.version);

          return (
            <div key={getRunSetKey(target)} className="card">
              <div className="card__body version-compare__meta-card">
                <div className="version-compare__meta-head">
                  <span className="badge">{index === 0 ? 'version A' : 'version B'}</span>
                  {target.version.branch_name && (
                    <span className="badge badge--neutral">{target.version.branch_name}</span>
                  )}
                </div>
                <div className="version-compare__version-name">{target.version.name || target.promptName}</div>
                <div className="field__hint">{target.promptId} / {target.version.version_id}</div>
                {target.version.revision_note && (
                  <div className="version-compare__revision-note">{target.version.revision_note}</div>
                )}
                <div className="version-compare__stat-row">
                  <span>{runSet?.runs.length ?? 0} saved runs</span>
                  <span>{runSet?.averageLatency ?? '-'}ms avg latency</span>
                </div>
                <div className="version-compare__stat-row">
                  <span>latest run: {runSet?.latestRun?.created_at.slice(0, 16).replace('T', ' ') ?? 'none'}</span>
                  <span>{runSet?.latestRun?.model ?? 'no model'}</span>
                </div>
                <div className="version-compare__chips">
                  {componentSummary.length > 0 ? componentSummary.map((component) => (
                    <span key={`${getRunSetKey(target)}-${component.key}`} className="version-tree__component-chip">
                      {component.label}
                    </span>
                  )) : (
                    <span className="version-tree__component-chip version-tree__component-chip--empty">
                      no active components
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
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
            <div className="field__hint">the selected versions keep the same enabled component structure.</div>
          )}
        </div>
      </div>

      {loading && (
        <div className="card">
          <div className="card__body version-compare__loading">Loading comparison runs...</div>
        </div>
      )}

      {error && (
        <div className="alert alert--danger">
          <span className="alert__icon">!</span>
          {error}
        </div>
      )}

      <div className="version-compare__diff-grid">
        <div className="card">
          <div className="card__header diff__header">
            <div>
              <h3 className="card__title diff__title">Prompt Diff</h3>
              <span className="field__hint">
                {leftTarget.version.version_id} vs {rightTarget.version.version_id}
              </span>
            </div>
          </div>
          <div className="card__body diff__body">
            <div className="diff__legend">
              <span><span className="diff__legend-removed">-</span> version A</span>
              <span><span className="diff__legend-added">+</span> version B</span>
            </div>
            {renderDiffLines(promptDiff)}
          </div>
        </div>

        <div className="card">
          <div className="card__header diff__header">
            <div>
              <h3 className="card__title diff__title">Latest Output Diff</h3>
              <span className="field__hint">
                latest saved run for each selected version
              </span>
            </div>
          </div>
          <div className="card__body diff__body">
            {outputDiff ? (
              <>
                <div className="diff__legend">
                  <span><span className="diff__legend-removed">-</span> version A latest output</span>
                  <span><span className="diff__legend-added">+</span> version B latest output</span>
                </div>
                {renderDiffLines(outputDiff)}
              </>
            ) : (
              <div className="diff__identical">
                save or run both selected versions to unlock output diff.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VersionComparisonWorkspace;
