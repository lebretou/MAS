import React from 'react';
import type {
  PromptComponent,
  PromptComponentType,
  PromptMessageRole,
} from '../../../types/prompt';
import { resizeTextarea } from '../../../utils/resizeTextarea';
import iconCollapse from '../../../assets/icon-collapse.svg';
import iconCopy from '../../../assets/icon-copy.svg';
import iconExpand from '../../../assets/icon-expand.svg';
import iconDragHandle from '../../../assets/icon-draghandle.svg';
import iconTrash from '../../../assets/icon-trash.svg';
import {
  COMPONENT_PLACEHOLDERS,
  COMPONENT_LABELS,
  PROVIDED_COMPONENT_TYPES,
  createPromptComponentId,
  getDefaultPromptMessageRole,
  getDefaultPromptComponentName,
  getPromptComponentDisplayName,
  getPromptComponentLabel,
  getPromptComponentKey,
  normalizePromptComponents,
} from '../promptEditor';

interface Props {
  components: PromptComponent[];
  onChange: (components: PromptComponent[]) => void;
  showAddControl?: boolean;
  collapsedSections?: Record<string, boolean>;
  onToggleCollapse?: (componentKey: string) => void;
  onCopyComponentContent?: (index: number) => void;
  onReorderComponent?: (fromIndex: number, toIndex: number) => void;
  registerSectionRef?: (componentKey: string, element: HTMLDivElement | null) => void;
  componentStats?: Record<PromptComponentType, number> | null;
  componentStatsLabel?: string | null;
  highlightRequest?: PromptVariableHighlightRequest | null;
}

export interface PromptVariableMatch {
  componentKey: string;
  componentIndex: number;
  start: number;
  end: number;
  token: string;
}

export interface PromptVariableHighlightRequest extends PromptVariableMatch {
  requestId: number;
}

interface HighlightSegments {
  before: string;
  token: string;
  after: string;
}

const MESSAGE_ROLE_LABELS: Record<PromptMessageRole, string> = {
  system: 'System',
  human: 'Human',
  ai: 'AI',
};

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

export function findPromptVariableMatch(
  components: PromptComponent[],
  variableName: string,
): PromptVariableMatch | null {
  const token = `{{${variableName}}}`;
  const normalizedComponents = normalizePromptComponents(components);

  const match = normalizedComponents.find((component) => component.enabled && component.content.includes(token))
    ?? normalizedComponents.find((component) => component.content.includes(token));

  if (match) {
    const componentIndex = normalizedComponents.indexOf(match);
    const start = match.content.indexOf(token);

    return {
      componentKey: getPromptComponentKey(match, componentIndex),
      componentIndex,
      start,
      end: start + token.length,
      token,
    };
  }

  return null;
}

function getHighlightSegments(value: string, start: number, end: number): HighlightSegments | null {
  if (start < 0 || end <= start || end > value.length) {
    return null;
  }

  return {
    before: value.slice(0, start),
    token: value.slice(start, end),
    after: value.slice(end),
  };
}

const PromptComponentEditor: React.FC<Props> = ({
  components,
  onChange,
  showAddControl = true,
  collapsedSections = {},
  onToggleCollapse,
  onCopyComponentContent,
  onReorderComponent,
  registerSectionRef,
  componentStats,
  componentStatsLabel,
  highlightRequest = null,
}) => {
  const textareaRefs = React.useRef<Array<HTMLTextAreaElement | null>>([]);
  const nameInputRef = React.useRef<HTMLInputElement | null>(null);
  const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = React.useState<number | null>(null);
  const [editingNameKey, setEditingNameKey] = React.useState<string | null>(null);
  const [editingNameValue, setEditingNameValue] = React.useState('');
  const [activeHighlightRequest, setActiveHighlightRequest] = React.useState<PromptVariableHighlightRequest | null>(null);
  const normalizedComponents = React.useMemo(
    () => normalizePromptComponents(components),
    [components]
  );
  const usedTypes = new Set(normalizedComponents.map(c => c.type));
  const availableTypes = [
    ...PROVIDED_COMPONENT_TYPES.filter(t => !usedTypes.has(t)),
    'custom' as const,
  ];

  React.useEffect(() => {
    textareaRefs.current.forEach((textarea) => {
      resizeTextarea(textarea);
    });
  }, [collapsedSections, normalizedComponents]);

  React.useEffect(() => {
    if (!editingNameKey) {
      return;
    }

    nameInputRef.current?.focus();
    nameInputRef.current?.select();
  }, [editingNameKey]);

  React.useEffect(() => {
    if (!highlightRequest) {
      return undefined;
    }

    setActiveHighlightRequest(highlightRequest);

    const textarea = textareaRefs.current[highlightRequest.componentIndex];
    textarea?.focus({ preventScroll: true });
    textarea?.setSelectionRange(highlightRequest.start, highlightRequest.end);

    const timeoutId = window.setTimeout(() => {
      setActiveHighlightRequest((current) => (
        current?.requestId === highlightRequest.requestId ? null : current
      ));

      if (textarea && document.activeElement === textarea) {
        textarea.setSelectionRange(highlightRequest.end, highlightRequest.end);
      }
    }, 1800);

    return () => window.clearTimeout(timeoutId);
  }, [highlightRequest]);

  const addComponent = (type: typeof availableTypes[number]) => {
    onChange([
      ...normalizedComponents,
      {
        component_id: createPromptComponentId(),
        type,
        name: getDefaultPromptComponentName(type),
        content: '',
        enabled: true,
        message_role: getDefaultPromptMessageRole(type),
      },
    ]);
  };

  const removeComponent = (index: number) => {
    onChange(normalizedComponents.filter((_, i) => i !== index));
  };

  const updateComponent = (index: number, patch: Partial<PromptComponent>) => {
    onChange(normalizedComponents.map((c, i) => (i === index ? { ...c, ...patch } : c)));
  };

  const startEditingName = (componentKey: string, currentName: string) => {
    setEditingNameKey(componentKey);
    setEditingNameValue(currentName);
  };

  const stopEditingName = () => {
    setEditingNameKey(null);
    setEditingNameValue('');
  };

  const commitNameEdit = (index: number, type: PromptComponentType) => {
    updateComponent(index, {
      name: editingNameValue.trim() || getDefaultPromptComponentName(type),
    });
    stopEditingName();
  };

  return (
    <div className="prompt-components">
      {normalizedComponents.map((comp, i) => {
        const componentKey = getPromptComponentKey(comp, i);
        const isCollapsed = collapsedSections[componentKey] ?? false;
        const displayName = getPromptComponentDisplayName(comp);
        const typeLabel = getPromptComponentLabel(comp.type);
        const isEditingName = editingNameKey === componentKey;
        const isVariableHighlighted = activeHighlightRequest?.componentKey === componentKey;
        const highlightSegments = isVariableHighlighted
          ? getHighlightSegments(comp.content, activeHighlightRequest.start, activeHighlightRequest.end)
          : null;

        return (
          <div
            key={componentKey}
            className={`prompt-components__row${draggedIndex === i ? ' is-dragging' : ''}${!onReorderComponent ? ' prompt-components__row--static' : ''}`}
          >
            {onReorderComponent && (
              <button
                type="button"
                className="icon-btn prompt-components__drag-handle"
                draggable
                onDragStart={(event) => {
                  setDraggedIndex(i);
                  setDropTargetIndex(null);
                  event.dataTransfer.effectAllowed = 'move';
                  event.dataTransfer.setData('text/plain', String(i));
                }}
                onDragEnd={() => {
                  setDraggedIndex(null);
                  setDropTargetIndex(null);
                }}
                aria-label={`drag ${displayName} component`}
                title="drag to reorder"
              >
                <span
                  className="prompt-components__action-icon"
                  style={getMaskIconStyle(iconDragHandle)}
                  aria-hidden
                />
              </button>
            )}
            <div
              ref={(element) => registerSectionRef?.(componentKey, element)}
              className={`card prompt-components__card${!comp.enabled ? ' prompt-components__card--disabled' : ''}${draggedIndex === i ? ' is-dragging' : ''}${dropTargetIndex === i ? ' is-drop-target' : ''}${isVariableHighlighted ? ' is-variable-highlighted' : ''}`}
              onDragOver={(event) => {
                if (!onReorderComponent || draggedIndex === null) {
                  return;
                }
                event.preventDefault();
                if (dropTargetIndex !== i) {
                  setDropTargetIndex(i);
                }
              }}
              onDrop={(event) => {
                if (!onReorderComponent || draggedIndex === null) {
                  return;
                }
                event.preventDefault();
                if (draggedIndex !== i) {
                  onReorderComponent(draggedIndex, i);
                }
                setDraggedIndex(null);
                setDropTargetIndex(null);
              }}
            >
              <div className="card__header prompt-components__header">
                <div className="prompt-components__header-left">
                  <div className="prompt-components__toggle">
                    <input
                      type="checkbox"
                      checked={comp.enabled}
                      onChange={e => updateComponent(i, { enabled: e.target.checked })}
                      aria-label={`enable ${displayName} component`}
                    />
                  </div>
                  {isEditingName ? (
                    <input
                      ref={nameInputRef}
                      className="input prompt-components__name-inline-input"
                      value={editingNameValue}
                      onChange={(e) => setEditingNameValue(e.target.value)}
                      onBlur={() => commitNameEdit(i, comp.type)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
                          event.preventDefault();
                          commitNameEdit(i, comp.type);
                        }
                        if (event.key === 'Escape') {
                          event.preventDefault();
                          stopEditingName();
                        }
                      }}
                      aria-label={`${typeLabel} component name`}
                    />
                  ) : (
                    <button
                      type="button"
                      className="type-badge prompt-components__name-trigger"
                      onClick={() => startEditingName(componentKey, displayName)}
                      title="click to rename"
                      disabled={!comp.enabled}
                      aria-label={`rename ${displayName} component`}
                    >
                      {displayName}
                    </button>
                  )}
                  <span className="badge badge--neutral prompt-components__role-badge">
                    {MESSAGE_ROLE_LABELS[comp.message_role ?? getDefaultPromptMessageRole(comp.type)]}
                  </span>
                  {componentStats?.[comp.type] != null && (
                    <span
                      className="badge badge--prevalence"
                      title={componentStatsLabel ? `${Math.round(componentStats[comp.type] * 100)}% of ${componentStatsLabel} agents include this` : undefined}
                    >
                      {Math.round(componentStats[comp.type] * 100)}%
                    </span>
                  )}
                </div>
                <div className="prompt-components__header-actions">
                  <div className="prompt-components__hover-actions">
                    <select
                      className="select prompt-components__role-select"
                      value={comp.message_role ?? getDefaultPromptMessageRole(comp.type)}
                      onChange={(e) => updateComponent(i, { message_role: e.target.value as PromptMessageRole })}
                      aria-label={`${displayName} message role`}
                    >
                      {Object.entries(MESSAGE_ROLE_LABELS).map(([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                    {onCopyComponentContent && (
                      <button
                        type="button"
                        className="icon-btn prompt-components__icon-btn"
                        onClick={() => onCopyComponentContent(i)}
                        aria-label={`copy ${displayName} content`}
                        title="copy content"
                      >
                        <span
                          className="prompt-components__action-icon"
                          style={getMaskIconStyle(iconCopy)}
                          aria-hidden
                        />
                      </button>
                    )}
                    {onToggleCollapse && (
                      <button
                        type="button"
                        className="icon-btn prompt-components__icon-btn"
                        onClick={() => onToggleCollapse(componentKey)}
                        aria-label={`${isCollapsed ? 'expand' : 'collapse'} ${displayName} component`}
                        title={isCollapsed ? 'expand' : 'collapse'}
                        aria-expanded={!isCollapsed}
                      >
                        <span
                          className="prompt-components__action-icon"
                          style={getMaskIconStyle(isCollapsed ? iconExpand : iconCollapse)}
                          aria-hidden
                        />
                      </button>
                    )}
                    <button
                      type="button"
                      className="icon-btn prompt-components__icon-btn"
                      onClick={() => removeComponent(i)}
                      title="Remove component"
                      aria-label={`remove ${displayName} component`}
                    >
                      <span
                        className="prompt-components__action-icon"
                        style={getMaskIconStyle(iconTrash)}
                        aria-hidden
                      />
                    </button>
                  </div>
                </div>
              </div>
              {!isCollapsed && (
                <div className="card__body prompt-components__body">
                  <div className="prompt-components__textarea-wrap">
                    {highlightSegments && (
                      <div className="prompt-components__textarea-highlight" aria-hidden="true">
                        <span className="prompt-components__textarea-highlight-copy">
                          {highlightSegments.before}
                          <mark className="prompt-components__textarea-highlight-token">
                            {highlightSegments.token}
                          </mark>
                          {highlightSegments.after}
                        </span>
                      </div>
                    )}
                    <textarea
                      ref={(element) => {
                        textareaRefs.current[i] = element;
                        resizeTextarea(element);
                      }}
                      className="textarea textarea--adaptive"
                      value={comp.content}
                      onChange={e => {
                        resizeTextarea(e.target);
                        updateComponent(i, { content: e.target.value });
                      }}
                      rows={2}
                      placeholder={COMPONENT_PLACEHOLDERS[comp.type]}
                      disabled={!comp.enabled}
                      aria-label={`${displayName} component content`}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })}

      {showAddControl && availableTypes.length > 0 && (
        <div className="prompt-components__add">
          {availableTypes.some(t => t !== 'custom') && (
            <select
              className="select prompt-components__type-select"
              defaultValue=""
              onChange={e => {
                if (e.target.value) {
                  addComponent(e.target.value as PromptComponentType);
                  e.target.value = '';
                }
              }}
            >
              <option value="" disabled>
                + Add component...
              </option>
              {availableTypes.filter(t => t !== 'custom').map(t => (
                <option key={t} value={t}>
                  {COMPONENT_LABELS[t]}
                </option>
              ))}
            </select>
          )}
          {availableTypes.includes('custom') && (
            <button
              type="button"
              className="btn btn--secondary btn--sm prompt-components__custom-btn"
              onClick={() => addComponent('custom')}
            >
              + Custom section
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default PromptComponentEditor;
