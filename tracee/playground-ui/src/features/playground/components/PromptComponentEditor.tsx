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

const PromptComponentEditor: React.FC<Props> = ({
  components,
  onChange,
  showAddControl = true,
  collapsedSections = {},
  onToggleCollapse,
  onCopyComponentContent,
  onReorderComponent,
  registerSectionRef,
}) => {
  const textareaRefs = React.useRef<Array<HTMLTextAreaElement | null>>([]);
  const nameInputRef = React.useRef<HTMLInputElement | null>(null);
  const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = React.useState<number | null>(null);
  const [editingNameKey, setEditingNameKey] = React.useState<string | null>(null);
  const [editingNameValue, setEditingNameValue] = React.useState('');
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
              className={`card prompt-components__card${!comp.enabled ? ' prompt-components__card--disabled' : ''}${draggedIndex === i ? ' is-dragging' : ''}${dropTargetIndex === i ? ' is-drop-target' : ''}`}
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
              )}
            </div>
          </div>
        );
      })}

      {showAddControl && availableTypes.length > 0 && (
        <div className="prompt-components__add">
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
            {availableTypes.map(t => (
              <option key={t} value={t}>
                {COMPONENT_LABELS[t]}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};

export default PromptComponentEditor;
