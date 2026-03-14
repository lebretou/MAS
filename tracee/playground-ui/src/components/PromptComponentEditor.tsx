import React from 'react';
import { PromptComponent, PromptComponentType } from '../types/prompt';

interface Props {
  components: PromptComponent[];
  onChange: (components: PromptComponent[]) => void;
}

const COMPONENT_TYPES: PromptComponentType[] = [
  'role',
  'goal',
  'constraints',
  'io_rules',
  'examples',
  'safety',
  'tool_instructions',
];

const COMPONENT_LABELS: Record<PromptComponentType, string> = {
  role: 'Role',
  goal: 'Goal',
  constraints: 'Constraints',
  io_rules: 'I/O Rules',
  examples: 'Examples',
  safety: 'Safety',
  tool_instructions: 'Tool Instructions',
};

const COMPONENT_PLACEHOLDERS: Record<PromptComponentType, string> = {
  role: 'You are a helpful assistant that...',
  goal: 'Your primary goal is to...',
  constraints: 'You must not... You should always...',
  io_rules: 'Input format: ... Output format: ...',
  examples: 'Example 1:\nInput: ...\nOutput: ...',
  safety: 'Never reveal... Always verify...',
  tool_instructions: 'When using the search tool...',
};

const PromptComponentEditor: React.FC<Props> = ({ components, onChange }) => {
  const usedTypes = new Set(components.map(c => c.type));
  const availableTypes = COMPONENT_TYPES.filter(t => !usedTypes.has(t));

  const addComponent = (type: PromptComponentType) => {
    onChange([...components, { type, content: '', enabled: true }]);
  };

  const removeComponent = (index: number) => {
    onChange(components.filter((_, i) => i !== index));
  };

  const updateComponent = (index: number, patch: Partial<PromptComponent>) => {
    onChange(components.map((c, i) => (i === index ? { ...c, ...patch } : c)));
  };

  return (
    <div className="prompt-components">
      {components.map((comp, i) => (
        <div
          key={`${comp.type}-${i}`}
          className={`card prompt-components__card${!comp.enabled ? ' prompt-components__card--disabled' : ''}`}
        >
          <div className="card__header prompt-components__header">
            <div className="prompt-components__header-left">
              <label className="check-label prompt-components__toggle">
                <input
                  type="checkbox"
                  checked={comp.enabled}
                  onChange={e => updateComponent(i, { enabled: e.target.checked })}
                />
                <span className="type-badge">{COMPONENT_LABELS[comp.type]}</span>
              </label>
            </div>
            <button
              type="button"
              className="icon-btn icon-btn--close"
              onClick={() => removeComponent(i)}
              title="Remove component"
            >
              &times;
            </button>
          </div>
          <div className="card__body prompt-components__body">
            <textarea
              className="textarea"
              value={comp.content}
              onChange={e => updateComponent(i, { content: e.target.value })}
              rows={4}
              placeholder={COMPONENT_PLACEHOLDERS[comp.type]}
              disabled={!comp.enabled}
            />
            <span className="field__hint">
              Use {'{{variable_name}}'} for input variables
            </span>
          </div>
        </div>
      ))}

      {availableTypes.length > 0 && (
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
