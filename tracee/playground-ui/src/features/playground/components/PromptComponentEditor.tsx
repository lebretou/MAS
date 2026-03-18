import React from 'react';
import type { PromptComponent, PromptComponentType } from '../../../types/prompt';

interface Props {
  components: PromptComponent[];
  onChange: (components: PromptComponent[]) => void;
  showAddControl?: boolean;
}

export const COMPONENT_TYPES: PromptComponentType[] = [
  'role',
  'goal',
  'task',
  'constraints',
  'io_rules',
  'inputs',
  'outputs',
  'examples',
  'safety',
  'tool_instructions',
  'external_information',
];

export const COMPONENT_LABELS: Record<PromptComponentType, string> = {
  role: 'Role',
  goal: 'Goal',
  task: 'Task',
  constraints: 'Constraints',
  io_rules: 'I/O Rules',
  inputs: 'Inputs',
  outputs: 'Outputs',
  examples: 'Examples',
  safety: 'Safety',
  tool_instructions: 'Tool Instructions',
  external_information: 'External Information',
};

const COMPONENT_PLACEHOLDERS: Record<PromptComponentType, string> = {
  role: 'You are a helpful assistant that...',
  goal: 'Your primary goal is to...',
  task: 'Your primary task is to...',
  constraints: 'You must not... You should always...',
  io_rules: 'Input format: ... Output format: ...',
  inputs: 'Available inputs, variables, or context...',
  outputs: 'Expected output format or requirements...',
  examples: 'Example 1:\nInput: ...\nOutput: ...',
  safety: 'Never reveal... Always verify...',
  tool_instructions: 'When using the search tool...',
  external_information: 'Relevant external facts or background context...',
};

const PromptComponentEditor: React.FC<Props> = ({
  components,
  onChange,
  showAddControl = true,
}) => {
  const textareaRefs = React.useRef<Array<HTMLTextAreaElement | null>>([]);
  const usedTypes = new Set(components.map(c => c.type));
  const availableTypes = COMPONENT_TYPES.filter(t => !usedTypes.has(t));

  React.useEffect(() => {
    textareaRefs.current.forEach((textarea) => {
      if (!textarea) return;
      textarea.style.height = '0px';
      textarea.style.height = `${textarea.scrollHeight}px`;
    });
  }, [components]);

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
              aria-label={`remove ${COMPONENT_LABELS[comp.type]} component`}
            >
              &times;
            </button>
          </div>
          <div className="card__body prompt-components__body">
            <textarea
              ref={(element) => {
                textareaRefs.current[i] = element;
              }}
              className="textarea"
              value={comp.content}
              onChange={e => {
                e.target.style.height = '0px';
                e.target.style.height = `${e.target.scrollHeight}px`;
                updateComponent(i, { content: e.target.value });
              }}
              rows={2}
              placeholder={COMPONENT_PLACEHOLDERS[comp.type]}
              disabled={!comp.enabled}
              aria-label={`${COMPONENT_LABELS[comp.type]} component content`}
            />
          </div>
        </div>
      ))}

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
