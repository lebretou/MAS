import React from 'react';
import type { PromptTool, PromptToolArgument, ToolArgumentType } from '../../../types/prompt';

interface Props {
  tools: PromptTool[];
  onChange: (tools: PromptTool[]) => void;
}

const ARGUMENT_TYPES: ToolArgumentType[] = [
  'string',
  'number',
  'integer',
  'boolean',
  'array',
  'object',
];

function createTool(): PromptTool {
  return {
    name: '',
    description: '',
    arguments: [],
  };
}

function createArgument(): PromptToolArgument {
  return {
    name: '',
    description: '',
    type: 'string',
    required: true,
    allowed_values: null,
  };
}

const PromptToolsEditor: React.FC<Props> = ({ tools, onChange }) => {
  const updateTool = (toolIndex: number, patch: Partial<PromptTool>) => {
    onChange(tools.map((tool, index) => (index === toolIndex ? { ...tool, ...patch } : tool)));
  };

  const removeTool = (toolIndex: number) => {
    onChange(tools.filter((_, index) => index !== toolIndex));
  };

  const addTool = () => {
    onChange([...tools, createTool()]);
  };

  const updateArgument = (
    toolIndex: number,
    argumentIndex: number,
    patch: Partial<PromptToolArgument>,
  ) => {
    onChange(
      tools.map((tool, index) => {
        if (index !== toolIndex) return tool;
        return {
          ...tool,
          arguments: tool.arguments.map((argument, nestedIndex) => {
            if (nestedIndex !== argumentIndex) return argument;
            const nextArgument = { ...argument, ...patch };
            if (nextArgument.type !== 'string') {
              nextArgument.allowed_values = null;
            }
            return nextArgument;
          }),
        };
      }),
    );
  };

  const addArgument = (toolIndex: number) => {
    onChange(
      tools.map((tool, index) => (
        index === toolIndex
          ? { ...tool, arguments: [...tool.arguments, createArgument()] }
          : tool
      )),
    );
  };

  const removeArgument = (toolIndex: number, argumentIndex: number) => {
    onChange(
      tools.map((tool, index) => (
        index === toolIndex
          ? { ...tool, arguments: tool.arguments.filter((_, nestedIndex) => nestedIndex !== argumentIndex) }
          : tool
      )),
    );
  };

  return (
    <div className="prompt-tools">
      <div className="prompt-tools__intro">
        <div className="field__hint">
          tools are available during playground runs, but tool calls are only recorded right now.
        </div>
      </div>

      {tools.map((tool, toolIndex) => (
        <div key={`tool-${toolIndex}`} className="card prompt-tools__card">
          <div className="card__header prompt-tools__header">
            <div className="prompt-tools__header-main">
              <span className="type-badge">Tool {toolIndex + 1}</span>
              <span className="field__hint">define the name, description, and arguments</span>
            </div>
            <button
              type="button"
              className="icon-btn icon-btn--close"
              onClick={() => removeTool(toolIndex)}
              title="Remove tool"
            >
              &times;
            </button>
          </div>

          <div className="card__body prompt-tools__body">
            <div className="form-grid">
              <div className="field">
                <label className="field__label">Name</label>
                <input
                  className="input"
                  value={tool.name}
                  onChange={(e) => updateTool(toolIndex, { name: e.target.value })}
                  placeholder="explore_dataset"
                />
              </div>

              <div className="field">
                <label className="field__label">Description</label>
                <input
                  className="input"
                  value={tool.description}
                  onChange={(e) => updateTool(toolIndex, { description: e.target.value })}
                  placeholder="Return basic information about a dataset."
                />
              </div>
            </div>

            <div className="prompt-tools__args">
              <div className="prompt-tools__args-header">
                <span className="field__label">Arguments</span>
                <button
                  type="button"
                  className="btn btn--ghost btn--sm"
                  onClick={() => addArgument(toolIndex)}
                >
                  + Add argument
                </button>
              </div>

              {tool.arguments.length === 0 && (
                <div className="prompt-tools__empty field__hint">
                  add arguments if the tool needs structured inputs.
                </div>
              )}

              {tool.arguments.map((argument, argumentIndex) => (
                <div key={`tool-${toolIndex}-arg-${argumentIndex}`} className="prompt-tools__arg-row">
                  <div className="form-grid">
                    <div className="field">
                      <label className="field__label">Argument name</label>
                      <input
                        className="input"
                        value={argument.name}
                        onChange={(e) => updateArgument(toolIndex, argumentIndex, { name: e.target.value })}
                        placeholder="dataset_name"
                      />
                    </div>

                    <div className="field">
                      <label className="field__label">Type</label>
                      <select
                        className="select"
                        value={argument.type}
                        onChange={(e) => updateArgument(toolIndex, argumentIndex, { type: e.target.value as ToolArgumentType })}
                      >
                        {ARGUMENT_TYPES.map((type) => (
                          <option key={type} value={type}>
                            {type}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="form-grid">
                    <div className="field">
                      <label className="field__label">Description</label>
                      <input
                        className="input"
                        value={argument.description ?? ''}
                        onChange={(e) => updateArgument(toolIndex, argumentIndex, { description: e.target.value })}
                        placeholder="Description of argument..."
                      />
                    </div>

                    <div className="field prompt-tools__arg-options">
                      <label className="check-label">
                        <input
                          type="checkbox"
                          checked={argument.required}
                          onChange={(e) => updateArgument(toolIndex, argumentIndex, { required: e.target.checked })}
                        />
                        Required
                      </label>
                    </div>
                  </div>

                  {argument.type === 'string' && (
                    <div className="field">
                      <label className="check-label">
                        <input
                          type="checkbox"
                          checked={Boolean(argument.allowed_values?.length)}
                          onChange={(e) => updateArgument(
                            toolIndex,
                            argumentIndex,
                            { allowed_values: e.target.checked ? [''] : null },
                          )}
                        />
                        Set allowed values
                      </label>
                      {argument.allowed_values && (
                        <input
                          className="input"
                          value={argument.allowed_values.join(', ')}
                          onChange={(e) => updateArgument(
                            toolIndex,
                            argumentIndex,
                            {
                              allowed_values: e.target.value
                                .split(',')
                                .map((value) => value.trim())
                                .filter(Boolean),
                            },
                          )}
                          placeholder="csv, parquet, json"
                        />
                      )}
                    </div>
                  )}

                  <div className="prompt-tools__arg-actions">
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      onClick={() => removeArgument(toolIndex, argumentIndex)}
                    >
                      Remove argument
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}

      <div className="prompt-tools__actions">
        <button type="button" className="btn btn--ghost btn--sm" onClick={addTool}>
          + Add tool
        </button>
      </div>
    </div>
  );
};

export default PromptToolsEditor;
