import React from 'react';
import type { PromptComponent, PromptTool } from '../../../types/prompt';
import iconStructure from '../../../assets/icon-structure.svg';
import iconTool from '../../../assets/icon-tool.svg';
import iconOutputSchema from '../../../assets/icon-outputschema.svg';
import iconVariable from '../../../assets/icon-variable.svg';
import {
  getDefaultPromptMessageRole,
  getPromptComponentDisplayName,
  getPromptComponentKey,
  normalizePromptComponents,
} from '../promptEditor';

interface Props {
  components: PromptComponent[];
  tools: PromptTool[];
  outputSchema: Record<string, unknown> | null;
  schemaEnabled: boolean;
  schemaError: string | null;
  detectedVariables: string[];
  inputVars: Record<string, string>;
  onJumpToSection: (componentKey: string) => void;
  onJumpToVariable: (variableName: string) => void;
}

const ROLE_LABELS = {
  system: 'system',
  human: 'human',
  ai: 'ai',
};

const SECONDARY_PREVIEW_LIMIT = 6;

export interface OutlineSchemaField {
  name: string;
  typeLabel: string;
}

export interface OutlineVariableState {
  name: string;
  isFilled: boolean;
}

function getSchemaTypeLabel(schema: Record<string, unknown>): string {
  const type = typeof schema.type === 'string' ? schema.type : 'unknown';
  if (type !== 'array') {
    return type;
  }

  const itemType = schema.items && typeof schema.items === 'object'
    ? (schema.items as Record<string, unknown>).type
    : null;

  return typeof itemType === 'string' ? `array<${itemType}>` : 'array';
}

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

export function getOutlineToolNames(tools: PromptTool[]): string[] {
  return tools
    .map((tool) => tool.name.trim())
    .filter(Boolean);
}

export function getOutlineSchemaFields(
  outputSchema: Record<string, unknown> | null | undefined,
): OutlineSchemaField[] {
  if (!outputSchema || typeof outputSchema.properties !== 'object') {
    return [];
  }

  return Object.entries(outputSchema.properties as Record<string, Record<string, unknown>>)
    .map(([name, schema]) => ({
      name,
      typeLabel: getSchemaTypeLabel(schema),
    }));
}

export function getOutlineVariableStates(
  detectedVariables: string[],
  inputVars: Record<string, string>,
): OutlineVariableState[] {
  return detectedVariables.map((name) => ({
    name,
    isFilled: Boolean(inputVars[name]?.trim()),
  }));
}

const PromptStructureOutline: React.FC<Props> = ({
  components,
  tools,
  outputSchema,
  schemaEnabled,
  schemaError,
  detectedVariables,
  inputVars,
  onJumpToSection,
  onJumpToVariable,
}) => {
  const normalizedComponents = React.useMemo(
    () => normalizePromptComponents(components),
    [components]
  );
  const toolNames = React.useMemo(() => getOutlineToolNames(tools), [tools]);
  const schemaFields = React.useMemo(() => getOutlineSchemaFields(outputSchema), [outputSchema]);
  const variableStates = React.useMemo(
    () => getOutlineVariableStates(detectedVariables, inputVars),
    [detectedVariables, inputVars]
  );
  const previewToolNames = toolNames.slice(0, SECONDARY_PREVIEW_LIMIT);
  const previewSchemaFields = schemaFields.slice(0, SECONDARY_PREVIEW_LIMIT);

  return (
    <div className="card">
      <div className="card__body playground-outline">
        <div className="create-run__field-head">
          <div className="playground-outline__headline">
            <div className="field__label">Prompt overview</div>
            <span className="field__hint">compact map of the active structure and attached configuration.</span>
          </div>
          <div className="playground-outline__summary">
            <span>{normalizedComponents.length} sections</span>
            <span>{toolNames.length} tools</span>
            <span>{schemaFields.length} schema fields</span>
            <span>{variableStates.length} variables</span>
          </div>
        </div>
        <div className="playground-outline__section">
          <div className="playground-outline__section-head">
            <span className="playground-outline__section-label">
              <span
                className="playground-outline__section-icon"
                style={getMaskIconStyle(iconStructure)}
                aria-hidden
              />
              <span className="section-label">Structure</span>
            </span>
          </div>
          {normalizedComponents.length === 0 ? (
            <div className="create-run__panel-note">add a component to start building the prompt.</div>
          ) : (
            <div className="playground-outline__list">
              {normalizedComponents.map((component, index) => {
                const componentKey = getPromptComponentKey(component, index);
                const role = component.message_role ?? getDefaultPromptMessageRole(component.type);
                const displayName = getPromptComponentDisplayName(component);

                return (
                  <button
                    key={componentKey}
                    type="button"
                    className={`playground-outline__row${!component.enabled ? ' is-disabled' : ''}`}
                    onClick={() => onJumpToSection(componentKey)}
                    aria-label={`jump to section ${index + 1}: ${displayName}`}
                  >
                    <span className="playground-outline__row-main">
                      <span className="playground-outline__row-index">{index + 1}</span>
                      <span className="playground-outline__row-title">{displayName}</span>
                    </span>
                    <span className="playground-outline__row-meta">
                      <span>{ROLE_LABELS[role]}</span>
                      <span>{component.content.length} chars</span>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="playground-outline__section">
          <div className="playground-outline__section-head">
            <span className="playground-outline__section-label">
              <span
                className="playground-outline__section-icon"
                style={getMaskIconStyle(iconTool)}
                aria-hidden
              />
              <span className="section-label">Tools</span>
            </span>
          </div>
          {toolNames.length === 0 ? (
            <div className="playground-outline__empty">no tools attached</div>
          ) : (
            <div className="playground-outline__token-list">
              {previewToolNames.map((toolName, index) => (
                <span key={`${toolName}:${index}`} className="playground-outline__token">
                  {toolName}
                </span>
              ))}
              {toolNames.length > previewToolNames.length && (
                <span className="playground-outline__more">
                  +{toolNames.length - previewToolNames.length} more
                </span>
              )}
            </div>
          )}
        </div>

        <div className="playground-outline__section">
          <div className="playground-outline__section-head">
            <span className="playground-outline__section-label">
              <span
                className="playground-outline__section-icon"
                style={getMaskIconStyle(iconOutputSchema)}
                aria-hidden
              />
              <span className="section-label">Output schema</span>
            </span>
          </div>
          {!schemaEnabled ? (
            <div className="playground-outline__empty">schema off</div>
          ) : schemaError ? (
            <div className="playground-outline__empty">schema needs attention</div>
          ) : schemaFields.length === 0 ? (
            <div className="playground-outline__empty">no schema fields</div>
          ) : (
            <div className="playground-outline__detail-list">
              {previewSchemaFields.map((field) => (
                <div key={field.name} className="playground-outline__detail-row">
                  <span className="playground-outline__detail-main">
                    <code className="playground-outline__detail-name">{field.name}</code>
                  </span>
                  <span className="playground-outline__detail-meta">{field.typeLabel}</span>
                </div>
              ))}
              {schemaFields.length > previewSchemaFields.length && (
                <div className="playground-outline__more">
                  +{schemaFields.length - previewSchemaFields.length} more fields
                </div>
              )}
            </div>
          )}
        </div>

        <div className="playground-outline__section">
          <div className="playground-outline__section-head">
            <span className="playground-outline__section-label">
              <span
                className="playground-outline__section-icon"
                style={getMaskIconStyle(iconVariable)}
                aria-hidden
              />
              <span className="section-label">Variables</span>
            </span>
          </div>
          {variableStates.length === 0 ? (
            <div className="playground-outline__empty">no variables detected</div>
          ) : (
            <div className="playground-outline__detail-list">
              {variableStates.map((variable) => (
                <button
                  key={variable.name}
                  type="button"
                  className="playground-outline__detail-row playground-outline__detail-row--interactive"
                  onClick={() => onJumpToVariable(variable.name)}
                  aria-label={`highlight ${variable.name} in the editor`}
                >
                  <span className="playground-outline__detail-main">
                    <code className="playground-outline__detail-name">{`{{${variable.name}}}`}</code>
                  </span>
                  <span className={`playground-outline__status${variable.isFilled ? ' is-filled' : ''}`}>
                    {variable.isFilled ? 'filled' : 'missing'}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromptStructureOutline;
