import React from 'react';
import { promptAPI } from '../../../services/api';
import type { PromptComponent, PromptTemplate } from '../../../types/prompt';
import { resizeTextarea } from '../../../utils/resizeTextarea';

interface GuidedPromptStartResult {
  templateId: string;
  templateName: string;
  components: PromptComponent[];
  tools: PromptTemplate['suggested_tools'];
  outputSchema: PromptTemplate['suggested_output_schema'];
}

interface Props {
  onApply: (result: GuidedPromptStartResult) => void;
}

function fillTemplateValue(template: string, values: Record<string, string>) {
  return template.replace(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g, (_, key: string) => {
    return values[key] ?? '';
  });
}

const GuidedPromptStart: React.FC<Props> = ({ onApply }) => {
  const textareaRefs = React.useRef<Record<string, HTMLTextAreaElement | null>>({});
  const [templates, setTemplates] = React.useState<PromptTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = React.useState<string>('');
  const [fieldValues, setFieldValues] = React.useState<Record<string, string>>({});
  const [showValidation, setShowValidation] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    promptAPI.getPromptTemplates()
      .then((loadedTemplates) => {
        setTemplates(loadedTemplates);
        const firstTemplate = loadedTemplates[0];
        if (!firstTemplate) return;
        setSelectedTemplateId(firstTemplate.template_id);
        setFieldValues(
          Object.fromEntries(firstTemplate.fields.map((field) => [field.field_id, field.default_value ?? '']))
        );
      })
      .catch(() => setError('Failed to load prompt templates.'))
      .finally(() => setLoading(false));
  }, []);

  const selectedTemplate = React.useMemo(
    () => templates.find((template) => template.template_id === selectedTemplateId) ?? null,
    [templates, selectedTemplateId]
  );
  const missingRequiredFields = React.useMemo(
    () => selectedTemplate
      ? selectedTemplate.fields.filter((field) => field.required && !fieldValues[field.field_id]?.trim())
      : [],
    [fieldValues, selectedTemplate]
  );

  React.useEffect(() => {
    Object.values(textareaRefs.current).forEach((textarea) => {
      resizeTextarea(textarea);
    });
  }, [fieldValues, selectedTemplateId]);

  const handleTemplateChange = (templateId: string) => {
    setSelectedTemplateId(templateId);
    setShowValidation(false);
    const template = templates.find((candidate) => candidate.template_id === templateId);
    if (!template) return;
    setFieldValues(
      Object.fromEntries(template.fields.map((field) => [field.field_id, field.default_value ?? '']))
    );
  };

  if (loading) {
    return <div className="field__hint">Loading templates...</div>;
  }

  if (error) {
    return <div className="field__error">{error}</div>;
  }

  if (!selectedTemplate) {
    return <div className="field__hint">No templates available yet.</div>;
  }

  return (
    <div className="guided-start">
      <div className="field">
        <label className="field__label" htmlFor="guided-template-select">
          Template
        </label>
        <select
          id="guided-template-select"
          className="select"
          value={selectedTemplate.template_id}
          onChange={(e) => handleTemplateChange(e.target.value)}
        >
          {templates.map((template) => (
            <option key={template.template_id} value={template.template_id}>
              {template.name}
            </option>
          ))}
        </select>
        {selectedTemplate.description && (
          <span className="field__hint">{selectedTemplate.description}</span>
        )}
      </div>

      <div className="guided-start__fields">
        {selectedTemplate.fields.map((field) => (
          <div key={field.field_id} className="field">
            <label className="field__label" htmlFor={`guided-${field.field_id}`}>
              {field.label}
            </label>
            <textarea
              id={`guided-${field.field_id}`}
              ref={(element) => {
                textareaRefs.current[field.field_id] = element;
              }}
              className="textarea textarea--adaptive guided-start__textarea"
              value={fieldValues[field.field_id] ?? ''}
              rows={field.input_type === 'text' ? 2 : 4}
              placeholder={field.placeholder ?? ''}
              onChange={(e) => {
                resizeTextarea(e.target);
                setFieldValues((current) => ({
                  ...current,
                  [field.field_id]: e.target.value,
                }));
              }}
            />
            {showValidation && field.required && !fieldValues[field.field_id]?.trim() && (
              <span className="field__error">{field.label} is required.</span>
            )}
            {field.description && <span className="field__hint">{field.description}</span>}
          </div>
        ))}
      </div>

      <div className="guided-start__actions">
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => {
            setShowValidation(true);
            if (missingRequiredFields.length > 0) {
              return;
            }
            onApply({
              templateId: selectedTemplate.template_id,
              templateName: selectedTemplate.name,
              components: selectedTemplate.components.map((component) => ({
                ...component,
                content: fillTemplateValue(component.content, fieldValues).trim(),
              })),
              tools: selectedTemplate.suggested_tools ?? [],
              outputSchema: selectedTemplate.suggested_output_schema ?? null,
            });
          }}
        >
          Apply template
        </button>
      </div>
    </div>
  );
};

export default GuidedPromptStart;
