import React from 'react';
import { guidedStartAPI } from '../../../services/api';
import type {
  GuidedStartArchetype,
  GuidedStartCatalog,
  GuidedStartConversationTurn,
  GuidedStartQuestion,
  GuidedStartStage,
  GuidedStartSuggestedComponent,
} from '../../../types/guidedStart';
import type { PromptComponent, PromptTool, SchemaProperty } from '../../../types/prompt';
import SchemaBuilder, {
  createSchemaProperty,
  getSchemaValidationError,
  toJsonSchema,
} from './SchemaBuilder';
import PromptToolsEditor from './PromptToolsEditor';
import { resizeTextarea } from '../../../utils/resizeTextarea';

type GuidedMode = 'deterministic' | 'ai';
type ApplyMode = 'replace' | 'merge';

interface GuidedPromptStartResult {
  sourceTemplateId: string;
  templateName: string;
  promptName: string;
  components: PromptComponent[];
  tools: PromptTool[];
  outputSchema: Record<string, unknown> | null;
  applyMode: ApplyMode;
}

interface Props {
  onApply: (result: GuidedPromptStartResult) => void;
  onClose: () => void;
  existingDraft: {
    components: PromptComponent[];
    tools: PromptTool[];
    outputSchema: Record<string, unknown> | null;
    hasUserContent: boolean;
  };
  modelConfig: {
    provider: string;
    model: string;
    temperature: number;
  };
}

function buildQuestionDefaults(questions: GuidedStartQuestion[]) {
  return Object.fromEntries(questions.map((question) => [question.question_id, question.default_value ?? '']));
}

function fillTemplateValue(template: string, values: Record<string, string>) {
  return template.replace(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g, (_, key: string) => values[key] ?? '');
}

function buildDraftFromSuggestions(
  suggestions: GuidedStartSuggestedComponent[],
  values: Record<string, string>,
) {
  return [...suggestions]
    .sort((a, b) => a.order_rank - b.order_rank)
    .map((suggestion) => ({
      type: suggestion.component_type,
      content: fillTemplateValue(suggestion.content_template, values).trim(),
      enabled: true,
    }))
    .filter((component) => component.content);
}

function formatPrevalenceText(prevalence: number, archetypeTitle: string) {
  return `used in ${Math.round(prevalence * 100)}% of ${archetypeTitle.toLowerCase()} prompts`;
}

function createSchemaPropertiesFromOutputSchema(
  outputSchema: Record<string, unknown> | null | undefined,
) {
  if (!outputSchema || typeof outputSchema.properties !== 'object') {
    return [] as SchemaProperty[];
  }

  return Object.entries(outputSchema.properties as Record<string, Record<string, unknown>>).map(([name, rawSchema]) => {
    const schema = rawSchema as Record<string, unknown> & {
      type?: SchemaProperty['type'];
      description?: string;
      items?: { type?: SchemaProperty['items'] };
    };

    return {
      ...createSchemaProperty(),
      name,
      type: schema.type ?? 'string',
      description: typeof schema.description === 'string' ? schema.description : '',
      required: Array.isArray(outputSchema.required)
        ? outputSchema.required.includes(name)
        : false,
      items: schema.items && typeof schema.items === 'object'
        ? schema.items.type ?? undefined
        : undefined,
    };
  });
}

const STAGES: GuidedStartStage[] = ['questions', 'tools', 'schema', 'review'];

const GuidedPromptStart: React.FC<Props> = ({
  onApply,
  onClose,
  existingDraft,
  modelConfig,
}) => {
  const textareaRefs = React.useRef<Record<string, HTMLTextAreaElement | null>>({});
  const [catalog, setCatalog] = React.useState<GuidedStartCatalog | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [mode, setMode] = React.useState<GuidedMode | null>(null);
  const [stage, setStage] = React.useState<GuidedStartStage>('role');
  const [selectedArchetypeId, setSelectedArchetypeId] = React.useState<string | null>(null);
  const [customRole, setCustomRole] = React.useState('');
  const [answers, setAnswers] = React.useState<Record<string, string>>({});
  const [showValidation, setShowValidation] = React.useState(false);
  const [draftComponents, setDraftComponents] = React.useState<PromptComponent[]>([]);
  const [tools, setTools] = React.useState<PromptTool[]>([]);
  const [schemaEnabled, setSchemaEnabled] = React.useState(false);
  const [schemaProperties, setSchemaProperties] = React.useState<SchemaProperty[]>([]);
  const [transcript, setTranscript] = React.useState<GuidedStartConversationTurn[]>([]);
  const [aiTurnInput, setAiTurnInput] = React.useState('');
  const [aiLoading, setAiLoading] = React.useState(false);
  const [aiError, setAiError] = React.useState<string | null>(null);
  const [updatedTypes, setUpdatedTypes] = React.useState<string[]>([]);

  React.useEffect(() => {
    guidedStartAPI.getCatalog()
      .then((response) => {
        setCatalog(response);
        setError(null);
      })
      .catch(() => setError('Failed to load guided start catalog.'))
      .finally(() => setLoading(false));
  }, []);

  React.useEffect(() => {
    Object.values(textareaRefs.current).forEach((textarea) => resizeTextarea(textarea));
  }, [answers, aiTurnInput, draftComponents, customRole]);

  const selectedArchetype = React.useMemo(
    () => catalog?.archetypes.find((item) => item.archetype_id === selectedArchetypeId) ?? null,
    [catalog, selectedArchetypeId],
  );
  const activeQuestions = React.useMemo(() => {
    if (selectedArchetype) {
      return selectedArchetype.starter_questions;
    }
    return catalog?.fallback_questions ?? [];
  }, [catalog, selectedArchetype]);
  const activeSuggestions = React.useMemo(() => {
    if (selectedArchetype) {
      return selectedArchetype.suggested_components;
    }
    return catalog?.fallback_components ?? [];
  }, [catalog, selectedArchetype]);
  const schemaError = schemaEnabled ? getSchemaValidationError(schemaProperties) : null;
  const missingRequiredFields = React.useMemo(
    () => activeQuestions.filter((question) => question.required && !(answers[question.question_id] ?? '').trim()),
    [activeQuestions, answers],
  );

  const initializeFlow = React.useCallback((
    nextMode: GuidedMode,
    archetype: GuidedStartArchetype | null,
  ) => {
    const nextQuestions = archetype?.starter_questions ?? catalog?.fallback_questions ?? [];
    const nextAnswers = buildQuestionDefaults(nextQuestions);
    const nextTools = archetype?.suggested_tools ?? [];
    const nextSchema = createSchemaPropertiesFromOutputSchema(archetype?.suggested_output_schema ?? null);

    setMode(nextMode);
    setStage('questions');
    setSelectedArchetypeId(archetype?.archetype_id ?? null);
    setCustomRole('');
    setAnswers(nextAnswers);
    setShowValidation(false);
    setDraftComponents([]);
    setTools(nextTools);
    setSchemaEnabled(nextSchema.length > 0);
    setSchemaProperties(nextSchema);
    setTranscript(nextMode === 'ai'
      ? [{
        role: 'assistant',
        content: archetype
          ? `I can help refine a ${archetype.title.toLowerCase()} scaffold. Describe the agent and I will update the working draft as we go.`
          : 'Describe the role you want to create, or pick the closest known role to start from.',
      }]
      : []);
    setAiTurnInput('');
    setAiLoading(false);
    setAiError(null);
    setUpdatedTypes([]);
  }, [catalog]);

  const refreshDraftFromAnswers = React.useCallback((nextAnswers: Record<string, string>) => {
    setDraftComponents(buildDraftFromSuggestions(activeSuggestions, {
      ...nextAnswers,
      custom_role: customRole.trim() || nextAnswers.custom_role || '',
    }));
  }, [activeSuggestions, customRole]);

  const moveToNextStage = React.useCallback(() => {
    if (stage === 'questions') {
      setStage('tools');
      return;
    }
    if (stage === 'tools') {
      setStage('schema');
      return;
    }
    if (stage === 'schema') {
      setStage('review');
    }
  }, [stage]);

  const handleDeterministicContinue = () => {
    setShowValidation(true);
    if (missingRequiredFields.length > 0) {
      return;
    }
    refreshDraftFromAnswers(answers);
    moveToNextStage();
  };

  const handleAiRespond = () => {
    if (!aiTurnInput.trim() || aiLoading) {
      return;
    }
    setAiLoading(true);
    setAiError(null);
    const nextTranscript = [...transcript, { role: 'user' as const, content: aiTurnInput.trim() }];
    const latestTurn = aiTurnInput.trim();

    guidedStartAPI.respond({
      provider: modelConfig.provider,
      model: modelConfig.model,
      temperature: modelConfig.temperature,
      stage,
      selected_archetype: selectedArchetypeId,
      custom_role: customRole.trim() || null,
      answers,
      current_draft: draftComponents,
      conversation_history: nextTranscript,
      latest_user_turn: latestTurn,
    })
      .then((response) => {
        const assistantMessage = response.follow_up_questions.length > 0
          ? `${response.assistant_message}\n\n${response.follow_up_questions.map((question) => `- ${question}`).join('\n')}`
          : response.assistant_message;
        const nextStage = response.current_stage === stage && response.stage_complete && response.status === 'ready_for_next_stage'
          ? (STAGES[STAGES.indexOf(stage) + 1] ?? response.current_stage)
          : response.current_stage;
        setTranscript([
          ...nextTranscript,
          { role: 'assistant', content: assistantMessage },
        ]);
        setAiTurnInput('');
        setDraftComponents(response.component_draft);
        setSelectedArchetypeId(response.selected_archetype ?? null);
        setUpdatedTypes(response.updated_component_types);
        setStage(nextStage);
        setAiError(null);
      })
      .catch(() => {
        setTranscript(nextTranscript);
        setAiError('The AI guide could not respond. Your current draft is still available.');
      })
      .finally(() => setAiLoading(false));
  };

  const handleApply = (applyMode: ApplyMode) => {
    const archetypeTitle = selectedArchetype?.title ?? (mode === 'ai' ? 'AI guided' : 'Custom');
    onApply({
      sourceTemplateId: `guided-${mode ?? 'deterministic'}-${selectedArchetypeId ?? 'custom'}`,
      templateName: archetypeTitle,
      promptName: `${archetypeTitle} prompt`,
      components: draftComponents,
      tools,
      outputSchema: schemaEnabled && !schemaError ? toJsonSchema(schemaProperties) : null,
      applyMode,
    });
  };

  const reviewArchetypeTitle = selectedArchetype?.title ?? 'custom role';

  if (loading) {
    return <div className="field__hint">Loading guided start...</div>;
  }

  if (error || !catalog) {
    return (
      <div className="guided-start guided-start--fallback">
        <div className="create-run__panel-note">
          {error ?? 'Guided start is unavailable right now.'}
        </div>
        <div className="guided-start__actions guided-start__actions--split">
          <button
            type="button"
            className="btn btn--secondary"
            onClick={() => {
              setLoading(true);
              guidedStartAPI.getCatalog()
                .then((response) => {
                  setCatalog(response);
                  setError(null);
                })
                .catch(() => setError('Failed to load guided start catalog.'))
                .finally(() => setLoading(false));
            }}
          >
            Try again
          </button>
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            Start from blank prompt
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="guided-start">
      {mode === null && (
        <div className="guided-start__entry">
          <div className="guided-start__intro">
            <span className="section-label">guided start</span>
            <div className="guided-start__title">Choose a role or open AI guided</div>
            <p className="guided-start__copy">
              Start with a common agent role, or use AI guided when the role is custom or still fuzzy.
              Every path lands in the same editable prompt components workspace.
            </p>
          </div>

          <div className="guided-start__role-grid">
            {catalog.archetypes.map((archetype) => (
              <button
                key={archetype.archetype_id}
                type="button"
                className="guided-start__role-card"
                onClick={() => initializeFlow('deterministic', archetype)}
              >
                <span className="guided-start__role-title">{archetype.title}</span>
                <span className="guided-start__role-summary">{archetype.summary}</span>
                <span className="guided-start__role-meta">
                  {archetype.example_jobs.slice(0, 2).join(' · ')}
                </span>
              </button>
            ))}

            <button
              type="button"
              className="guided-start__role-card guided-start__role-card--ai"
              onClick={() => initializeFlow('ai', null)}
            >
              <span className="guided-start__role-title">AI guided</span>
              <span className="guided-start__role-summary">
                Use a bounded assistant for custom roles, clarifying questions, and draft refinement.
              </span>
              <span className="guided-start__role-meta">same stages, same editable component model</span>
            </button>
          </div>
        </div>
      )}

      {mode !== null && (
        <>
          <div className="guided-start__header">
            <div className="guided-start__header-copy">
              <span className="section-label">{mode === 'ai' ? 'AI guided' : 'structured flow'}</span>
              <div className="guided-start__title">
                {selectedArchetype?.title ?? (mode === 'ai' ? 'Custom role' : 'Choose role')}
              </div>
              <p className="guided-start__copy">
                {mode === 'ai'
                  ? 'The assistant stays bounded to the current stage and updates the draft component by component.'
                  : 'Answer the minimum starter questions, then move through tools, output schema, and review.'}
              </p>
            </div>

            <div className="guided-start__stage-list" aria-label="Guided start stages">
              {STAGES.map((item, index) => (
                <button
                  key={item}
                  type="button"
                  className={`guided-start__stage-pill${stage === item ? ' is-active' : ''}`}
                  onClick={() => {
                    if (item === 'review' && draftComponents.length === 0) {
                      return;
                    }
                    if ((item === 'tools' || item === 'schema' || item === 'review') && stage === 'questions' && draftComponents.length === 0) {
                      return;
                    }
                    setStage(item);
                  }}
                >
                  <span>{index + 1}</span>
                  <span>{item === 'schema' ? 'Output schema' : item}</span>
                </button>
              ))}
            </div>
          </div>

          {stage === 'questions' && (
            <div className={`guided-start__stage guided-start__stage--${mode}`}>
              <div className="guided-start__main">
                {mode === 'deterministic' ? (
                  <>
                    <div className="guided-start__inline-roles">
                      {catalog.archetypes.map((archetype) => (
                        <button
                          key={archetype.archetype_id}
                          type="button"
                          className={`guided-start__chip${selectedArchetypeId === archetype.archetype_id ? ' is-active' : ''}`}
                          onClick={() => initializeFlow('deterministic', archetype)}
                        >
                          {archetype.title}
                        </button>
                      ))}
                    </div>

                    <div className="guided-start__fields">
                      {activeQuestions.map((question) => (
                        <div key={question.question_id} className="field">
                          <label className="field__label" htmlFor={`guided-${question.question_id}`}>
                            {question.label}
                          </label>
                          <textarea
                            id={`guided-${question.question_id}`}
                            ref={(element) => {
                              textareaRefs.current[question.question_id] = element;
                            }}
                            className="textarea textarea--adaptive guided-start__textarea"
                            value={answers[question.question_id] ?? ''}
                            rows={question.input_type === 'text' ? 2 : 4}
                            placeholder={question.placeholder ?? ''}
                            onChange={(event) => {
                              resizeTextarea(event.target);
                              setAnswers((current) => ({
                                ...current,
                                [question.question_id]: event.target.value,
                              }));
                            }}
                          />
                          {showValidation && question.required && !(answers[question.question_id] ?? '').trim() && (
                            <span className="field__error">{question.label} is required.</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="guided-start__ai-layout">
                    <div className="guided-start__ai-panel">
                      <div className="guided-start__inline-roles">
                        {catalog.archetypes.map((archetype) => (
                          <button
                            key={archetype.archetype_id}
                            type="button"
                            className={`guided-start__chip${selectedArchetypeId === archetype.archetype_id ? ' is-active' : ''}`}
                            onClick={() => {
                              setSelectedArchetypeId(archetype.archetype_id);
                            setAnswers(buildQuestionDefaults(archetype.starter_questions));
                            }}
                          >
                            {archetype.title}
                          </button>
                        ))}
                      </div>

                      <div className="field">
                        <label className="field__label" htmlFor="guided-custom-role">
                          Custom role
                        </label>
                        <textarea
                          id="guided-custom-role"
                          ref={(element) => {
                            textareaRefs.current.custom_role = element;
                          }}
                          className="textarea textarea--adaptive guided-start__textarea"
                          value={customRole}
                          rows={3}
                          placeholder="Describe the role if it does not fit the known cards."
                          onChange={(event) => {
                            resizeTextarea(event.target);
                            setCustomRole(event.target.value);
                            setAnswers((current) => ({
                              ...current,
                              custom_role: event.target.value,
                            }));
                          }}
                        />
                      </div>

                      <div className="guided-start__transcript">
                        {transcript.map((turn, index) => (
                          <div
                            key={`${turn.role}-${index}`}
                            className={`guided-start__turn guided-start__turn--${turn.role}`}
                          >
                            <span className="guided-start__turn-role">{turn.role === 'assistant' ? 'assistant' : 'you'}</span>
                            <div>{turn.content}</div>
                          </div>
                        ))}
                        {transcript.length === 0 && (
                          <div className="create-run__panel-note">
                            Tell the assistant what the agent should do and it will build a working draft in place.
                          </div>
                        )}
                      </div>

                      <div className="field">
                        <label className="field__label" htmlFor="guided-ai-turn">
                          Current turn
                        </label>
                        <textarea
                          id="guided-ai-turn"
                          ref={(element) => {
                            textareaRefs.current.ai_turn = element;
                          }}
                          className="textarea textarea--adaptive guided-start__textarea"
                          value={aiTurnInput}
                          rows={4}
                          placeholder="Describe the agent, clarify a requirement, or ask for a refinement."
                          onChange={(event) => {
                            resizeTextarea(event.target);
                            setAiTurnInput(event.target.value);
                          }}
                        />
                        {aiError && <span className="field__error">{aiError}</span>}
                      </div>
                    </div>

                    <div className="guided-start__draft-panel">
                      <div className="guided-start__draft-head">
                        <span className="field__label">Working draft</span>
                        <span className="field__hint">Visible component-by-component updates</span>
                      </div>
                      {draftComponents.length > 0 ? (
                        <div className="guided-start__review-list">
                          {draftComponents.map((component) => (
                            <div key={component.type} className="guided-start__review-card">
                              <div className="guided-start__review-head">
                                <span className="type-badge">{component.type.replace('_', ' ')}</span>
                                {updatedTypes.includes(component.type) && (
                                  <span className="badge badge--primary">updated</span>
                                )}
                              </div>
                              <div className="guided-start__review-preview">{component.content}</div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="create-run__panel-note">
                          The draft will appear here as the assistant fills in the prompt components.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="guided-start__actions guided-start__actions--split">
                <button type="button" className="btn btn--ghost" onClick={() => setMode(null)}>
                  Back to role cards
                </button>
                <div className="guided-start__action-group">
                  <button
                    type="button"
                    className="btn btn--secondary"
                    onClick={() => initializeFlow(mode, selectedArchetype)}
                  >
                    Start over
                  </button>
                  {mode === 'deterministic' ? (
                    <button type="button" className="btn btn--primary" onClick={handleDeterministicContinue}>
                      Continue
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="btn btn--secondary"
                        onClick={() => {
                          if (draftComponents.length > 0) {
                            moveToNextStage();
                          }
                        }}
                        disabled={draftComponents.length === 0}
                      >
                        Skip
                      </button>
                      <button type="button" className="btn btn--primary" onClick={handleAiRespond} disabled={aiLoading}>
                        {aiLoading ? 'Thinking...' : 'Continue'}
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {stage === 'tools' && (
            <div className="guided-start__stage">
              <div className="create-run__panel-surface">
                <div className="create-run__panel-surface-head">
                  <div className="create-run__panel-surface-copy">
                    <span className="field__label">Tools</span>
                    <span className="field__hint">
                      Define callable tools here or skip this stage if the prompt should stay tool-free.
                    </span>
                  </div>
                </div>
                <PromptToolsEditor tools={tools} onChange={setTools} />
              </div>

              <div className="guided-start__actions guided-start__actions--split">
                <button type="button" className="btn btn--ghost" onClick={() => setStage('questions')}>
                  Back
                </button>
                <div className="guided-start__action-group">
                  <button type="button" className="btn btn--secondary" onClick={moveToNextStage}>
                    Skip
                  </button>
                  <button type="button" className="btn btn--primary" onClick={moveToNextStage}>
                    Continue
                  </button>
                </div>
              </div>
            </div>
          )}

          {stage === 'schema' && (
            <div className="guided-start__stage">
              <div className="create-run__panel-surface">
                <div className="create-run__panel-surface-head">
                  <div className="create-run__panel-surface-copy">
                    <span className="field__label">Output schema</span>
                    <span className="field__hint">
                      Add a structured output contract here, or continue without one.
                    </span>
                  </div>
                  <label className="check-label create-run__panel-check">
                    <input
                      type="checkbox"
                      checked={schemaEnabled}
                      onChange={(event) => {
                        setSchemaEnabled(event.target.checked);
                        if (event.target.checked && schemaProperties.length === 0) {
                          setSchemaProperties([createSchemaProperty()]);
                        }
                      }}
                    />
                    Enable output schema
                  </label>
                </div>

                {schemaEnabled ? (
                  <>
                    <SchemaBuilder properties={schemaProperties} onChange={setSchemaProperties} />
                    {schemaError && <span className="field__error">{schemaError}</span>}
                  </>
                ) : (
                  <div className="create-run__panel-note">
                    Skip this when the prompt can return unstructured text or tool calls.
                  </div>
                )}
              </div>

              <div className="guided-start__actions guided-start__actions--split">
                <button type="button" className="btn btn--ghost" onClick={() => setStage('tools')}>
                  Back
                </button>
                <div className="guided-start__action-group">
                  <button type="button" className="btn btn--secondary" onClick={moveToNextStage}>
                    Skip
                  </button>
                  <button type="button" className="btn btn--primary" onClick={moveToNextStage} disabled={Boolean(schemaError)}>
                    Continue
                  </button>
                </div>
              </div>
            </div>
          )}

          {stage === 'review' && (
            <div className="guided-start__stage">
              <div className="guided-start__review-headline">
                <div>
                  <span className="field__label">Review</span>
                  <span className="field__hint">
                    {mode === 'ai'
                      ? 'Review the working draft, then apply it into the editor as a replace or merge.'
                      : 'Each suggested component shows prevalence only. Edit any section before applying.'}
                  </span>
                </div>
                <span className="badge badge--neutral">{reviewArchetypeTitle}</span>
              </div>

              <div className="guided-start__review-list">
                {draftComponents.map((component) => {
                  const suggestion = activeSuggestions.find((item) => item.component_type === component.type);
                  return (
                    <div key={component.type} className="guided-start__review-card">
                      <div className="guided-start__review-head">
                        <span className="type-badge">{suggestion?.title ?? component.type}</span>
                        <div className="guided-start__review-meta">
                          {suggestion && (
                            <span className="badge badge--neutral">
                              {formatPrevalenceText(suggestion.prevalence, reviewArchetypeTitle)}
                            </span>
                          )}
                          {mode === 'ai' && updatedTypes.includes(component.type) && (
                            <span className="badge badge--primary">updated this turn</span>
                          )}
                        </div>
                      </div>
                      <textarea
                        ref={(element) => {
                          textareaRefs.current[`review-${component.type}`] = element;
                        }}
                        className="textarea textarea--adaptive guided-start__review-textarea"
                        value={component.content}
                        rows={4}
                        onChange={(event) => {
                          resizeTextarea(event.target);
                          setDraftComponents((current) => current.map((item) => (
                            item.type === component.type
                              ? { ...item, content: event.target.value }
                              : item
                          )));
                        }}
                      />
                    </div>
                  );
                })}
              </div>

              {mode === 'ai' && (
                <div className="create-run__panel-surface">
                  <div className="field">
                    <label className="field__label" htmlFor="guided-review-turn">
                      Refine with AI
                    </label>
                    <textarea
                      id="guided-review-turn"
                      ref={(element) => {
                        textareaRefs.current.review_turn = element;
                      }}
                      className="textarea textarea--adaptive guided-start__textarea"
                      value={aiTurnInput}
                      rows={3}
                      placeholder="Ask the assistant to tighten wording, reduce ambiguity, or adapt the role."
                      onChange={(event) => {
                        resizeTextarea(event.target);
                        setAiTurnInput(event.target.value);
                      }}
                    />
                    {aiError && <span className="field__error">{aiError}</span>}
                  </div>
                  <div className="guided-start__actions">
                    <button type="button" className="btn btn--secondary" onClick={handleAiRespond} disabled={aiLoading || !aiTurnInput.trim()}>
                      {aiLoading ? 'Thinking...' : 'Continue'}
                    </button>
                  </div>
                </div>
              )}

              <div className="guided-start__actions guided-start__actions--split">
                <div className="guided-start__action-group">
                  <button type="button" className="btn btn--ghost" onClick={() => setStage('schema')}>
                    Back
                  </button>
                  {mode === 'deterministic' && (
                    <button
                      type="button"
                      className="btn btn--secondary"
                      onClick={() => {
                        setMode('ai');
                        setTranscript((current) => current.length > 0 ? current : [{
                          role: 'assistant',
                          content: `I can refine this ${reviewArchetypeTitle.toLowerCase()} scaffold. Tell me what should change and I will update the draft.`,
                        }]);
                        setAiError(null);
                      }}
                    >
                      Open AI guided
                    </button>
                  )}
                  <button type="button" className="btn btn--secondary" onClick={() => initializeFlow(mode, selectedArchetype)}>
                    Start over
                  </button>
                </div>

                <div className="guided-start__action-group">
                  {existingDraft.hasUserContent && (
                    <button type="button" className="btn btn--secondary" onClick={() => handleApply('merge')}>
                      Merge into current draft
                    </button>
                  )}
                  <button type="button" className="btn btn--primary" onClick={() => handleApply('replace')}>
                    {existingDraft.hasUserContent ? 'Replace guided sections' : 'Apply to editor'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default GuidedPromptStart;
