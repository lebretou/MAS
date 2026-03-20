import React from 'react';
import type {
  Prompt,
  PromptComponent,
  PromptListItem,
  PromptTool,
  PromptVersion,
  PromptWithVersions,
  SchemaProperty,
} from '../../../types/prompt';
import SchemaBuilder, {
  createSchemaProperty,
  getSchemaValidationError,
  toJsonSchema,
} from './SchemaBuilder';
import PromptComponentEditor from './PromptComponentEditor';
import PromptToolsEditor from './PromptToolsEditor';
import PromptVersionTree from './PromptVersionTree';
import GuidedPromptStart from './GuidedPromptStart';
import { useRunExecution } from '../../../hooks/useRunExecution';
import type { PlaygroundRun } from '../../../types/playground';
import iconModelConfig from '../../../assets/icon-modelconfig.svg';
import iconTool from '../../../assets/icon-tool.svg';
import iconOutputSchema from '../../../assets/icon-outputschema.svg';
import iconVariable from '../../../assets/icon-variable.svg';
import { promptAPI } from '../../../services/api';

interface Props {
  onRunComplete: (results: Array<PlaygroundRun | null>, errors: Array<string | null>) => void;
  anchorOutput: string;
  anchorLabel: string | null;
  onAnchorChange: (value: string) => void;
  onClearAnchor: () => void;
}

const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini'],
  anthropic: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
};

const DEFAULT_PROMPT_COMPONENTS: PromptComponent[] = [
  {
    type: 'role',
    content: 'You are a precise data extraction assistant.',
    enabled: true,
  },
  {
    type: 'task',
    content: [
      'Read the release brief in {{release_brief}} and extract the requested fields.',
      'Use these rules exactly:',
      '- product_name: trimmed text after "Product:"',
      '- owner: trimmed text after "Owner:"',
      '- launch_date: trimmed text after "Launch date:"',
      '- approval_required: true only if the approval line says "yes"; otherwise false',
      '- blocker_count: 0 if blockers says "none"; otherwise count the semicolon-separated blockers',
      '- is_blocked: true if blocker_count is greater than 0, otherwise false',
      '- is_high_risk: true if approval_required is true or blocker_count is 2 or more; otherwise false',
      'Do not infer extra information and do not rewrite extracted values.',
    ].join('\n'),
    enabled: true,
  },
  {
    type: 'outputs',
    content: 'Return valid JSON only and follow the schema exactly.',
    enabled: true,
  },
];

const DEFAULT_INPUT_VARS: Record<string, string> = {
  release_brief: [
    'Product: smart meeting recap',
    'Owner: maya chen',
    'Launch date: 2026-04-15',
    'Approval: yes',
    'Blockers: api latency spike; missing mobile qa signoff',
  ].join('\n'),
};

const TOOL_NAME_REGEX = /^[A-Za-z0-9_-]{1,64}$/;
const TOOL_ARGUMENT_NAME_REGEX = /^[A-Za-z_][A-Za-z0-9_]*$/;

type WorkspacePanel = 'guided' | 'model' | 'variables' | 'tools' | 'schema' | 'anchor';

interface LoadedPromptContext {
  prompt: Prompt;
  version: PromptVersion;
  executeSignature: string;
  saveSignature: string;
}

interface CompareTarget {
  promptId: string;
  promptName: string;
  version: PromptVersion;
}

const PANEL_TITLES: Record<WorkspacePanel, string> = {
  guided: 'Guided prompt start',
  model: 'Model configuration',
  variables: 'Input variables',
  tools: 'Tools',
  schema: 'Output schema',
  anchor: 'Anchor output',
};

function createDefaultSchemaProperties(): SchemaProperty[] {
  return [
    {
      ...createSchemaProperty(),
      name: 'product_name',
      type: 'string',
      description: 'exact product name from the brief',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'owner',
      type: 'string',
      description: 'exact owner name from the brief',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'launch_date',
      type: 'string',
      description: 'exact launch date from the brief',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'approval_required',
      type: 'boolean',
      description: 'true when approval is yes',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'blocker_count',
      type: 'integer',
      description: 'number of blockers listed in the brief',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'is_blocked',
      type: 'boolean',
      description: 'true when blocker_count > 0',
      required: true,
    },
    {
      ...createSchemaProperty(),
      name: 'is_high_risk',
      type: 'boolean',
      description: 'true when approval is yes or blocker_count >= 2',
      required: true,
    },
  ];
}

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(',')}]`;
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, nestedValue]) => `${JSON.stringify(key)}:${stableSerialize(nestedValue)}`);

    return `{${entries.join(',')}}`;
  }

  return JSON.stringify(value);
}

function getExecutionPromptSignature(
  components: PromptComponent[],
  tools: PromptTool[],
  outputSchema: Record<string, unknown> | null,
): string {
  return stableSerialize({
    components,
    tools,
    outputSchema,
  });
}

function getSavedPromptSignature(
  components: PromptComponent[],
  variables: Record<string, string>,
  tools: PromptTool[],
  outputSchema: Record<string, unknown> | null,
): string {
  return stableSerialize({
    components,
    variables,
    tools,
    outputSchema,
  });
}

function collectPromptVariables(components: PromptComponent[]): string[] {
  const variableNames = new Set<string>();

  components.forEach((component) => {
    const matches = component.content.matchAll(/\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g);
    for (const match of matches) {
      const variableName = match[1]?.trim();
      if (variableName) {
        variableNames.add(variableName);
      }
    }
  });

  return Array.from(variableNames);
}

const PromptForm: React.FC<Props> = ({
  onRunComplete,
  anchorOutput,
  anchorLabel,
  onAnchorChange,
  onClearAnchor,
}) => {
  const triggerRefs = React.useRef<Record<WorkspacePanel, HTMLButtonElement | null>>({
    guided: null,
    model: null,
    variables: null,
    tools: null,
    schema: null,
    anchor: null,
  });
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null);
  const inputVarRefs = React.useRef<Record<string, HTMLTextAreaElement | null>>({});
  const anchorOutputRef = React.useRef<HTMLTextAreaElement | null>(null);
  const [promptComponents, setPromptComponents] = React.useState<PromptComponent[]>(DEFAULT_PROMPT_COMPONENTS);
  const [inputVars, setInputVars] = React.useState<Record<string, string>>(DEFAULT_INPUT_VARS);
  const [provider, setProvider] = React.useState('openai');
  const [model, setModel] = React.useState('gpt-4o');
  const [temperature, setTemperature] = React.useState(0);
  const [numRuns, setNumRuns] = React.useState(1);
  const [activePanel, setActivePanel] = React.useState<WorkspacePanel | null>(null);
  const [loadedPromptContext, setLoadedPromptContext] = React.useState<LoadedPromptContext | null>(null);
  const [savedPrompts, setSavedPrompts] = React.useState<PromptListItem[]>([]);
  const [savedPromptsLoading, setSavedPromptsLoading] = React.useState(true);
  const [savedPromptsError, setSavedPromptsError] = React.useState<string | null>(null);
  const [selectedPromptId, setSelectedPromptId] = React.useState('');
  const [selectedPromptData, setSelectedPromptData] = React.useState<PromptWithVersions | null>(null);
  const [selectedPromptLoading, setSelectedPromptLoading] = React.useState(false);
  const [comparisonTargets, setComparisonTargets] = React.useState<CompareTarget[]>([]);
  const [promptIdInput, setPromptIdInput] = React.useState('');
  const [promptNameInput, setPromptNameInput] = React.useState('');
  const [saveError, setSaveError] = React.useState<string | null>(null);
  const [saveMessage, setSaveMessage] = React.useState<string | null>(null);

  const [schemaEnabled, setSchemaEnabled] = React.useState(true);
  const [schemaProperties, setSchemaProperties] = React.useState<SchemaProperty[]>(() => createDefaultSchemaProperties());
  const [tools, setTools] = React.useState<PromptTool[]>([]);

  const [toolError, setToolError] = React.useState<string | null>(null);
  const [revisionNote, setRevisionNote] = React.useState('');
  const [appliedTemplateId, setAppliedTemplateId] = React.useState<string | null>(null);

  const {
    loading, setupError, execute,
  } = useRunExecution(onRunComplete);

  const refreshPromptList = React.useCallback(async () => {
    const prompts = await promptAPI.getAllPrompts();
    setSavedPrompts(prompts);
    setSelectedPromptId((current) => current || prompts[0]?.prompt_id || '');
    return prompts;
  }, []);

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    const models = PROVIDER_MODELS[newProvider];
    if (models && models.length > 0) {
      setModel(models[0]);
    }
  };

  const handleSchemaToggle = (enabled: boolean) => {
    setSchemaEnabled(enabled);

    if (enabled && schemaProperties.length === 0) {
      setSchemaProperties(createDefaultSchemaProperties());
    }
  };

  const schemaError = schemaEnabled
    ? getSchemaValidationError(schemaProperties)
    : null;
  const activePanelId = activePanel ? `playground-config-panel-${activePanel}` : undefined;
  const activePromptComponents = React.useMemo(
    () => promptComponents.filter((component) => component.enabled && component.content.trim()),
    [promptComponents]
  );
  const detectedVariables = React.useMemo(
    () => collectPromptVariables(activePromptComponents),
    [activePromptComponents]
  );

  React.useEffect(() => {
    if (!activePanel) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setActivePanel(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activePanel]);

  React.useEffect(() => {
    const textareaElements = [
      ...detectedVariables.map((variableName) => inputVarRefs.current[variableName]),
      anchorOutputRef.current,
    ];
    textareaElements.forEach((textarea) => {
      if (!textarea) return;
      textarea.style.height = '0px';
      textarea.style.height = `${textarea.scrollHeight}px`;
    });
  }, [detectedVariables, inputVars, anchorOutput]);

  React.useEffect(() => {
    if (!activePanel) {
      return undefined;
    }

    closeButtonRef.current?.focus();

    return () => {
      triggerRefs.current[activePanel]?.focus();
    };
  }, [activePanel]);

  React.useEffect(() => {
    refreshPromptList()
      .catch(() => setSavedPromptsError('Failed to load saved prompts.'))
      .finally(() => setSavedPromptsLoading(false));
  }, [refreshPromptList]);

  React.useEffect(() => {
    if (!selectedPromptId) {
      setSelectedPromptData(null);
      return;
    }

    setComparisonTargets([]);

    setSelectedPromptLoading(true);
    promptAPI.getPrompt(selectedPromptId)
      .then(setSelectedPromptData)
      .catch(() => setSavedPromptsError('Failed to load prompt versions.'))
      .finally(() => setSelectedPromptLoading(false));
  }, [selectedPromptId]);

  const normalizedTools = React.useMemo(() => {
    return tools
      .map((tool) => ({
        ...tool,
        name: tool.name.trim(),
        description: tool.description.trim(),
        arguments: tool.arguments.map((argument) => ({
          ...argument,
          name: argument.name.trim(),
          description: argument.description?.trim() ?? '',
          allowed_values: argument.allowed_values?.filter(Boolean) ?? null,
        })),
      }))
      .filter((tool) => tool.name || tool.description || tool.arguments.length > 0);
  }, [tools]);

  const loadVersionIntoEditor = React.useCallback((prompt: Prompt, version: PromptVersion) => {
    setPromptComponents(version.components);
    setTools(version.tools ?? []);
    setSchemaEnabled(Boolean(version.output_schema));
    setSchemaProperties(
      version.output_schema && typeof version.output_schema.properties === 'object'
        ? Object.entries(version.output_schema.properties as Record<string, Record<string, unknown>>).map(([name, rawSchema]) => {
            const schema = rawSchema as Record<string, unknown> & {
              type?: SchemaProperty['type'];
              description?: string;
              items?: { type?: SchemaProperty['items'] };
            };
            return ({
            ...createSchemaProperty(),
            name,
            type: schema.type ?? 'string',
            description: typeof schema.description === 'string' ? schema.description : '',
            required: Array.isArray(version.output_schema?.required)
              ? version.output_schema.required.includes(name)
              : false,
            items: schema.items && typeof schema.items === 'object'
              ? schema.items.type ?? undefined
              : undefined,
          });
        })
        : createDefaultSchemaProperties()
    );
    setInputVars(version.variables ?? {});
    setRevisionNote('');
    setAppliedTemplateId(version.source_template_id ?? null);
    setPromptIdInput(prompt.prompt_id);
    setPromptNameInput(prompt.name);
    setSaveError(null);
    setSaveMessage(null);
    setLoadedPromptContext({
      prompt,
      version,
      executeSignature: getExecutionPromptSignature(
        version.components.filter((component) => component.enabled && component.content.trim()),
        version.tools ?? [],
        (version.output_schema as Record<string, unknown> | null) ?? null
      ),
      saveSignature: getSavedPromptSignature(
        version.components,
        version.variables ?? {},
        version.tools ?? [],
        (version.output_schema as Record<string, unknown> | null) ?? null
      ),
    });
  }, []);

  const toggleCompareTarget = React.useCallback((promptName: string, version: PromptVersion) => {
    setComparisonTargets((current) => {
      const exists = current.find((target) => (
        target.promptId === version.prompt_id && target.version.version_id === version.version_id
      ));
      if (exists) {
        return current.filter((target) => !(
          target.promptId === version.prompt_id && target.version.version_id === version.version_id
        ));
      }
      if (current.length === 2) {
        return [
          current[1],
          {
            promptId: version.prompt_id,
            promptName,
            version,
          },
        ];
      }
      return [
        ...current,
        {
          promptId: version.prompt_id,
          promptName,
          version,
        },
      ];
    });
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setToolError(null);

    if (schemaError) {
      return;
    }

    const emptyTool = normalizedTools.find((tool) => !tool.name || !tool.description);
    if (emptyTool) {
      setToolError('Each tool needs a name and description.');
      return;
    }

    const duplicateToolNames = new Set<string>();
    for (const tool of normalizedTools) {
      if (tool.name === 'structured_output') {
        setToolError('Tool name "structured_output" is reserved.');
        return;
      }
      if (!TOOL_NAME_REGEX.test(tool.name)) {
        setToolError(`Tool "${tool.name}" must use only letters, numbers, underscores, or hyphens.`);
        return;
      }
      if (duplicateToolNames.has(tool.name)) {
        setToolError('Tool names must be unique.');
        return;
      }
      duplicateToolNames.add(tool.name);

      const emptyArgument = tool.arguments.find((argument) => !argument.name);
      if (emptyArgument) {
        setToolError(`Tool "${tool.name}" has an argument without a name.`);
        return;
      }

      const argumentNames = new Set<string>();
      for (const argument of tool.arguments) {
        if (!TOOL_ARGUMENT_NAME_REGEX.test(argument.name)) {
          setToolError(`Argument "${argument.name}" in tool "${tool.name}" must start with a letter or underscore.`);
          return;
        }
        if (argumentNames.has(argument.name)) {
          setToolError(`Tool "${tool.name}" has duplicate argument names.`);
          return;
        }
        argumentNames.add(argument.name);
      }
    }

    const submittedInputVariables = Object.fromEntries(
      detectedVariables.map((variableName) => [variableName, inputVars[variableName] ?? ''])
    );
    const outputSchema =
      schemaEnabled && !schemaError
        ? toJsonSchema(schemaProperties)
        : null;

    const executionResult = await execute({
      components: activePromptComponents,
      tools: normalizedTools,
      inputVariables: submittedInputVariables,
      model,
      provider,
      temperature,
      numRuns,
      outputSchema,
      promptContext: {
        promptId: loadedPromptContext?.prompt.prompt_id ?? null,
        promptName: loadedPromptContext?.prompt.name ?? 'Playground Prompt',
        versionId: loadedPromptContext?.version.version_id ?? null,
        branchName: loadedPromptContext?.version.branch_name ?? null,
        loadedSignature: loadedPromptContext?.executeSignature ?? null,
        revisionNote: revisionNote.trim() || null,
        sourceTemplateId: appliedTemplateId,
      },
    });

    if (!executionResult) {
      return;
    }

    await refreshPromptList();
    if (executionResult.promptId) {
      setSelectedPromptId(executionResult.promptId);
      const promptData = await promptAPI.getPrompt(executionResult.promptId);
      setSelectedPromptData(promptData);
      const executedVersion = promptData.versions.find((version) => version.version_id === executionResult.versionId);
      if (executedVersion) {
        loadVersionIntoEditor(promptData.prompt, executedVersion);
      }
    }
  };

  const handleSavePrompt = async () => {
    setSaveError(null);
    setSaveMessage(null);

    const nextPromptId = promptIdInput.trim();
    const nextPromptName = promptNameInput.trim();
    if (!nextPromptId || !nextPromptName) {
      setSaveError('Prompt ID and prompt name are required to save.');
      return;
    }

    if (schemaError) {
      setSaveError(schemaError);
      return;
    }

    try {
      const outputSchema =
        schemaEnabled && !schemaError
          ? toJsonSchema(schemaProperties)
          : null;
      const savedInputVariables = Object.fromEntries(
        collectPromptVariables(promptComponents).map((variableName) => [variableName, inputVars[variableName] ?? ''])
      );
      const saveSignature = getSavedPromptSignature(
        promptComponents,
        savedInputVariables,
        normalizedTools,
        outputSchema
      );

      if (
        loadedPromptContext &&
        loadedPromptContext.prompt.prompt_id === nextPromptId &&
        loadedPromptContext.saveSignature === saveSignature
      ) {
        if (loadedPromptContext.prompt.name !== nextPromptName) {
          await promptAPI.updatePrompt(nextPromptId, {
            name: nextPromptName,
            description: loadedPromptContext.prompt.description ?? 'Saved from playground',
          });
        }
        setSaveMessage(`Already saved as ${nextPromptId}/${loadedPromptContext.version.version_id}.`);
        await refreshPromptList();
        return;
      }

      const existingPromptData = await promptAPI.getPrompt(nextPromptId).catch(() => null);
      if (!existingPromptData) {
        await promptAPI.createPrompt({
          prompt_id: nextPromptId,
          name: nextPromptName,
          description: 'Saved from playground',
        });
      } else if (existingPromptData.prompt.name !== nextPromptName) {
        await promptAPI.updatePrompt(nextPromptId, {
          name: nextPromptName,
          description: existingPromptData.prompt.description ?? 'Saved from playground',
        });
      }

      const latestPromptData = existingPromptData ?? await promptAPI.getPrompt(nextPromptId);
      const parentVersionId =
        loadedPromptContext?.prompt.prompt_id === nextPromptId
          ? loadedPromptContext.version.version_id
          : latestPromptData.prompt.latest_version_id ?? null;

      const createdVersion = await promptAPI.createVersion(nextPromptId, {
        name: revisionNote.trim() || `${nextPromptName} version`,
        components: promptComponents,
        variables: savedInputVariables,
        output_schema: outputSchema,
        tools: normalizedTools,
        parent_version_id: parentVersionId,
        branch_name: loadedPromptContext?.prompt.prompt_id === nextPromptId
          ? loadedPromptContext.version.branch_name ?? undefined
          : undefined,
        revision_note: revisionNote.trim() || undefined,
        source_template_id: appliedTemplateId ?? undefined,
      });

      const promptData = await promptAPI.getPrompt(nextPromptId);
      setSelectedPromptId(nextPromptId);
      setSelectedPromptData(promptData);
      setPromptIdInput(nextPromptId);
      setPromptNameInput(nextPromptName);
      const savedVersion = promptData.versions.find((version) => version.version_id === createdVersion.version_id);
      if (savedVersion) {
        loadVersionIntoEditor(promptData.prompt, savedVersion);
      }
      await refreshPromptList();
      setSaveMessage(`Saved ${nextPromptId}/${createdVersion.version_id}.`);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save prompt.');
    }
  };

  return (
    <div>
      <div className="card">
        <div className="card__body">
          <form onSubmit={handleSubmit} className="flex-col create-run__form">
            <div className="create-run__workspace">
              <div className="create-run__toolbar-stack">
                <div className="create-run__workspace-header">
                  <div className="create-run__workspace-copy">
                    <label className="field__label">Prompt Workspace</label>
                    <span className="field__hint">
                      shape the prompt first, then open model, tools, or schema only when you need them.
                    </span>
                  </div>

                  <div className="create-run__workspace-toolbar">
                    <button
                      type="button"
                      ref={(element) => {
                        triggerRefs.current.guided = element;
                      }}
                      className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'guided' ? ' is-active' : ''}`}
                      onClick={() => setActivePanel((current) => (current === 'guided' ? null : 'guided'))}
                      aria-pressed={activePanel === 'guided'}
                      aria-expanded={activePanel === 'guided'}
                      aria-controls={activePanel === 'guided' ? activePanelId : undefined}
                    >
                      Guided start
                    </button>
                    <button
                      type="button"
                      ref={(element) => {
                        triggerRefs.current.model = element;
                      }}
                      className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'model' ? ' is-active' : ''}`}
                      onClick={() => setActivePanel((current) => (current === 'model' ? null : 'model'))}
                      aria-pressed={activePanel === 'model'}
                      aria-expanded={activePanel === 'model'}
                      aria-controls={activePanel === 'model' ? activePanelId : undefined}
                    >
                      <img src={iconModelConfig} alt="" className="create-run__panel-btn-icon" aria-hidden />
                      Model config
                    </button>
                    <button
                      type="button"
                      ref={(element) => {
                        triggerRefs.current.variables = element;
                      }}
                      className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'variables' ? ' is-active' : ''}`}
                      onClick={() => setActivePanel((current) => (current === 'variables' ? null : 'variables'))}
                      aria-pressed={activePanel === 'variables'}
                      aria-expanded={activePanel === 'variables'}
                      aria-controls={activePanel === 'variables' ? activePanelId : undefined}
                    >
                      <img src={iconVariable} alt="" className="create-run__panel-btn-icon" aria-hidden />
                      Variables ({detectedVariables.length})
                    </button>
                    <button
                      type="button"
                      ref={(element) => {
                        triggerRefs.current.tools = element;
                      }}
                      className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'tools' ? ' is-active' : ''}`}
                      onClick={() => setActivePanel((current) => (current === 'tools' ? null : 'tools'))}
                      aria-pressed={activePanel === 'tools'}
                      aria-expanded={activePanel === 'tools'}
                      aria-controls={activePanel === 'tools' ? activePanelId : undefined}
                    >
                      <img src={iconTool} alt="" className="create-run__panel-btn-icon" aria-hidden />
                      Tools ({normalizedTools.length})
                    </button>
                    <button
                      type="button"
                      ref={(element) => {
                        triggerRefs.current.schema = element;
                      }}
                      className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'schema' ? ' is-active' : ''}`}
                      onClick={() => setActivePanel((current) => (current === 'schema' ? null : 'schema'))}
                      aria-pressed={activePanel === 'schema'}
                      aria-expanded={activePanel === 'schema'}
                      aria-controls={activePanel === 'schema' ? activePanelId : undefined}
                    >
                      <img src={iconOutputSchema} alt="" className="create-run__panel-btn-icon" aria-hidden />
                      Output schema ({schemaEnabled ? schemaProperties.length : 0})
                    </button>
                  </div>
                </div>

                {schemaError && activePanel !== 'schema' && (
                  <div className="alert alert--warning create-run__config-alert">
                    <span className="alert__icon">!</span>
                    Output schema needs attention: {schemaError}
                  </div>
                )}
                {activePanel && (
                  <div className="create-run__config-overlay" role="presentation">
                    <div
                      className="create-run__config-backdrop"
                      onClick={() => setActivePanel(null)}
                    />
                    <div
                      id={activePanelId}
                      className="create-run__config-popover"
                      role="dialog"
                      aria-modal="true"
                      aria-label={activePanel ? PANEL_TITLES[activePanel] : 'Workspace panel'}
                    >
                      <div className="create-run__config-popover-head">
                        <div className="create-run__panel-title">{activePanel ? PANEL_TITLES[activePanel] : ''}</div>
                        <button
                          type="button"
                          ref={closeButtonRef}
                          className="icon-btn icon-btn--close"
                          aria-label={`close ${activePanel} configuration`}
                          onClick={() => setActivePanel(null)}
                        >
                          &times;
                        </button>
                      </div>

                      {activePanel === 'guided' && (
                        <GuidedPromptStart
                          onApply={({ templateId, templateName, components, tools: templateTools, outputSchema }) => {
                            setPromptComponents(components);
                            setTools(templateTools ?? []);
                            setAppliedTemplateId(templateId);
                            setLoadedPromptContext(null);
                            setRevisionNote('');
                            setPromptNameInput((current) => current || templateName);
                            setPromptIdInput((current) => current || templateId);
                            if (outputSchema && typeof outputSchema === 'object' && 'properties' in outputSchema) {
                              setSchemaEnabled(true);
                              setSchemaProperties(
                                Object.entries(outputSchema.properties as Record<string, Record<string, unknown>>).map(([name, rawSchema]) => {
                                  const schema = rawSchema as Record<string, unknown> & {
                                    type?: SchemaProperty['type'];
                                    description?: string;
                                    items?: { type?: SchemaProperty['items'] };
                                  };
                                  return ({
                                  ...createSchemaProperty(),
                                  name,
                                  type: schema.type ?? 'string',
                                  description: typeof schema.description === 'string' ? schema.description : '',
                                  required: Array.isArray((outputSchema as { required?: string[] }).required)
                                    ? ((outputSchema as { required?: string[] }).required ?? []).includes(name)
                                    : false,
                                  items: schema.items && typeof schema.items === 'object'
                                    ? schema.items.type ?? undefined
                                    : undefined,
                                });
                              })
                              );
                            } else {
                              setSchemaEnabled(false);
                            }
                            setActivePanel(null);
                          }}
                        />
                      )}

                      {activePanel === 'model' && (
                        <>
                          <div className="field__hint">
                            keep execution settings nearby without interrupting prompt drafting.
                          </div>

                          <div className="form-grid">
                            <div className="field">
                              <label className="field__label" htmlFor="playground-provider">
                                Provider
                              </label>
                              <select
                                id="playground-provider"
                                className="select"
                                value={provider}
                                onChange={(e) => handleProviderChange(e.target.value)}
                              >
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                              </select>
                            </div>

                            <div className="field">
                              <label className="field__label" htmlFor="playground-model">
                                Model
                              </label>
                              <select
                                id="playground-model"
                                className="select"
                                value={model}
                                onChange={(e) => setModel(e.target.value)}
                              >
                                {(PROVIDER_MODELS[provider] ?? []).map((availableModel) => (
                                  <option key={availableModel} value={availableModel}>
                                    {availableModel}
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>

                          <div className="form-grid">
                            <div className="field">
                              <label className="field__label" htmlFor="playground-temperature">
                                Temperature: {temperature}
                              </label>
                              <input
                                id="playground-temperature"
                                className="input"
                                type="range"
                                min="0"
                                max="2"
                                step="0.1"
                                value={temperature}
                                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                              />
                            </div>

                            <div className="field">
                              <label className="field__label" htmlFor="playground-num-runs">
                                Number of Runs (max 10)
                              </label>
                              <input
                                id="playground-num-runs"
                                className="input"
                                type="number"
                                min="1"
                                max="10"
                                value={numRuns}
                                onChange={(e) =>
                                  setNumRuns(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))
                                }
                              />
                            </div>
                          </div>
                        </>
                      )}

                      {activePanel === 'variables' && (
                        <>
                          <div className="field__hint">
                            rows appear automatically when the prompt includes {'{{variable_name}}'} placeholders.
                          </div>

                          {detectedVariables.length > 0 ? (
                            <div className="create-run__variables-table">
                              <div className="create-run__variables-head">
                                <span>Variable</span>
                                <span>Value</span>
                              </div>
                              {detectedVariables.map((variableName) => (
                                <div key={variableName} className="create-run__variables-row">
                                  <div className="create-run__variables-name">
                                    <code>{`{{${variableName}}}`}</code>
                                  </div>
                                  <textarea
                                    ref={(element) => {
                                      inputVarRefs.current[variableName] = element;
                                    }}
                                    className="textarea textarea--code create-run__variables-textarea"
                                    value={inputVars[variableName] ?? ''}
                                    onChange={(e) => {
                                      e.target.style.height = '0px';
                                      e.target.style.height = `${e.target.scrollHeight}px`;
                                      setInputVars((current) => ({
                                        ...current,
                                        [variableName]: e.target.value,
                                      }));
                                    }}
                                    rows={2}
                                    placeholder={`value for ${variableName}`}
                                    aria-label={`${variableName} input variable value`}
                                  />
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="create-run__panel-note">
                              add a placeholder like {'{{release_brief}}'} in the prompt and a variable row will appear here.
                            </div>
                          )}
                        </>
                      )}

                      {activePanel === 'tools' && (
                        <>
                          <div className="field__hint">
                            define the callable surface without pushing it to the bottom of the form.
                          </div>

                          <PromptToolsEditor
                            tools={tools}
                            onChange={(nextTools) => {
                              setTools(nextTools);
                              setToolError(null);
                            }}
                          />
                          {schemaEnabled && normalizedTools.length > 0 && (
                            <span className="field__hint">
                              when tools are attached, the full schema stays in the prompt text, but provider-side schema enforcement is disabled.
                            </span>
                          )}
                          {toolError && <span className="field__error">{toolError}</span>}
                        </>
                      )}

                      {activePanel === 'schema' && (
                        <>
                          <div className="field__hint">
                            describe the output shape explicitly when you want structured consistency checks.
                          </div>

                          <label className="check-label">
                            <input
                              type="checkbox"
                              checked={schemaEnabled}
                              onChange={(e) => handleSchemaToggle(e.target.checked)}
                            />
                            Enable output schema
                          </label>

                          {schemaEnabled ? (
                            <>
                              <SchemaBuilder
                                properties={schemaProperties}
                                onChange={setSchemaProperties}
                              />
                              {schemaError && <span className="field__error">{schemaError}</span>}
                            </>
                          ) : (
                            <div className="create-run__panel-note">
                              turn this on when you want provider-side schema guidance and field-level deviation analysis.
                            </div>
                          )}
                        </>
                      )}

                      {activePanel === 'anchor' && (
                        <>
                          <div className="create-run__anchor-header">
                            <div className="field__hint">
                              store an example output here when you want to compare future runs against it.
                            </div>
                            {anchorOutput.trim() && (
                              <button
                                type="button"
                                className="btn btn--ghost btn--sm"
                                onClick={onClearAnchor}
                              >
                                Clear anchor
                              </button>
                            )}
                          </div>

                          <textarea
                            ref={anchorOutputRef}
                            id="anchor-output"
                            className="textarea textarea--code"
                            value={anchorOutput}
                            onChange={(e) => {
                              e.target.style.height = '0px';
                              e.target.style.height = `${e.target.scrollHeight}px`;
                              onAnchorChange(e.target.value);
                            }}
                            rows={8}
                            placeholder="optional example output to use as an anchor in the visualization"
                          />
                          <span className="field__hint">
                            {anchorLabel
                              ? `active reference: ${anchorLabel}. edit this field to replace it.`
                              : 'paste an example output here to compare future runs against it.'}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="create-run__workspace-body">
                <div className="form-grid create-run__prompt-meta-grid">
                  <div className="field">
                    <label className="field__label" htmlFor="playground-prompt-id">
                      Prompt ID
                    </label>
                    <input
                      id="playground-prompt-id"
                      className="input"
                      value={promptIdInput}
                      onChange={(e) => setPromptIdInput(e.target.value)}
                      placeholder="planner-prompt"
                    />
                  </div>
                  <div className="field">
                    <label className="field__label" htmlFor="playground-prompt-name">
                      Prompt name
                    </label>
                    <input
                      id="playground-prompt-name"
                      className="input"
                      value={promptNameInput}
                      onChange={(e) => setPromptNameInput(e.target.value)}
                      placeholder="Planner agent prompt"
                    />
                  </div>
                </div>

                <div className="field">
                  <div className="create-run__field-head">
                    <label className="field__label">Prompt Components</label>
                    <span className="field__hint">
                      {promptComponents.filter((component) => component.enabled && component.content.trim()).length} active
                    </span>
                  </div>
                  <span className="field__hint create-run__component-hint">
                    use {'{{variable_name}}'} anywhere in the prompt when you want to reference input variables.
                  </span>
                  <PromptComponentEditor
                    components={promptComponents}
                    onChange={setPromptComponents}
                  />
                </div>

                <div className="field create-run__saved-versions">
                  <div className="create-run__field-head">
                    <label className="field__label">Saved Versions</label>
                    <span className="field__hint">
                      inspect saved history here without leaving the playground
                    </span>
                  </div>

                  {loadedPromptContext && (
                    <div className="create-run__active-version">
                      <div className="create-run__active-version-head">
                        <span className="badge badge--primary">
                          editing {loadedPromptContext.prompt.name} / {loadedPromptContext.version.version_id}
                        </span>
                        <button
                          type="button"
                          className="btn btn--ghost btn--sm"
                          onClick={() => {
                            setLoadedPromptContext(null);
                            setRevisionNote('');
                          }}
                        >
                          detach
                        </button>
                      </div>
                      <div className="field">
                        <label className="field__label" htmlFor="prompt-revision-note">
                          Revision note
                        </label>
                        <input
                          id="prompt-revision-note"
                          className="input"
                          value={revisionNote}
                          onChange={(e) => setRevisionNote(e.target.value)}
                          placeholder="what changed in this revision?"
                        />
                      </div>
                    </div>
                  )}

                  <div className="create-run__saved-toolbar">
                    <select
                      className="select"
                      value={selectedPromptId}
                      onChange={(e) => setSelectedPromptId(e.target.value)}
                      disabled={savedPromptsLoading || savedPrompts.length === 0}
                    >
                      <option value="">
                        {savedPromptsLoading ? 'Loading prompts...' : 'Select saved prompt'}
                      </option>
                      {savedPrompts.map((prompt) => (
                        <option key={prompt.prompt_id} value={prompt.prompt_id}>
                          {prompt.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  {savedPromptsError && <span className="field__error">{savedPromptsError}</span>}

                  {selectedPromptLoading ? (
                    <div className="create-run__panel-note">Loading saved versions...</div>
                  ) : selectedPromptData ? (
                    <>
                      <PromptVersionTree
                        promptId={selectedPromptData.prompt.prompt_id}
                        promptName={selectedPromptData.prompt.name}
                        versions={selectedPromptData.versions}
                        activeVersionId={
                          loadedPromptContext?.prompt.prompt_id === selectedPromptData.prompt.prompt_id
                            ? loadedPromptContext.version.version_id
                            : null
                        }
                        compareTargets={comparisonTargets.map((target) => ({
                          promptId: target.promptId,
                          versionId: target.version.version_id,
                        }))}
                        onLoadVersion={(version) => loadVersionIntoEditor(selectedPromptData.prompt, version)}
                        onToggleCompare={(version) => toggleCompareTarget(selectedPromptData.prompt.name, version)}
                      />
                      {comparisonTargets.length > 0 && (
                        <div className="create-run__compare-targets">
                          {comparisonTargets.map((target, index) => (
                            <span key={`${target.promptId}-${target.version.version_id}`} className="badge badge--neutral">
                              compare {index + 1}: {target.promptName} / {target.version.version_id}
                            </span>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="create-run__panel-note">
                      select a saved prompt to inspect versions and load one into the editor.
                    </div>
                  )}
                </div>
              </div>
            </div>

            <hr className="divider" />

            <div className="flex-row create-run__actions">
              <button
                type="button"
                className="btn btn--secondary"
                onClick={handleSavePrompt}
              >
                Save prompt
              </button>
              <button
                type="button"
                ref={(element) => {
                  triggerRefs.current.anchor = element;
                }}
                className={`btn btn--secondary${activePanel === 'anchor' ? ' is-active' : ''}`}
                onClick={() => setActivePanel((current) => (current === 'anchor' ? null : 'anchor'))}
                aria-pressed={activePanel === 'anchor'}
                aria-expanded={activePanel === 'anchor'}
                aria-controls={activePanel === 'anchor' ? activePanelId : undefined}
              >
                Anchor output {anchorOutput.trim() ? '(set)' : ''}
              </button>
              <button
                type="submit"
                className="btn btn--primary"
                disabled={loading || !!schemaError}
              >
                {loading && <span className="spinner create-run__spinner" />}
                {loading ? 'Executing...' : 'Execute Run'}
              </button>
            </div>

            {saveMessage && (
              <div className="alert alert--success">
                <span className="alert__icon">i</span>
                {saveMessage}
              </div>
            )}
            {saveError && (
              <div className="alert alert--danger">
                <span className="alert__icon">!</span>
                {saveError}
              </div>
            )}
          </form>
        </div>
      </div>

      {setupError && (
        <div className="alert alert--danger create-run__mt">
          <span className="alert__icon">!</span>
          {setupError}
        </div>
      )}

      {toolError && !setupError && (
        <div className="alert alert--danger create-run__mt">
          <span className="alert__icon">!</span>
          {toolError}
        </div>
      )}

    </div>
  );
};

export default PromptForm;
