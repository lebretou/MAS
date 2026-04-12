import React from 'react';
import { useSearchParams } from 'react-router-dom';
import type {
  Prompt,
  PromptComponent,
  PromptComponentType,
  PromptListItem,
  PromptTool,
  PromptVersion,
  PromptWithVersions,
  SchemaProperty,
} from '../../../types/prompt';
import type { GuidedStartRole } from '../../../types/guidedStart';
import SchemaBuilder, {
  createSchemaProperty,
  getSchemaValidationError,
  toJsonSchema,
} from './SchemaBuilder';
import PromptComponentEditor, {
  findPromptVariableMatch,
  type PromptVariableHighlightRequest,
  type PromptVariableMatch,
} from './PromptComponentEditor';
import PromptToolsEditor from './PromptToolsEditor';
import PromptVersionTree from './PromptVersionTree';
import GuidedPromptStart from './GuidedPromptStart';
import GuidedOverlay from './GuidedOverlay';
import type { GuidedOverlayStep } from './GuidedOverlay';
import VersionComparisonWorkspace from './VersionComparisonWorkspace';
import PromptStructureOutline from './PromptStructureOutline';
import PromptResolvedView from './PromptResolvedView';
import PromptDiffWorkspace from './PromptDiffWorkspace';
import { useRunExecution } from '../../../hooks/useRunExecution';
import type { PlaygroundAnalysisGroup } from '../../../types/playground';
import iconModelConfig from '../../../assets/icon-modelconfig.svg';
import iconGuidedStart from '../../../assets/icon-guidedstart.svg';
import iconTool from '../../../assets/icon-tool.svg';
import iconOutputSchema from '../../../assets/icon-outputschema.svg';
import iconVariable from '../../../assets/icon-variable.svg';
import iconSave from '../../../assets/icons-save.svg';
import iconAnchor from '../../../assets/icon-anchor.svg';
import iconExecuteRun from '../../../assets/icon-executerun.svg';
import iconCreateNewPrompt from '../../../assets/icon-createnewprompt.svg';
import iconLoadFromExisting from '../../../assets/icon-loadfromexisting.svg';
import iconTrash from '../../../assets/icon-trash.svg';
import { promptAPI } from '../../../services/api';
import { generateUniquePromptId, slugifyPromptName } from '../../../utils/promptNaming';
import { resizeTextarea } from '../../../utils/resizeTextarea';
import {
  normalizePromptComponents,
  preparePromptComponentsForEditor,
  resolvePromptMessages,
  serializePromptMessages,
} from '../promptEditor';
import {
  clearStoredPromptVersionSelection,
  readStoredPromptSession,
  type PromptWorkflow,
  writeStoredPromptSession,
} from '../promptPlaygroundSession';

function getMaskIconStyle(icon: string): React.CSSProperties {
  return {
    WebkitMaskImage: `url("${icon}")`,
    maskImage: `url("${icon}")`,
  };
}

interface Props {
  mode: 'author' | 'analysis';
  hasResults: boolean;
  resultCount: number;
  analysisContent: React.ReactNode;
  onBackToEdit: () => void;
  onViewResults: () => void;
  onRunComplete: (groups: PlaygroundAnalysisGroup[]) => void;
  anchorOutput: string;
  anchorLabel: string | null;
  onAnchorChange: (value: string) => void;
  onClearAnchor: () => void;
}

const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ['gpt-4o', 'gpt-4o-mini'],
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
type SaveIntent = 'current' | 'new';
type EditorView = 'components' | 'resolved' | 'diff';

interface LoadedPromptContext {
  prompt: Prompt;
  version: PromptVersion;
  executeSignature: string;
  saveSignature: string;
}

interface LoadVersionIntoEditorOptions {
  preserveCompareTarget?: boolean;
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

const PANEL_SUMMARIES: Record<WorkspacePanel, { eyebrow: string; description: string }> = {
  guided: {
    eyebrow: 'Prompt setup',
    description: 'Start from a template and keep the first step aligned with the rest of the workspace.',
  },
  model: {
    eyebrow: 'Execution settings',
    description: 'Tune provider, model, and temperature in one compact layout without leaving the editor.',
  },
  variables: {
    eyebrow: 'Prompt inputs',
    description: 'Detected placeholders become editable values here, with textareas that expand until they reach a sensible limit.',
  },
  tools: {
    eyebrow: 'Callable surface',
    description: 'Define tool names, descriptions, and arguments with the same field rhythm used across the playground.',
  },
  schema: {
    eyebrow: 'Structured output',
    description: 'Keep schema controls and field definitions in a single place so output constraints stay easy to scan.',
  },
  anchor: {
    eyebrow: 'Reference output',
    description: 'Store a comparison anchor here and keep longer examples readable without manual resizing.',
  },
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

function schemaPropertiesFromOutputSchema(
  outputSchema: Record<string, unknown> | null | undefined,
): SchemaProperty[] {
  if (!outputSchema || typeof outputSchema.properties !== 'object') {
    return createDefaultSchemaProperties();
  }

  const properties = Object.entries(outputSchema.properties as Record<string, Record<string, unknown>>).map(([name, rawSchema]) => {
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

  return properties.length > 0 ? properties : createDefaultSchemaProperties();
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

function snapshotPromptVariables(
  components: PromptComponent[],
  inputVars: Record<string, string> | null | undefined,
) {
  return Object.fromEntries(
    collectPromptVariables(
      components.filter((component) => component.enabled && component.content.trim())
    ).map((variableName) => [variableName, inputVars?.[variableName] ?? ''])
  );
}

function normalizePromptTools(tools: PromptTool[]) {
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
}


const PromptForm: React.FC<Props> = ({
  mode,
  hasResults,
  resultCount,
  analysisContent,
  onBackToEdit,
  onViewResults,
  onRunComplete,
  anchorOutput,
  anchorLabel,
  onAnchorChange,
  onClearAnchor,
}) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const storedPromptSession = React.useMemo(() => readStoredPromptSession(), []);
  const requestedPromptId = searchParams.get('promptId') ?? '';
  const requestedVersionId = searchParams.get('versionId') ?? '';
  const requestedPromptSelectionId = requestedPromptId || storedPromptSession.selectedPromptId;
  const initialRequestedVersionId = requestedPromptId
    ? requestedVersionId
    : storedPromptSession.selectedVersionId;
  const initialPromptWorkflow: PromptWorkflow = requestedPromptId
    ? 'existing'
    : storedPromptSession.promptWorkflow;
  const triggerRefs = React.useRef<Record<WorkspacePanel, HTMLButtonElement | null>>({
    guided: null,
    model: null,
    variables: null,
    tools: null,
    schema: null,
    anchor: null,
  });
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null);
  const saveFieldRef = React.useRef<HTMLInputElement | null>(null);
  const saveActionRef = React.useRef<HTMLButtonElement | null>(null);
  const inputVarRefs = React.useRef<Record<string, HTMLTextAreaElement | null>>({});
  const anchorOutputRef = React.useRef<HTMLTextAreaElement | null>(null);
  const sectionRefs = React.useRef<Record<string, HTMLDivElement | null>>({});
  const promptRequestIdRef = React.useRef(0);
  const saveRequestIdRef = React.useRef(0);
  const saveInFlightRef = React.useRef(false);
  const previousActivePanelRef = React.useRef<WorkspacePanel | null>(null);
  const selectedPromptIdRef = React.useRef('');
  const promptWorkflowRef = React.useRef<PromptWorkflow>('new');
  const requestedVersionIdRef = React.useRef(initialRequestedVersionId);
  const variableHighlightRequestIdRef = React.useRef(0);
  const [promptComponents, setPromptComponents] = React.useState<PromptComponent[]>(
    () => preparePromptComponentsForEditor(DEFAULT_PROMPT_COMPONENTS)
  );
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
  const [selectedPromptId, setSelectedPromptId] = React.useState(requestedPromptSelectionId);
  const [selectedPromptData, setSelectedPromptData] = React.useState<PromptWithVersions | null>(null);
  const [selectedPromptLoading, setSelectedPromptLoading] = React.useState(false);
  const [comparisonTargets, setComparisonTargets] = React.useState<CompareTarget[]>([]);
  const [analysisPanel, setAnalysisPanel] = React.useState<'results' | 'compare'>('results');
  const [editorView, setEditorView] = React.useState<EditorView>('components');
  const [editorCompareTarget, setEditorCompareTarget] = React.useState<CompareTarget | null>(null);
  const [collapsedSections, setCollapsedSections] = React.useState<Record<string, boolean>>({});
  const [pendingScrollSectionKey, setPendingScrollSectionKey] = React.useState<string | null>(null);
  const [pendingVariableMatch, setPendingVariableMatch] = React.useState<PromptVariableMatch | null>(null);
  const [editorHighlightRequest, setEditorHighlightRequest] = React.useState<PromptVariableHighlightRequest | null>(null);
  const [promptWorkflow, setPromptWorkflow] = React.useState<PromptWorkflow>(initialPromptWorkflow);
  const [saveDialogOpen, setSaveDialogOpen] = React.useState(false);
  const [saveIntent, setSaveIntent] = React.useState<SaveIntent>('current');
  const [saveLoading, setSaveLoading] = React.useState(false);
  const [promptNameInput, setPromptNameInput] = React.useState('');
  const [saveError, setSaveError] = React.useState<string | null>(null);
  const [isRunCountOpen, setIsRunCountOpen] = React.useState(false);
  const runCountRef = React.useRef<HTMLDivElement>(null);

  const [schemaEnabled, setSchemaEnabled] = React.useState(true);
  const [schemaProperties, setSchemaProperties] = React.useState<SchemaProperty[]>(() => createDefaultSchemaProperties());
  const [tools, setTools] = React.useState<PromptTool[]>([]);

  const [toolError, setToolError] = React.useState<string | null>(null);
  const [revisionNote, setRevisionNote] = React.useState('');
  const [appliedTemplateId, setAppliedTemplateId] = React.useState<string | null>(null);

  const [guidedOverlayStep, setGuidedOverlayStep] = React.useState<GuidedOverlayStep | null>(null);
  const [selectedGuidedRole, setSelectedGuidedRole] = React.useState<GuidedStartRole | null>(null);
  const workspaceBodyRef = React.useRef<HTMLDivElement | null>(null);
  const componentStats = React.useMemo<Record<PromptComponentType, number> | null>(() => {
    if (!selectedGuidedRole || guidedOverlayStep === null) {
      return null;
    }
    return Object.fromEntries(
      selectedGuidedRole.components.map((comp) => [comp.component_type, comp.prevalence])
    ) as Record<PromptComponentType, number>;
  }, [selectedGuidedRole, guidedOverlayStep]);
  const workspaceModeHintId = React.useId();

  const {
    loading, setupError, execute,
  } = useRunExecution();

  React.useEffect(() => {
    setActivePanel(null);
  }, [mode]);

  const handleEditorViewClick = React.useCallback((view: EditorView) => {
    if (mode !== 'author') {
      onBackToEdit();
    }
    setEditorView(view);
  }, [mode, onBackToEdit]);
  const handleAnalysisNavClick = React.useCallback(() => {
    if (mode === 'analysis') {
      setAnalysisPanel('results');
      return;
    }

    onViewResults();
  }, [mode, onViewResults]);

  const isEditorActive = (view: EditorView) => mode === 'author' && editorView === view;

  const workspaceNav = (
    <div className="seg-control playground-workspace-nav" role="group" aria-label="Workspace view">
      <button
        type="button"
        className={`seg-control__btn${isEditorActive('components') ? ' is-active' : ''}`}
        onClick={() => handleEditorViewClick('components')}
        aria-pressed={isEditorActive('components')}
      >
        Components
      </button>
      <button
        type="button"
        className={`seg-control__btn${isEditorActive('resolved') ? ' is-active' : ''}`}
        onClick={() => handleEditorViewClick('resolved')}
        aria-pressed={isEditorActive('resolved')}
      >
        Resolved
      </button>
      {editorCompareTarget && (
        <button
          type="button"
          className={`seg-control__btn${isEditorActive('diff') ? ' is-active' : ''}`}
          onClick={() => handleEditorViewClick('diff')}
          aria-pressed={isEditorActive('diff')}
        >
          Diff
        </button>
      )}
      <span className="playground-workspace-nav__divider" aria-hidden="true" />
      <button
        type="button"
        className={`seg-control__btn${mode === 'analysis' ? ' is-active' : ''}`}
        onClick={handleAnalysisNavClick}
        aria-pressed={mode === 'analysis'}
        aria-describedby={mode === 'author' && !hasResults ? workspaceModeHintId : undefined}
        disabled={!hasResults}
      >
        <span>Outputs</span>
        {hasResults && <span className="playground-workspace-nav__count">{resultCount}</span>}
      </button>
    </div>
  );

  const refreshPromptList = React.useCallback(async () => {
    const prompts = await promptAPI.getAllPrompts();
    setSavedPrompts(prompts);
    setSavedPromptsError(null);
    setSelectedPromptId((current) => (
      current && prompts.some((prompt) => prompt.prompt_id === current) ? current : ''
    ));
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
  const normalizedPromptComponents = React.useMemo(
    () => normalizePromptComponents(promptComponents),
    [promptComponents]
  );
  const activePanelId = activePanel ? `playground-config-panel-${activePanel}` : undefined;
  const activePanelDescriptionId = activePanel ? `${activePanelId}-description` : undefined;
  const existingPromptHintId = 'playground-existing-prompt-hint';
  const existingPromptErrorId = 'playground-existing-prompt-error';
  const activePromptComponents = React.useMemo(
    () => normalizedPromptComponents.filter((component) => component.enabled && component.content.trim()),
    [normalizedPromptComponents]
  );
  const detectedVariables = React.useMemo(
    () => collectPromptVariables(activePromptComponents),
    [activePromptComponents]
  );
  let existingPromptHint = 'choose a prompt and the latest saved version will load automatically.';
  if (savedPromptsLoading) {
    existingPromptHint = 'loading available prompts.';
  } else if (savedPrompts.length === 0) {
    existingPromptHint = 'no saved prompts yet. save or create one first.';
  } else if (selectedPromptLoading) {
    existingPromptHint = 'loading the latest saved version for this prompt.';
  } else if (selectedPromptData) {
    existingPromptHint = 'this prompt has no saved version yet. save it once to create v1.';
  }
  const existingPromptDescribedBy = [
    !loadedPromptContext ? existingPromptHintId : null,
    savedPromptsError ? existingPromptErrorId : null,
  ].filter(Boolean).join(' ') || undefined;

  React.useEffect(() => {
    selectedPromptIdRef.current = selectedPromptId;
  }, [selectedPromptId]);

  React.useEffect(() => {
    promptWorkflowRef.current = promptWorkflow;
  }, [promptWorkflow]);

  React.useEffect(() => {
    writeStoredPromptSession({
      promptWorkflow,
      selectedPromptId,
    });
  }, [promptWorkflow, selectedPromptId]);

  React.useEffect(() => {
    if (!activePanel && !saveDialogOpen) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (saveDialogOpen) {
          setSaveDialogOpen(false);
        }
        setActivePanel(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activePanel, saveDialogOpen]);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (runCountRef.current && !runCountRef.current.contains(event.target as Node)) {
        setIsRunCountOpen(false);
      }
    };
    if (isRunCountOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isRunCountOpen]);

  React.useEffect(() => {
    const textareaElements = [
      ...detectedVariables.map((variableName) => inputVarRefs.current[variableName]),
      anchorOutputRef.current,
    ];
    textareaElements.forEach((textarea) => {
      resizeTextarea(textarea);
    });
  }, [detectedVariables, inputVars, anchorOutput]);

  React.useEffect(() => {
    if (activePanel) {
      previousActivePanelRef.current = activePanel;
      closeButtonRef.current?.focus();
      return undefined;
    }

    if (previousActivePanelRef.current) {
      triggerRefs.current[previousActivePanelRef.current]?.focus();
      previousActivePanelRef.current = null;
    }

    return undefined;
  }, [activePanel]);
  React.useEffect(() => {
    if (editorView !== 'components' || !pendingScrollSectionKey) {
      return;
    }

    const target = sectionRefs.current[pendingScrollSectionKey];
    if (!target) {
      setPendingScrollSectionKey(null);
      setPendingVariableMatch(null);
      return;
    }

    const matchToHighlight = pendingVariableMatch?.componentKey === pendingScrollSectionKey
      ? pendingVariableMatch
      : null;
    const frameId = requestAnimationFrame(() => {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
      setPendingScrollSectionKey(null);
      if (matchToHighlight) {
        variableHighlightRequestIdRef.current += 1;
        setEditorHighlightRequest({
          ...matchToHighlight,
          requestId: variableHighlightRequestIdRef.current,
        });
        setPendingVariableMatch(null);
      }
    });

    return () => cancelAnimationFrame(frameId);
  }, [editorView, pendingScrollSectionKey, pendingVariableMatch]);

  React.useEffect(() => {
    if (!saveDialogOpen) {
      return;
    }

    const target = saveFieldRef.current ?? saveActionRef.current;
    target?.focus();
  }, [saveDialogOpen, saveIntent, promptWorkflow, loadedPromptContext]);

  React.useEffect(() => {
    refreshPromptList()
      .catch(() => setSavedPromptsError('Failed to load saved prompts.'))
      .finally(() => setSavedPromptsLoading(false));
  }, [refreshPromptList]);

  const normalizedTools = React.useMemo(() => {
    return normalizePromptTools(tools);
  }, [tools]);
  const schemaSuppressedByTools = normalizedTools.length > 0;
  const activeSchemaError = schemaSuppressedByTools ? null : schemaError;
  const currentOutputSchema = React.useMemo(
    () => (schemaEnabled && !schemaError ? toJsonSchema(schemaProperties) : null),
    [schemaEnabled, schemaError, schemaProperties]
  );
  const runtimeOutputSchema = schemaSuppressedByTools ? null : currentOutputSchema;
  const currentResolvedPrompt = React.useMemo(
    () => serializePromptMessages(resolvePromptMessages(normalizedPromptComponents, inputVars)),
    [normalizedPromptComponents, inputVars]
  );
  const defaultEditorSaveSignature = React.useMemo(
    () => {
      const defaultComponents = preparePromptComponentsForEditor(DEFAULT_PROMPT_COMPONENTS);
      return getSavedPromptSignature(
        defaultComponents,
        snapshotPromptVariables(defaultComponents, DEFAULT_INPUT_VARS),
        [],
        toJsonSchema(createDefaultSchemaProperties())
      );
    },
    []
  );
  const currentSaveSignature = React.useMemo(
    () => getSavedPromptSignature(
      normalizedPromptComponents,
      snapshotPromptVariables(normalizedPromptComponents, inputVars),
      normalizedTools,
      currentOutputSchema
    ),
    [normalizedPromptComponents, inputVars, normalizedTools, currentOutputSchema]
  );
  const draftLeaf = React.useMemo(() => {
    if (
      !loadedPromptContext
      || !selectedPromptData
      || loadedPromptContext.prompt.prompt_id !== selectedPromptData.prompt.prompt_id
      || loadedPromptContext.saveSignature === currentSaveSignature
    ) {
      return null;
    }

    return {
      promptId: loadedPromptContext.prompt.prompt_id,
      parentVersionId: loadedPromptContext.version.version_id,
      versionId: `draft-${loadedPromptContext.version.version_id}`,
      name: revisionNote.trim() || 'unsaved draft',
      revisionNote: revisionNote.trim() || 'current editor changes',
      components: normalizedPromptComponents,
      variables: snapshotPromptVariables(normalizedPromptComponents, inputVars),
      tools: normalizedTools,
      outputSchema: currentOutputSchema,
    };
  }, [
    loadedPromptContext,
    selectedPromptData,
    currentSaveSignature,
    currentOutputSchema,
    inputVars,
    normalizedPromptComponents,
    normalizedTools,
    revisionNote,
  ]);
  const generatedPromptIdPreview = React.useMemo(
    () => slugifyPromptName(promptNameInput || 'playground prompt'),
    [promptNameInput]
  );
  const activePanelBadges = React.useMemo(() => {
    if (!activePanel) {
      return [];
    }

    switch (activePanel) {
      case 'guided':
        return selectedGuidedRole ? [selectedGuidedRole.name] : ['choose a role'];
      case 'model':
        return [provider, model, `${numRuns} run${numRuns === 1 ? '' : 's'}`];
      case 'variables':
        return [`${detectedVariables.length} variable${detectedVariables.length === 1 ? '' : 's'}`];
      case 'tools':
        return [`${normalizedTools.length} tool${normalizedTools.length === 1 ? '' : 's'}`];
      case 'schema':
        return [
          schemaSuppressedByTools ? 'paused by tools' : schemaEnabled ? 'enabled' : 'disabled',
          `${schemaEnabled ? schemaProperties.length : 0} field${schemaProperties.length === 1 ? '' : 's'}`,
        ];
      case 'anchor':
        return [anchorOutput.trim() ? 'anchor set' : 'optional'];
      default:
        return [];
    }
  }, [
    activePanel,
    anchorOutput,
    detectedVariables.length,
    model,
    normalizedTools.length,
    numRuns,
    provider,
    schemaSuppressedByTools,
    schemaEnabled,
    schemaProperties.length,
  ]);
  const executeButtonDisabled = loading || !!activeSchemaError || (promptWorkflow === 'existing' && !loadedPromptContext);
  const executeRunControl = (
    <div className="create-run__execute-control" role="group" aria-label="Execute run controls" ref={runCountRef}>
      <button
        type="submit"
        className="btn btn--primary create-run__action-btn create-run__execute-btn"
        disabled={executeButtonDisabled}
      >
        <span
          className="create-run__action-icon"
          style={getMaskIconStyle(iconExecuteRun)}
          aria-hidden
        />
        <span>{mode === 'analysis' ? (loading ? 'Executing...' : 'Run again') : (loading ? 'Executing...' : 'Start')}</span>
        {loading && <span className="spinner create-run__spinner" aria-hidden />}
      </button>
      <button
        type="button"
        className="btn btn--primary create-run__execute-dropdown-btn"
        onClick={() => setIsRunCountOpen(!isRunCountOpen)}
        disabled={executeButtonDisabled}
        aria-label="Configure number of runs"
        aria-expanded={isRunCountOpen}
      >
        <span className="create-run__execute-dropdown-icon" aria-hidden />
      </button>

      {isRunCountOpen && (
        <div className="create-run__run-count-popover">
          <div className="create-run__run-count-row">
            <div className="create-run__run-count-text">
              <label htmlFor="playground-num-runs-popover" className="create-run__run-count-title">Repetitions</label>
              <p className="create-run__run-count-desc">
                Run the same prompt multiple times to reduce variability in results.
              </p>
            </div>
            <input
              id="playground-num-runs-popover"
              type="number"
              className="input create-run__run-count-input"
              min="1"
              max="10"
              value={numRuns}
              onChange={(e) => setNumRuns(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
            />
          </div>
        </div>
      )}
    </div>
  );
  const currentEditorVersionLabel = React.useMemo(() => {
    if (loadedPromptContext) {
      return loadedPromptContext.prompt.name;
    }

    return promptNameInput.trim() || 'Untitled prompt';
  }, [loadedPromptContext, promptNameInput]);
  const currentEditorPromptId = loadedPromptContext?.prompt.prompt_id
    ?? (promptWorkflow === 'new' ? generatedPromptIdPreview : selectedPromptData?.prompt.prompt_id ?? null);
  const currentEditorVersionId = draftLeaf?.versionId
    ?? loadedPromptContext?.version.version_id
    ?? null;
  const canSavePrompt = React.useMemo(() => {
    if (promptWorkflow === 'existing') {
      return Boolean(
        (loadedPromptContext && loadedPromptContext.saveSignature !== currentSaveSignature)
        || revisionNote.trim()
      );
    }

    return Boolean(
      promptNameInput.trim()
      || revisionNote.trim()
      || currentSaveSignature !== defaultEditorSaveSignature
    );
  }, [
    currentSaveSignature,
    defaultEditorSaveSignature,
    loadedPromptContext,
    promptNameInput,
    promptWorkflow,
    revisionNote,
  ]);
  const editorCompareTargetResolvedPrompt = React.useMemo(() => {
    if (!editorCompareTarget) {
      return '';
    }

    return serializePromptMessages(resolvePromptMessages(
      normalizePromptComponents(editorCompareTarget.version.components),
      editorCompareTarget.version.variables ?? {}
    ));
  }, [editorCompareTarget]);
  const versionTreeCompareTargets = React.useMemo(() => {
    const mappedTargets = comparisonTargets.map((target) => ({
      promptId: target.promptId,
      versionId: target.version.version_id,
    }));
    const editorTarget = editorCompareTarget
      ? [{ promptId: editorCompareTarget.promptId, versionId: editorCompareTarget.version.version_id }]
      : [];

    if (mode === 'author') {
      return editorTarget;
    }

    return [...mappedTargets, ...editorTarget].filter((target, index, allTargets) => (
      allTargets.findIndex((candidate) => (
        candidate.promptId === target.promptId && candidate.versionId === target.versionId
      )) === index
    ));
  }, [comparisonTargets, editorCompareTarget, mode]);
  const resetEditorState = React.useCallback((nextPromptName = '') => {
    setPromptComponents(preparePromptComponentsForEditor(DEFAULT_PROMPT_COMPONENTS));
    setInputVars(DEFAULT_INPUT_VARS);
    setTools([]);
    setToolError(null);
    setSchemaEnabled(true);
    setSchemaProperties(createDefaultSchemaProperties());
    setRevisionNote('');
    setAppliedTemplateId(null);
    setPromptNameInput(nextPromptName);
    setLoadedPromptContext(null);
    setCollapsedSections({});
    setPendingScrollSectionKey(null);
    setEditorCompareTarget(null);
    setEditorView('components');
  }, []);

  const loadVersionIntoEditor = React.useCallback((
    prompt: Prompt,
    version: PromptVersion,
    options: LoadVersionIntoEditorOptions = {},
  ) => {
    const editorComponents = preparePromptComponentsForEditor(version.components);
    const parsedSchemaProperties = schemaPropertiesFromOutputSchema(
      (version.output_schema as Record<string, unknown> | null) ?? null
    );
    const canonicalOutputSchema = version.output_schema
      ? toJsonSchema(parsedSchemaProperties)
      : null;

    setPromptComponents(editorComponents);
    setTools(version.tools ?? []);
    setSchemaEnabled(Boolean(version.output_schema));
    setSchemaProperties(parsedSchemaProperties);
    setInputVars(version.variables ?? {});
    setRevisionNote('');
    setAppliedTemplateId(version.source_template_id ?? null);
    setPromptNameInput(prompt.name);
    setPromptWorkflow('existing');
    setSelectedPromptId(prompt.prompt_id);
    setSaveError(null);
    setCollapsedSections({});
    setPendingScrollSectionKey(null);
    if (!options.preserveCompareTarget) {
      setEditorCompareTarget(null);
      setEditorView('components');
    }
    setLoadedPromptContext({
      prompt,
      version,
      executeSignature: getExecutionPromptSignature(
        editorComponents.filter((component) => component.enabled && component.content.trim()),
        normalizePromptTools(version.tools ?? []),
        canonicalOutputSchema
      ),
      saveSignature: getSavedPromptSignature(
        editorComponents,
        snapshotPromptVariables(editorComponents, version.variables ?? {}),
        normalizePromptTools(version.tools ?? []),
        canonicalOutputSchema
      ),
    });
  }, []);

  React.useEffect(() => {
    if (selectedPromptId === requestedPromptSelectionId) {
      return;
    }

    requestedVersionIdRef.current = '';
    clearStoredPromptVersionSelection();
  }, [requestedPromptSelectionId, selectedPromptId]);

  React.useEffect(() => {
    if (!requestedPromptId || selectedPromptId === requestedPromptSelectionId) {
      return;
    }

    const nextSearchParams = new URLSearchParams(searchParams);
    nextSearchParams.delete('promptId');
    nextSearchParams.delete('versionId');
    setSearchParams(nextSearchParams, { replace: true });
  }, [
    requestedPromptId,
    requestedPromptSelectionId,
    searchParams,
    selectedPromptId,
    setSearchParams,
  ]);

  React.useEffect(() => {
    if (!selectedPromptId) {
      setSelectedPromptData(null);
      setLoadedPromptContext(null);
      setSelectedPromptLoading(false);
      return;
    }

    let cancelled = false;
    const requestId = promptRequestIdRef.current + 1;
    promptRequestIdRef.current = requestId;
    setComparisonTargets([]);
    setSelectedPromptData(null);
    setLoadedPromptContext(null);
    setSelectedPromptLoading(true);
    promptAPI.getPrompt(selectedPromptId)
      .then((promptData) => {
        if (cancelled || promptRequestIdRef.current !== requestId) {
          return;
        }

        setSavedPromptsError(null);
        setSelectedPromptData(promptData);
        const requestedVersionId = promptData.prompt.prompt_id === requestedPromptSelectionId
          ? requestedVersionIdRef.current
          : '';
        const requestedVersion = requestedVersionId
          ? promptData.versions.find((version) => version.version_id === requestedVersionId) ?? null
          : null;
        const latestVersion = promptData.versions.find(
          (version) => version.version_id === promptData.prompt.latest_version_id
        ) ?? [...promptData.versions].sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
        const versionToLoad = requestedVersion ?? latestVersion;

        if (requestedVersionId) {
          requestedVersionIdRef.current = '';
          clearStoredPromptVersionSelection();
        }

        if (versionToLoad) {
          loadVersionIntoEditor(promptData.prompt, versionToLoad);
          return;
        }

        resetEditorState(promptData.prompt.name);
      })
      .catch(() => {
        if (!cancelled && promptRequestIdRef.current === requestId) {
          setSelectedPromptData(null);
          setLoadedPromptContext(null);
          setSavedPromptsError('Failed to load prompt versions.');
        }
      })
      .finally(() => {
        if (!cancelled && promptRequestIdRef.current === requestId) {
          setSelectedPromptLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedPromptId, loadVersionIntoEditor, requestedPromptSelectionId, resetEditorState]);

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
  const toggleEditorCompareTarget = React.useCallback((promptName: string, version: PromptVersion) => {
    setEditorCompareTarget((current) => {
      const isSameTarget = current?.promptId === version.prompt_id && current.version.version_id === version.version_id;
      if (isSameTarget) {
        setEditorView((currentView) => (currentView === 'diff' ? 'components' : currentView));
        return null;
      }

      setEditorView('diff');
      return {
        promptId: version.prompt_id,
        promptName,
        version,
      };
    });
  }, []);
  const toggleCollapsedSection = React.useCallback((componentKey: string) => {
    setCollapsedSections((current) => ({
      ...current,
      [componentKey]: !(current[componentKey] ?? false),
    }));
  }, []);
  const copyPromptComponentContent = React.useCallback((index: number) => {
    const source = normalizedPromptComponents[index];
    if (!source?.content || !globalThis.navigator?.clipboard?.writeText) {
      return;
    }

    void globalThis.navigator.clipboard.writeText(source.content);
  }, [normalizedPromptComponents]);
  const reorderPromptComponent = React.useCallback((fromIndex: number, toIndex: number) => {
    setPromptComponents((current) => {
      if (
        fromIndex === toIndex
        || fromIndex < 0
        || toIndex < 0
        || fromIndex >= current.length
        || toIndex >= current.length
      ) {
        return current;
      }

      const next = [...current];
      const [item] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, item);
      return next;
    });
  }, []);
  const handleJumpToSection = React.useCallback((componentKey: string) => {
    setPendingVariableMatch(null);
    setCollapsedSections((current) => {
      if (!(current[componentKey] ?? false)) {
        return current;
      }

      return {
        ...current,
        [componentKey]: false,
      };
    });
    setEditorView('components');
    setPendingScrollSectionKey(componentKey);
  }, []);
  const handleJumpToVariable = React.useCallback((variableName: string) => {
    const match = findPromptVariableMatch(normalizedPromptComponents, variableName);

    setEditorView('components');
    if (!match) {
      setPendingVariableMatch(null);
      return;
    }

    setCollapsedSections((current) => {
      if (!(current[match.componentKey] ?? false)) {
        return current;
      }

      return {
        ...current,
        [match.componentKey]: false,
      };
    });
    setPendingVariableMatch(match);
    setPendingScrollSectionKey(match.componentKey);
  }, [normalizedPromptComponents]);

  const handleStartNewPrompt = React.useCallback((nextPromptName = '') => {
    setPromptWorkflow('new');
    setSelectedPromptId('');
    setSelectedPromptData(null);
    setComparisonTargets([]);
    resetEditorState(nextPromptName);
    setSaveError(null);
  }, [resetEditorState]);

  const validateNormalizedTools = React.useCallback(() => {
    const emptyTool = normalizedTools.find((tool) => !tool.name || !tool.description);
    if (emptyTool) {
      return 'Each tool needs a name and description.';
    }

    const duplicateToolNames = new Set<string>();
    for (const tool of normalizedTools) {
      if (tool.name === 'structured_output') {
        return 'Tool name "structured_output" is reserved.';
      }
      if (!TOOL_NAME_REGEX.test(tool.name)) {
        return `Tool "${tool.name}" must use only letters, numbers, underscores, or hyphens.`;
      }
      if (duplicateToolNames.has(tool.name)) {
        return 'Tool names must be unique.';
      }
      duplicateToolNames.add(tool.name);

      const emptyArgument = tool.arguments.find((argument) => !argument.name);
      if (emptyArgument) {
        return `Tool "${tool.name}" has an argument without a name.`;
      }

      const argumentNames = new Set<string>();
      for (const argument of tool.arguments) {
        if (!TOOL_ARGUMENT_NAME_REGEX.test(argument.name)) {
          return `Argument "${argument.name}" in tool "${tool.name}" must start with a letter or underscore.`;
        }
        if (argumentNames.has(argument.name)) {
          return `Tool "${tool.name}" has duplicate argument names.`;
        }
        argumentNames.add(argument.name);
      }
    }

    return null;
  }, [normalizedTools]);

  const openSaveDialog = React.useCallback((intent: SaveIntent = 'current') => {
    setSaveIntent(intent);
    setActivePanel(null);
    setSaveDialogOpen(true);
    setSaveError(null);
  }, []);

  React.useEffect(() => {
    if (comparisonTargets.length === 2) {
      setAnalysisPanel('compare');
      return;
    }

    setAnalysisPanel((current) => (current === 'compare' ? 'results' : current));
  }, [comparisonTargets.length]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setToolError(null);
    setIsRunCountOpen(false);

    if (activeSchemaError) {
      return;
    }

    if (promptWorkflow === 'existing' && !loadedPromptContext) {
      setSavedPromptsError(
        selectedPromptData
          ? 'This prompt has no loaded saved version yet. Save it once to create the first version.'
          : 'Select an existing prompt from the history rail first.'
      );
      return;
    }

    const toolValidationError = validateNormalizedTools();
    if (toolValidationError) {
      setToolError(toolValidationError);
      return;
    }

    const submittedInputVariables = Object.fromEntries(
      detectedVariables.map((variableName) => [variableName, inputVars[variableName] ?? ''])
    );
    const selectedPromptIdAtSubmit = selectedPromptIdRef.current;
    const promptWorkflowAtSubmit = promptWorkflowRef.current;
    const executionPromptName = promptWorkflow === 'existing'
      ? loadedPromptContext?.prompt.name ?? selectedPromptData?.prompt.name ?? 'Playground Prompt'
      : promptNameInput.trim() || 'Untitled prompt';

    const executionResult = await execute({
      components: activePromptComponents,
      tools: normalizedTools,
      inputVariables: submittedInputVariables,
      model,
      provider,
      temperature,
      numRuns,
      outputSchema: runtimeOutputSchema,
      promptContext: {
        promptId: promptWorkflow === 'existing'
          ? loadedPromptContext?.prompt.prompt_id ?? selectedPromptData?.prompt.prompt_id ?? null
          : null,
        promptName: executionPromptName,
        versionId: promptWorkflow === 'existing' ? currentEditorVersionId : null,
        branchName: promptWorkflow === 'existing' ? loadedPromptContext?.version.branch_name ?? null : null,
        loadedSignature: loadedPromptContext?.executeSignature ?? null,
        revisionNote: revisionNote.trim() || null,
        sourceTemplateId: appliedTemplateId,
      },
    });

    if (!executionResult) {
      return;
    }

    if (editorCompareTarget) {
      const compareInputVariables = {
        ...(editorCompareTarget.version.variables ?? {}),
        ...submittedInputVariables,
      };
      const comparisonResult = await execute({
        components: normalizePromptComponents(editorCompareTarget.version.components),
        tools: normalizePromptTools(editorCompareTarget.version.tools ?? []),
        inputVariables: compareInputVariables,
        model,
        provider,
        temperature,
        numRuns,
        outputSchema: editorCompareTarget.version.tools?.length
          ? null
          : editorCompareTarget.version.output_schema as Record<string, unknown> | null,
        promptContext: {
          promptId: editorCompareTarget.promptId,
          promptName: editorCompareTarget.promptName,
          versionId: editorCompareTarget.version.version_id,
          branchName: editorCompareTarget.version.branch_name ?? null,
          useExistingVersion: true,
        },
      });

      if (!comparisonResult) {
        onRunComplete([
          {
            id: 'primary',
            label: executionPromptName,
            tone: 'primary',
            promptId: executionResult.promptId,
            versionId: executionResult.versionId,
            results: executionResult.results,
            runErrors: executionResult.errors,
          },
          {
            id: 'compare',
            label: editorCompareTarget.version.name || editorCompareTarget.promptName,
            tone: 'compare',
            promptId: editorCompareTarget.promptId,
            versionId: editorCompareTarget.version.version_id,
            results: Array.from({ length: numRuns }, () => null),
            runErrors: Array.from({ length: numRuns }, () => 'comparison run failed'),
          },
        ]);
        return;
      }

      onRunComplete([
        {
          id: 'primary',
          label: executionPromptName,
          tone: 'primary',
          promptId: executionResult.promptId,
          versionId: executionResult.versionId,
          results: executionResult.results,
          runErrors: executionResult.errors,
        },
        {
          id: 'compare',
          label: editorCompareTarget.version.name || editorCompareTarget.promptName,
          tone: 'compare',
          promptId: comparisonResult.promptId,
          versionId: comparisonResult.versionId,
          results: comparisonResult.results,
          runErrors: comparisonResult.errors,
        },
      ]);
    } else {
      onRunComplete([{
        id: 'primary',
        label: executionPromptName,
        tone: 'primary',
        promptId: executionResult.promptId,
        versionId: executionResult.versionId,
        results: executionResult.results,
        runErrors: executionResult.errors,
      }]);
    }

    await refreshPromptList().catch(() => setSavedPromptsError('Run completed, but prompt list refresh failed.'));
    if (
      executionResult.promptId &&
      selectedPromptIdRef.current === selectedPromptIdAtSubmit &&
      promptWorkflowRef.current === promptWorkflowAtSubmit
    ) {
      setSelectedPromptId(executionResult.promptId);
      if (executionResult.promptId !== selectedPromptIdAtSubmit) {
        return;
      }
      const promptData = await promptAPI.getPrompt(executionResult.promptId).catch(() => null);
      if (!promptData) {
        setSavedPromptsError('Run completed, but the updated prompt version could not be reloaded.');
        return;
      }
      if (
        selectedPromptIdRef.current !== selectedPromptIdAtSubmit
        || promptWorkflowRef.current !== promptWorkflowAtSubmit
      ) {
        return;
      }
      setSelectedPromptData(promptData);
      const executedVersion = promptData.versions.find((version) => version.version_id === executionResult.versionId);
      if (executedVersion) {
        loadVersionIntoEditor(promptData.prompt, executedVersion, {
          preserveCompareTarget: Boolean(editorCompareTarget),
        });
      }
    }
  };

  const handleSavePrompt = async (intent: SaveIntent = 'current') => {
    if (saveInFlightRef.current || saveLoading) {
      return;
    }

    saveInFlightRef.current = true;
    const requestId = saveRequestIdRef.current + 1;
    saveRequestIdRef.current = requestId;
    const selectedPromptIdAtSave = selectedPromptIdRef.current;
    const promptWorkflowAtSave = promptWorkflowRef.current;
    setSaveLoading(true);
    setSaveError(null);

    const effectiveWorkflow: PromptWorkflow =
      intent === 'new' ? 'new' : promptWorkflow;

    if (effectiveWorkflow === 'existing' && !loadedPromptContext && !selectedPromptData) {
      setSaveError('Select an existing prompt from the history rail first.');
      setSaveLoading(false);
      saveInFlightRef.current = false;
      return;
    }

    const nextPromptName = effectiveWorkflow === 'existing'
      ? loadedPromptContext?.prompt.name ?? selectedPromptData?.prompt.name ?? ''
      : promptNameInput.trim();

    if (!nextPromptName) {
      setSaveError('Prompt name is required to create a new prompt.');
      setSaveLoading(false);
      saveInFlightRef.current = false;
      return;
    }

    if (schemaError) {
      setSaveError(schemaError);
      setSaveLoading(false);
      saveInFlightRef.current = false;
      return;
    }

    const toolValidationError = validateNormalizedTools();
    if (toolValidationError) {
      setToolError(toolValidationError);
      setSaveLoading(false);
      saveInFlightRef.current = false;
      return;
    }

    try {
      const savedInputVariables = snapshotPromptVariables(normalizedPromptComponents, inputVars);
      const saveSignature = getSavedPromptSignature(
        normalizedPromptComponents,
        savedInputVariables,
        normalizedTools,
        currentOutputSchema
      );
      const nextPromptId = effectiveWorkflow === 'existing'
        ? loadedPromptContext?.prompt.prompt_id ?? selectedPromptData?.prompt.prompt_id ?? ''
        : await generateUniquePromptId(nextPromptName, promptAPI.getPrompt);

      if (
        effectiveWorkflow === 'existing' &&
        loadedPromptContext &&
        loadedPromptContext.prompt.prompt_id === nextPromptId &&
        loadedPromptContext.saveSignature === saveSignature &&
        !revisionNote.trim()
      ) {
        if (loadedPromptContext.prompt.name !== nextPromptName) {
          await promptAPI.updatePrompt(nextPromptId, {
            name: nextPromptName,
            description: loadedPromptContext.prompt.description ?? 'Saved from playground',
          });
        }
        if (
          saveRequestIdRef.current === requestId
          && selectedPromptIdRef.current === selectedPromptIdAtSave
          && promptWorkflowRef.current === promptWorkflowAtSave
        ) {
          setSaveDialogOpen(false);
        }
        await refreshPromptList().catch(() => setSavedPromptsError('Prompt saved, but prompt list refresh failed.'));
        return;
      }

      const existingPromptData = effectiveWorkflow === 'existing'
        ? await promptAPI.getPrompt(nextPromptId).catch(() => null)
        : null;
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
        effectiveWorkflow === 'existing' && loadedPromptContext?.prompt.prompt_id === nextPromptId
          ? loadedPromptContext.version.version_id
          : latestPromptData.prompt.latest_version_id ?? null;

      const createdVersion = await promptAPI.createVersion(nextPromptId, {
        name: revisionNote.trim() || `${nextPromptName} version`,
        components: normalizedPromptComponents,
        variables: savedInputVariables,
        output_schema: currentOutputSchema,
        tools: normalizedTools,
        parent_version_id: parentVersionId,
        branch_name: effectiveWorkflow === 'existing' && loadedPromptContext?.prompt.prompt_id === nextPromptId
          ? loadedPromptContext.version.branch_name ?? undefined
          : undefined,
        revision_note: revisionNote.trim() || undefined,
        source_template_id: appliedTemplateId ?? undefined,
      });
      const promptData = await promptAPI.getPrompt(nextPromptId);
      if (
        saveRequestIdRef.current === requestId
        && selectedPromptIdRef.current === selectedPromptIdAtSave
        && promptWorkflowRef.current === promptWorkflowAtSave
      ) {
        setSelectedPromptId(nextPromptId);
        setSelectedPromptData(promptData);
        setPromptNameInput(nextPromptName);
        const savedVersion = promptData.versions.find((version) => version.version_id === createdVersion.version_id);
        if (savedVersion) {
          loadVersionIntoEditor(promptData.prompt, savedVersion);
        }
        setSaveDialogOpen(false);
      }
      await refreshPromptList().catch(() => setSavedPromptsError('Prompt saved, but prompt list refresh failed.'));
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save prompt.');
    } finally {
      if (saveRequestIdRef.current === requestId) {
        setSaveLoading(false);
      }
      if (saveRequestIdRef.current === requestId) {
        saveInFlightRef.current = false;
      }
    }
  };

  return (
    <div>
      <div className="card">
        <div className="card__body">
          <form onSubmit={handleSubmit} className="flex-col create-run__form">
            <div className={`playground-shell playground-shell--${mode}`}>
              <aside className="playground-shell__rail">
                <div className="card">
                  <div className="card__body playground-rail__section">
                    <div className="seg-control playground-seg-control" role="group" aria-label="Prompt workflow">
                      <button
                        type="button"
                        className={`seg-control__btn${promptWorkflow === 'new' ? ' is-active' : ''}`}
                        onClick={() => handleStartNewPrompt()}
                        aria-pressed={promptWorkflow === 'new'}
                      >
                        <span
                          className="playground-seg-control__icon"
                          style={getMaskIconStyle(iconCreateNewPrompt)}
                          aria-hidden
                        />
                        <span>New prompt</span>
                      </button>
                      <button
                        type="button"
                        className={`seg-control__btn${promptWorkflow === 'existing' ? ' is-active' : ''}`}
                        onClick={() => setPromptWorkflow('existing')}
                        aria-pressed={promptWorkflow === 'existing'}
                      >
                        <span
                          className="playground-seg-control__icon"
                          style={getMaskIconStyle(iconLoadFromExisting)}
                          aria-hidden
                        />
                        <span>Existing prompt</span>
                      </button>
                    </div>
                    {promptWorkflow === 'new' ? (
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
                        <span className="field__hint">
                          prompt id will be generated automatically from <code>{generatedPromptIdPreview}</code>.
                        </span>
                      </div>
                    ) : (
                      <>
                        <div className="field">
                          <label className="field__label" htmlFor="playground-existing-prompt">
                            Select prompt
                          </label>
                          <select
                            id="playground-existing-prompt"
                            className="select"
                            value={selectedPromptId}
                            onChange={(e) => setSelectedPromptId(e.target.value)}
                            aria-describedby={existingPromptDescribedBy}
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
                          {loadedPromptContext ? null : (
                            <span id={existingPromptHintId} className="field__hint">
                              {existingPromptHint}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                    {savedPromptsError && <span id={existingPromptErrorId} className="field__error">{savedPromptsError}</span>}
                  </div>
                </div>

                <div className="card">
                  <div className="card__body playground-rail__section">
                    <div className="create-run__field-head">
                      <label className="field__label">Version history</label>
                    </div>
                    {mode !== 'author' && comparisonTargets.length > 0 && (
                      <div className="create-run__compare-targets">
                        {comparisonTargets.map((target, index) => (
                          <span key={`${target.promptId}-${target.version.version_id}`} className="badge badge--neutral">
                            compare {index + 1}: {target.promptName} / {target.version.version_id}
                          </span>
                        ))}
                      </div>
                    )}
                    {selectedPromptLoading ? (
                      <div className="create-run__panel-note">Loading saved versions...</div>
                    ) : selectedPromptData ? (
                      <PromptVersionTree
                        promptId={selectedPromptData.prompt.prompt_id}
                        versions={selectedPromptData.versions}
                        activeVersionId={
                          loadedPromptContext?.prompt.prompt_id === selectedPromptData.prompt.prompt_id
                            ? loadedPromptContext.version.version_id
                            : null
                        }
                        compareTargets={versionTreeCompareTargets}
                        draftLeaf={draftLeaf}
                        onLoadVersion={(version) => loadVersionIntoEditor(selectedPromptData.prompt, version)}
                        onToggleCompare={(version) => (
                          mode === 'author'
                            ? toggleEditorCompareTarget(selectedPromptData.prompt.name, version)
                            : toggleCompareTarget(selectedPromptData.prompt.name, version)
                        )}
                      />
                    ) : (
                      <div className="create-run__panel-note">
                        select a saved prompt to inspect versions and load one into the editor.
                      </div>
                    )}
                  </div>
                </div>
              </aside>

              <div className="playground-shell__main">
                {mode === 'author' ? (
                  <div className="create-run__workspace">
                    <div className="create-run__toolbar-stack">
                      <div className="create-run__workspace-header">
                        <div className="create-run__workspace-copy">
                          <label className="field__label">Author mode</label>
                          <span className="field__hint">
                            shape the prompt in components, inspect the resolved prompt text, or compare the current draft against another version.
                          </span>
                        </div>

                        <div className="create-run__workspace-toolbar">
                          <button
                            type="button"
                            ref={(element) => {
                              triggerRefs.current.guided = element;
                            }}
                            className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'guided' ? ' is-active' : ''}`}
                            onClick={() => {
                              setActivePanel((current) => (current === 'guided' ? null : 'guided'));
                            }}
                            aria-pressed={activePanel === 'guided'}
                            aria-expanded={activePanel === 'guided'}
                            aria-controls={activePanel === 'guided' ? activePanelId : undefined}
                          >
                            <img src={iconGuidedStart} alt="" className="create-run__panel-btn-icon" aria-hidden />
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
                            className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'variables' ? ' is-active' : ''}${guidedOverlayStep === 3 ? ' is-guided-highlight' : ''}`}
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
                            className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'tools' ? ' is-active' : ''}${guidedOverlayStep === 4 ? ' is-guided-highlight' : ''}`}
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
                            className={`btn btn--secondary btn--sm create-run__panel-btn${activePanel === 'schema' ? ' is-active' : ''}${guidedOverlayStep === 4 ? ' is-guided-highlight' : ''}`}
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

                      <div className="create-run__workspace-subheader">
                        <div className="create-run__workspace-subheader-main">
                          {workspaceNav}
                        </div>
                        <div className="create-run__workspace-subheader-actions">
                          {editorCompareTarget && (
                            <button
                              type="button"
                              className="btn btn--primary create-run__compare-clear-btn"
                              onClick={() => {
                                setEditorCompareTarget(null);
                                setEditorView('components');
                              }}
                            >
                              <span
                                className="create-run__action-icon"
                                style={getMaskIconStyle(iconTrash)}
                                aria-hidden
                              />
                              clear compare
                            </button>
                          )}
                          <button
                            type="button"
                            className="btn btn--secondary create-run__action-btn"
                            onClick={() => openSaveDialog('current')}
                            disabled={!canSavePrompt}
                          >
                            <span
                              className="create-run__action-icon"
                              style={getMaskIconStyle(iconSave)}
                              aria-hidden
                            />
                            <span>Save prompt</span>
                          </button>
                          <button
                            type="button"
                            ref={(element) => {
                              triggerRefs.current.anchor = element;
                            }}
                            className={`btn btn--secondary create-run__action-btn${activePanel === 'anchor' ? ' is-active' : ''}`}
                            onClick={() => setActivePanel((current) => (current === 'anchor' ? null : 'anchor'))}
                            aria-pressed={activePanel === 'anchor'}
                            aria-expanded={activePanel === 'anchor'}
                            aria-controls={activePanel === 'anchor' ? activePanelId : undefined}
                          >
                            <span
                              className="create-run__action-icon"
                              style={getMaskIconStyle(iconAnchor)}
                              aria-hidden
                            />
                            <span>Anchor output{anchorOutput.trim() ? ' (set)' : ''}</span>
                          </button>
                          {executeRunControl}
                        </div>
                        {!hasResults && (
                          <span id={workspaceModeHintId} className="field__hint create-run__workspace-mode-hint">
                            run the prompt once to unlock analysis.
                          </span>
                        )}
                      </div>

                      {activeSchemaError && activePanel !== 'schema' && (
                        <div className="alert alert--warning create-run__config-alert">
                          <span className="alert__icon">!</span>
                          Output schema needs attention: {activeSchemaError}
                        </div>
                      )}
                    </div>

                    <div className="create-run__workspace-body" ref={workspaceBodyRef}>
                      {editorView === 'components' && (
                        <div className="field">
                          <div className="create-run__field-head">
                            <label className="field__label">Prompt Components</label>
                            <span className="field__hint">
                              {normalizedPromptComponents.filter((component) => component.enabled && component.content.trim()).length} active
                            </span>
                          </div>
                          <PromptComponentEditor
                            components={normalizedPromptComponents}
                            onChange={setPromptComponents}
                            collapsedSections={collapsedSections}
                            onToggleCollapse={toggleCollapsedSection}
                            onCopyComponentContent={copyPromptComponentContent}
                            onReorderComponent={reorderPromptComponent}
                            highlightRequest={editorHighlightRequest}
                            registerSectionRef={(componentKey, element) => {
                              sectionRefs.current[componentKey] = element;
                            }}
                            componentStats={componentStats}
                            componentStatsLabel={selectedGuidedRole?.name ?? null}
                          />
                        </div>
                      )}
                      {editorView === 'resolved' && (
                        <PromptResolvedView resolvedPrompt={currentResolvedPrompt} />
                      )}
                      {editorView === 'diff' && editorCompareTarget && (
                        <PromptDiffWorkspace
                          currentDraft={{
                            label: currentEditorVersionLabel,
                            promptId: currentEditorPromptId,
                            versionId: currentEditorVersionId,
                            revisionNote: revisionNote.trim() || loadedPromptContext?.version.revision_note || null,
                            components: normalizedPromptComponents,
                            variables: snapshotPromptVariables(normalizedPromptComponents, inputVars),
                            tools: normalizedTools,
                            outputSchema: currentOutputSchema,
                            resolvedPrompt: currentResolvedPrompt,
                          }}
                          target={editorCompareTarget}
                          targetResolvedPrompt={editorCompareTargetResolvedPrompt}
                        />
                      )}
                      {editorView === 'diff' && !editorCompareTarget && (
                        <div className="create-run__panel-note">
                          choose a saved version from the history rail to compare it with the current editor state.
                        </div>
                      )}
                    </div>

                    {guidedOverlayStep && selectedGuidedRole && (
                      <GuidedOverlay
                        step={guidedOverlayStep}
                        role={selectedGuidedRole}
                        getAnchor={() => (
                          guidedOverlayStep === 3 ? triggerRefs.current.variables
                          : guidedOverlayStep === 4 ? triggerRefs.current.tools
                          : triggerRefs.current.guided
                        )}
                        onNext={() => setGuidedOverlayStep((current) => (current && current < 4 ? (current + 1) as GuidedOverlayStep : current))}
                        onOpenTools={() => {
                          setGuidedOverlayStep(null);
                          setActivePanel('tools');
                        }}
                        onOpenSchema={() => {
                          setGuidedOverlayStep(null);
                          setActivePanel('schema');
                        }}
                        onDone={() => {
                          setGuidedOverlayStep(null);
                        }}
                      />
                    )}

                  </div>
                ) : (
                  <div className="create-run__workspace">
                    <div className="create-run__toolbar-stack">
                      <div className="create-run__workspace-header">
                        <div className="create-run__workspace-copy">
                          <label className="field__label">Analysis mode</label>
                          <span className="field__hint">
                            inspect results in the same workspace frame and switch to version compare only when you need it.
                          </span>
                        </div>
                      </div>
                      <div className="create-run__workspace-subheader">
                        <div className="create-run__workspace-subheader-main">
                          {workspaceNav}
                        </div>
                        <div className="create-run__workspace-subheader-actions">
                          <button
                            type="button"
                            className="btn btn--secondary create-run__action-btn"
                            onClick={() => openSaveDialog('current')}
                            disabled={!canSavePrompt}
                          >
                            <span
                              className="create-run__action-icon"
                              style={getMaskIconStyle(iconSave)}
                              aria-hidden
                            />
                            <span>Save prompt</span>
                          </button>
                          <button
                            type="button"
                            ref={(element) => {
                              triggerRefs.current.anchor = element;
                            }}
                            className={`btn btn--secondary create-run__action-btn${activePanel === 'anchor' ? ' is-active' : ''}`}
                            onClick={() => setActivePanel((current) => (current === 'anchor' ? null : 'anchor'))}
                            aria-pressed={activePanel === 'anchor'}
                            aria-expanded={activePanel === 'anchor'}
                            aria-controls={activePanel === 'anchor' ? activePanelId : undefined}
                          >
                            <span
                              className="create-run__action-icon"
                              style={getMaskIconStyle(iconAnchor)}
                              aria-hidden
                            />
                            <span>Anchor output{anchorOutput.trim() ? ' (set)' : ''}</span>
                          </button>
                          {executeRunControl}
                        </div>
                      </div>
                    </div>

                    <div className="create-run__workspace-body">
                      {analysisPanel === 'compare' && comparisonTargets.length === 2 ? (
                        <VersionComparisonWorkspace
                          targets={[comparisonTargets[0], comparisonTargets[1]]}
                        />
                      ) : hasResults ? (
                        analysisContent
                      ) : (
                        <div className="card">
                          <div className="empty-state create-run__empty-body">
                            <div className="empty-state__title">No analysis yet</div>
                            <div className="empty-state__desc">
                              run the prompt once to switch into the analysis workspace.
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              {mode === 'author' && (
                <aside className="playground-shell__right">
                  <PromptStructureOutline
                    components={normalizedPromptComponents}
                    tools={normalizedTools}
                    outputSchema={currentOutputSchema}
                    schemaEnabled={schemaEnabled}
                    schemaError={schemaError}
                    detectedVariables={detectedVariables}
                    inputVars={inputVars}
                    onJumpToSection={handleJumpToSection}
                    onJumpToVariable={handleJumpToVariable}
                  />
                </aside>
              )}
            </div>

            {activePanel && (
              <div className="create-run__config-overlay" role="presentation">
                <div
                  className="create-run__config-backdrop"
                  onClick={() => setActivePanel(null)}
                />
                <div
                  id={activePanelId}
                  className={`create-run__config-popover create-run__config-popover--workspace create-run__config-popover--${activePanel}`}
                  role="dialog"
                  aria-modal="true"
                  aria-label={activePanel ? PANEL_TITLES[activePanel] : 'Workspace panel'}
                  aria-describedby={activePanelDescriptionId}
                >
                  <div className="create-run__config-popover-head">
                    <div className="create-run__panel-head-copy">
                      <span className="section-label">workspace panel</span>
                      <div className="create-run__panel-title">{activePanel ? PANEL_TITLES[activePanel] : ''}</div>
                    </div>
                    <button
                      type="button"
                      ref={closeButtonRef}
                      className="icon-btn icon-btn--close"
                      aria-label={`close ${activePanel ? PANEL_TITLES[activePanel] : 'configuration'}`}
                      onClick={() => setActivePanel(null)}
                    >
                      &times;
                    </button>
                  </div>

                  <div className="create-run__config-popover-scroll">
                    <div className="create-run__panel-intro">
                      <div className="create-run__panel-intro-copy">
                        <span className="section-label">{PANEL_SUMMARIES[activePanel].eyebrow}</span>
                        <span id={activePanelDescriptionId} className="field__hint">
                          {PANEL_SUMMARIES[activePanel].description}
                        </span>
                      </div>
                      {activePanelBadges.length > 0 && (
                        <div className="create-run__panel-intro-meta">
                          {activePanelBadges.map((badge) => (
                            <span key={badge} className="badge badge--neutral">
                              {badge}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="create-run__panel-stack">
                    {activePanel === 'guided' && (
                      <div className="create-run__panel-surface">
                        <GuidedPromptStart
                          onClose={() => setActivePanel(null)}
                          onSelectRole={(role) => {
                            setSelectedGuidedRole(role);

                            const components = role.components.map((comp) => ({
                              type: comp.component_type,
                              content: comp.placeholder,
                              enabled: true,
                            }));
                            const editorComponents = preparePromptComponentsForEditor(components);
                            setPromptComponents(editorComponents);
                            setInputVars((current) => snapshotPromptVariables(editorComponents, current));
                            setAppliedTemplateId(`guided-${role.role_id}`);
                            setPromptNameInput((current) => current || `${role.name} prompt`);
                            setCollapsedSections({});
                            setPendingScrollSectionKey(null);
                            setEditorCompareTarget(null);
                            setEditorView('components');

                            setActivePanel(null);
                            setGuidedOverlayStep(2);
                          }}
                        />
                      </div>
                    )}

                    {activePanel === 'model' && (
                      <div className="create-run__panel-surface">
                        <div className="create-run__panel-surface-head">
                          <div className="create-run__panel-surface-copy">
                            <span className="field__label">Execution settings</span>
                            <span className="field__hint">OpenAI provider and model controls stay here, while run count now lives next to execute.</span>
                          </div>
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
                        </div>
                      </div>
                    )}

                    {activePanel === 'variables' && (
                      <div className="create-run__panel-surface">
                        <div className="create-run__panel-surface-head">
                          <div className="create-run__panel-surface-copy">
                            <span className="field__label">Detected variables</span>
                            <span className="field__hint">Rows appear automatically when the prompt includes {'{{variable_name}}'} placeholders.</span>
                          </div>
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
                                  className="textarea textarea--adaptive textarea--code create-run__variables-textarea"
                                  value={inputVars[variableName] ?? ''}
                                  onChange={(e) => {
                                    resizeTextarea(e.target);
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
                      </div>
                    )}

                    {activePanel === 'tools' && (
                      <div className="create-run__panel-surface">
                        <div className="create-run__panel-surface-head">
                          <div className="create-run__panel-surface-copy">
                            <span className="field__label">Tool definitions</span>
                            <span className="field__hint">Longer descriptions expand in place so tool setup stays readable.</span>
                          </div>
                        </div>
                        <PromptToolsEditor
                          tools={tools}
                          onChange={(nextTools) => {
                            setTools(nextTools);
                            setToolError(null);
                          }}
                        />
                        {toolError && <span className="field__error">{toolError}</span>}
                      </div>
                    )}

                    {activePanel === 'schema' && (
                      <div className="create-run__panel-surface">
                        <div className="create-run__panel-surface-head">
                          <div className="create-run__panel-surface-copy">
                            <span className="field__label">Schema fields</span>
                            <span className="field__hint">
                              {schemaSuppressedByTools
                                ? 'Schema enforcement pauses while tools are enabled. The fields stay here for later reuse.'
                                : 'Use schema enforcement when you want structured output checks and comparison-ready fields.'}
                            </span>
                          </div>
                          <label className="check-label create-run__panel-check">
                            <input
                              type="checkbox"
                              checked={schemaEnabled}
                              onChange={(e) => handleSchemaToggle(e.target.checked)}
                            />
                            Enable output schema
                          </label>
                        </div>

                        {schemaEnabled ? (
                          <>
                            {schemaSuppressedByTools && (
                              <div className="create-run__panel-note">
                                tool-enabled runs skip schema enforcement and schema-based result validation.
                              </div>
                            )}
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
                      </div>
                    )}

                    {activePanel === 'anchor' && (
                      <div className="create-run__panel-surface">
                        <div className="create-run__anchor-header">
                          <div className="create-run__panel-surface-copy">
                            <span className="field__label">Reference output</span>
                            <span className="field__hint">Store an example output here when you want to compare future runs against it.</span>
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
                          className="textarea textarea--adaptive textarea--code create-run__panel-textarea create-run__panel-textarea--lg"
                          value={anchorOutput}
                          onChange={(e) => {
                            resizeTextarea(e.target);
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
                      </div>
                    )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {saveDialogOpen && (
              <div className="create-run__config-overlay" role="presentation">
                <div
                  className="create-run__config-backdrop"
                  onClick={() => setSaveDialogOpen(false)}
                />
                <div
                  className="create-run__config-popover create-run__save-popover"
                  role="dialog"
                  aria-modal="true"
                  aria-label="Save prompt"
                >
                  <div className="create-run__config-popover-head">
                    <div className="create-run__panel-title">Save prompt</div>
                    <button
                      type="button"
                      className="icon-btn icon-btn--close"
                      aria-label="close save prompt"
                      onClick={() => setSaveDialogOpen(false)}
                    >
                      &times;
                    </button>
                  </div>

                  {promptWorkflow === 'existing' && loadedPromptContext && (
                    <div className="seg-control playground-seg-control" role="group" aria-label="Save destination">
                      <button
                        type="button"
                        className={`seg-control__btn${saveIntent === 'current' ? ' is-active' : ''}`}
                        onClick={() => setSaveIntent('current')}
                        aria-pressed={saveIntent === 'current'}
                      >
                        <span
                          className="playground-seg-control__icon"
                          style={getMaskIconStyle(iconSave)}
                          aria-hidden
                        />
                        <span>Save new version</span>
                      </button>
                      <button
                        type="button"
                        className={`seg-control__btn${saveIntent === 'new' ? ' is-active' : ''}`}
                        onClick={() => setSaveIntent('new')}
                        aria-pressed={saveIntent === 'new'}
                      >
                        <span
                          className="playground-seg-control__icon"
                          style={getMaskIconStyle(iconCreateNewPrompt)}
                          aria-hidden
                        />
                        <span>Save as new prompt</span>
                      </button>
                    </div>
                  )}

                  {saveIntent === 'new' || promptWorkflow === 'new' ? (
                    <div className="field">
                      <label className="field__label" htmlFor="playground-save-prompt-name">
                        Prompt name
                      </label>
                      <input
                        id="playground-save-prompt-name"
                        ref={saveFieldRef}
                        className="input"
                        value={promptNameInput}
                        onChange={(e) => setPromptNameInput(e.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
                            event.preventDefault();
                            handleSavePrompt(saveIntent);
                          }
                        }}
                        placeholder="Planner agent prompt"
                      />
                      <span className="field__hint">
                        a prompt id will be generated automatically when you save.
                      </span>
                    </div>
                  ) : loadedPromptContext ? (
                    <div className="playground-save__summary">
                      <div className="playground-save__summary-row">
                        <span className="badge badge--primary">{loadedPromptContext.prompt.name}</span>
                        <span className="badge badge--neutral">{loadedPromptContext.version.version_id}</span>
                      </div>
                      <div className="field__hint">{loadedPromptContext.prompt.prompt_id}</div>
                    </div>
                  ) : null}

                  <div className="field">
                    <label className="field__label" htmlFor="prompt-revision-note">
                      Revision note
                    </label>
                    <input
                      id="prompt-revision-note"
                      ref={saveIntent === 'new' || promptWorkflow === 'new' ? null : saveFieldRef}
                      className="input"
                      value={revisionNote}
                      onChange={(e) => setRevisionNote(e.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
                          event.preventDefault();
                          handleSavePrompt(saveIntent);
                        }
                      }}
                      placeholder="what changed in this revision?"
                    />
                  </div>

                  {toolError && (
                    <div className="alert alert--danger">
                      <span className="alert__icon">!</span>
                      {toolError}
                    </div>
                  )}

                  <div className="playground-save__actions">
                    <button
                      type="button"
                      className="btn btn--secondary"
                      disabled={saveLoading}
                      onClick={() => setSaveDialogOpen(false)}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      ref={saveActionRef}
                      className="btn btn--primary"
                      disabled={saveLoading}
                      onClick={() => handleSavePrompt(saveIntent)}
                    >
                      {saveLoading ? 'Saving...' : 'Save prompt'}
                    </button>
                  </div>
                </div>
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
