import type { PromptComponent, PromptComponentType, PromptTool } from "./prompt";

export type GuidedStartStage = "role" | "questions" | "tools" | "schema" | "review";

export interface GuidedStartQuestion {
  question_id: string;
  label: string;
  description?: string | null;
  input_type: string;
  required: boolean;
  placeholder?: string | null;
  default_value?: string;
}

export interface GuidedStartSuggestedComponent {
  component_type: PromptComponentType;
  title: string;
  prevalence: number;
  order_rank: number;
  content_template: string;
}

export interface GuidedStartArchetype {
  archetype_id: string;
  title: string;
  summary: string;
  example_jobs: string[];
  sample_size: number;
  starter_questions: GuidedStartQuestion[];
  suggested_components: GuidedStartSuggestedComponent[];
  suggested_tools: PromptTool[];
  suggested_output_schema?: Record<string, unknown> | null;
}

export interface GuidedStartCatalog {
  version: string;
  generated_at: string;
  fallback_questions: GuidedStartQuestion[];
  fallback_components: GuidedStartSuggestedComponent[];
  archetypes: GuidedStartArchetype[];
}

export interface GuidedStartConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export interface GuidedStartLlmRequest {
  provider: string;
  model: string;
  temperature: number;
  stage: GuidedStartStage;
  selected_archetype?: string | null;
  custom_role?: string | null;
  answers: Record<string, string>;
  current_draft: PromptComponent[];
  conversation_history: GuidedStartConversationTurn[];
  latest_user_turn: string;
}

export interface GuidedStartLlmResponse {
  assistant_message: string;
  selected_archetype?: string | null;
  selected_archetype_title?: string | null;
  component_draft: PromptComponent[];
  current_stage: GuidedStartStage;
  follow_up_questions: string[];
  stage_complete: boolean;
  status: "needs_input" | "ready_for_next_stage" | "ready_to_apply";
  updated_component_types: PromptComponentType[];
}
