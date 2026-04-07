import type { PromptComponentType } from "./prompt";

export interface GuidedStartRoleComponent {
  component_type: PromptComponentType;
  prevalence: number;
  placeholder: string;
}

export interface GuidedStartRole {
  role_id: string;
  name: string;
  summary: string;
  sample_size: number;
  components: GuidedStartRoleComponent[];
}

export interface GuidedStartCatalog {
  version: string;
  generated_at: string;
  roles: GuidedStartRole[];
}
