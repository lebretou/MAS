import client from "../api/client";
import type { GuidedStartCatalog } from "../types/guidedStart";
import type {
  PlaygroundAnalysisRequest,
  PlaygroundAnalysisResponse,
  PlaygroundRun,
  PlaygroundRunCreate,
  PlaygroundRunResponse,
} from "../types/playground";
import type {
  Prompt,
  PromptListItem,
  PromptTemplate,
  PromptVersion,
  PromptWithVersions,
  CreatePromptRequest,
  CreateVersionRequest,
} from "../types/prompt";

export const playgroundAPI = {
  async createRun(data: PlaygroundRunCreate): Promise<PlaygroundRun> {
    const { data: response } = await client.post<PlaygroundRunResponse>("/playground/run", data);
    return response.run;
  },

  async getAllRuns(): Promise<PlaygroundRun[]> {
    const { data } = await client.get<PlaygroundRun[]>("/playground/runs");
    return data;
  },

  async getRun(runId: string): Promise<PlaygroundRun> {
    const { data } = await client.get<PlaygroundRun>(`/playground/runs/${runId}`);
    return data;
  },

  async getRunsByPrompt(promptId: string): Promise<PlaygroundRun[]> {
    const { data } = await client.get<PlaygroundRun[]>("/playground/runs", {
      params: { prompt_id: promptId },
    });
    return data;
  },

  async analyzeOutputs(data: PlaygroundAnalysisRequest): Promise<PlaygroundAnalysisResponse> {
    const { data: response } = await client.post<PlaygroundAnalysisResponse>("/playground/analyze", data);
    return response;
  },
};

export const promptAPI = {
  async createPrompt(data: CreatePromptRequest): Promise<Prompt> {
    const { data: response } = await client.post<Prompt>("/prompts", data);
    return response;
  },

  async updatePrompt(promptId: string, data: { name: string; description?: string | null }): Promise<Prompt> {
    const { data: response } = await client.patch<Prompt>(`/prompts/${promptId}`, data);
    return response;
  },

  async deletePrompt(promptId: string): Promise<void> {
    await client.delete(`/prompts/${promptId}`);
  },

  async createVersion(promptId: string, data: CreateVersionRequest): Promise<PromptVersion> {
    const { data: response } = await client.post<PromptVersion>(`/prompts/${promptId}/versions`, data);
    return response;
  },

  async getAllPrompts(): Promise<PromptListItem[]> {
    const { data } = await client.get<PromptListItem[]>("/prompts");
    return data;
  },

  async getPrompt(promptId: string): Promise<PromptWithVersions> {
    const { data } = await client.get<PromptWithVersions>(`/prompts/${promptId}`);
    return data;
  },

  async getLatestVersion(promptId: string): Promise<PromptVersion> {
    const { data } = await client.get<PromptVersion>(`/prompts/${promptId}/latest`);
    return data;
  },

  async getPromptTemplates(): Promise<PromptTemplate[]> {
    const { data } = await client.get<PromptTemplate[]>("/prompt-templates");
    return data;
  },

  async duplicatePrompt(
    sourcePromptId: string,
    newPromptId: string,
    newName: string,
  ): Promise<{ prompt: Prompt; version: PromptVersion }> {
    const source = await promptAPI.getPrompt(sourcePromptId);
    const latestVersion = source.versions.length > 0
      ? source.versions.reduce((a, b) => a.created_at > b.created_at ? a : b)
      : null;

    const prompt = await promptAPI.createPrompt({
      prompt_id: newPromptId,
      name: newName,
      description: source.prompt.description,
    });

    let version: PromptVersion | null = null;
    if (latestVersion) {
      version = await promptAPI.createVersion(newPromptId, {
        name: latestVersion.name,
        components: latestVersion.components,
        variables: latestVersion.variables,
        output_schema: latestVersion.output_schema,
        tools: latestVersion.tools,
        revision_note: `duplicated from ${sourcePromptId}/${latestVersion.version_id}`,
      });
    }

    return { prompt, version: version! };
  },
};

export const guidedStartAPI = {
  async getCatalog(): Promise<GuidedStartCatalog> {
    const { data } = await client.get<GuidedStartCatalog>("/guided-start/catalog");
    return data;
  },
};