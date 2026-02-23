import client from "./client";
import type { Prompt, PromptVersion, CreatePromptRequest, CreateVersionRequest } from "../types/prompt";

export async function fetchPrompts(): Promise<Prompt[]> {
  const { data } = await client.get<Prompt[]>("/prompts");
  return data;
}

export async function fetchPrompt(promptId: string): Promise<Prompt> {
  const { data } = await client.get<Prompt>(`/prompts/${promptId}`);
  return data;
}

export async function fetchLatestVersion(promptId: string): Promise<PromptVersion> {
  const { data } = await client.get<PromptVersion>(`/prompts/${promptId}/latest`);
  return data;
}

export async function createPrompt(req: CreatePromptRequest): Promise<Prompt> {
  const { data } = await client.post<Prompt>("/prompts", req);
  return data;
}

export async function createVersion(promptId: string, req: CreateVersionRequest): Promise<PromptVersion> {
  const { data } = await client.post<PromptVersion>(`/prompts/${promptId}/versions`, req);
  return data;
}
