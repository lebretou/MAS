import axios, { AxiosResponse } from 'axios';
import { PlaygroundRun, PlaygroundRunCreate, PlaygroundRunResponse } from '../types/playground';
import { 
  Prompt, 
  PromptVersion, 
  CreatePromptRequest, 
  CreateVersionRequest, 
} from '../types/prompt';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const playgroundAPI = {

    createRun: async (data: PlaygroundRunCreate): Promise<PlaygroundRun> => {
        const response: AxiosResponse<PlaygroundRunResponse> = await api.post('/api/playground/run/', data);
        return response.data.run;
    },

    getAllRuns: async (): Promise<PlaygroundRun[]> => {
        const response: AxiosResponse<PlaygroundRun[]> = await api.get('/api/playground/runs/');
        return response.data;
    },

    getRun: async (runId: string): Promise<PlaygroundRun> => {
        const response: AxiosResponse<PlaygroundRun> = await api.get(`/api/playground/runs/${runId}/`);
        return response.data;
    },

    getRunsByPrompt: async (promptId: string): Promise<PlaygroundRun[]> => {
        const response: AxiosResponse<PlaygroundRun[]> = await api.get(`/api/playground/runs/?prompt_id=${promptId}`);
        return response.data;
    }
};

export const promptAPI = {

    createPrompt: async (data: CreatePromptRequest): Promise<Prompt> => {
    const response: AxiosResponse<Prompt> = await api.post('/api/prompts', data);
    return response.data;
  },

  createVersion: async (promptId: string, data: CreateVersionRequest): Promise<PromptVersion> => {
    const response: AxiosResponse<PromptVersion> = await api.post(`/api/prompts/${promptId}/versions`, data);
    return response.data;
  },

  getAllPrompts: async (): Promise<Prompt[]> => {
    const response: AxiosResponse<Prompt[]> = await api.get('/api/prompts');
    return response.data;
  },

  getPrompt: async (promptId: string): Promise<Prompt> => {
    const response: AxiosResponse<Prompt> = await api.get(`/api/prompts/${promptId}`);
    return response.data;
  },

  getLatestVersion: async (promptId: string): Promise<PromptVersion> => {
    const response: AxiosResponse<PromptVersion> = await api.get(`/api/prompts/${promptId}/latest`);
    return response.data;
  },
};