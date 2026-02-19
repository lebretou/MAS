export interface PlaygroundRun {

    // identifiers
    run_id: string;
    created_at: string;

    //prompt reference (which prompt version used)
    prompt_id: string;
    version_id: string;
    
    // model config
    model: string;
    provider: string;
    temperature: number;
    max_tokens: number | null;

    input_variables: Record<string, string>;
    resolved_prompt: string;
    output: string; // LLM response

    latency_ms: number | null;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;

    model_config_id: string | null;

    created_by: string | null;
    tags: string[] | null;
    notes: string | null;
}

export interface PlaygroundRunCreate {

    model_config_id?: string | null;
    
    prompt_id: string;
    version_id?: string;
    input_variables?: Record<string, string>;

    // inline model config (used if model_config_id is None)
    model?: string;
    provider?: string;
    temperature?: number;
    max_tokens?: number | null;

    // optional metadata
    tags?: string[] | null;
    notes?: string | null;

}

export interface PlaygroundRunResponse {
    run: PlaygroundRun;
    message: string;
}
