import Ajv from "ajv";
import type { ErrorObject } from "ajv";
import type { JsonSchema } from "../types/schema";
import type { TraceEvent } from "../types/trace";

export type OutputSchemaValidationState = "valid" | "invalid" | "missing";

export interface OutputSchemaValidationResult {
  key: string;
  state: OutputSchemaValidationState;
}

export interface ResolvedOutputCandidate {
  value: unknown;
  eventId?: string;
}

interface OutputCandidateScore {
  invalidCount: number;
  missingCount: number;
  extraTopLevelCount: number;
  totalTopLevelCount: number;
}

const ajv = new Ajv({ allErrors: false, strict: false });

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function getSchemaProperties(schema?: JsonSchema): Record<string, JsonSchema> {
  if (!schema || !isRecord(schema.properties)) {
    return {};
  }

  const properties: Record<string, JsonSchema> = {};

  for (const [key, value] of Object.entries(schema.properties)) {
    if (isRecord(value)) {
      properties[key] = value;
    }
  }

  return properties;
}

function getRequiredKeys(schema?: JsonSchema): Set<string> {
  if (!Array.isArray(schema?.required)) {
    return new Set();
  }

  return new Set(schema.required.filter((key): key is string => typeof key === "string"));
}

function parseJsonOutput(llmOutput: string): unknown {
  try {
    return JSON.parse(llmOutput);
  } catch {
    return undefined;
  }
}

function compileSchema(schema: JsonSchema) {
  try {
    return ajv.compile(schema);
  } catch {
    return null;
  }
}

function normalizeOutputValue(outputValue: unknown): unknown {
  if (typeof outputValue !== "string") {
    return outputValue;
  }

  return parseJsonOutput(outputValue);
}

function getTopLevelKey(instancePath: string): string | undefined {
  const segments = instancePath.split("/").filter(Boolean);
  return segments[0];
}

function pushUniqueResult(
  results: OutputSchemaValidationResult[],
  key: string,
  state: OutputSchemaValidationState,
) {
  if (results.some((result) => result.key === key)) {
    return;
  }

  results.push({ key, state });
}

function applyAjvErrors(
  errors: ErrorObject[] | null | undefined,
  baseStates: Map<string, OutputSchemaValidationState>,
): OutputSchemaValidationResult[] {
  const extraResults: OutputSchemaValidationResult[] = [];
  let hasGlobalError = false;

  for (const error of errors ?? []) {
    if (error.keyword === "required" && typeof error.params.missingProperty === "string") {
      const topLevelKey = getTopLevelKey(error.instancePath);
      if (topLevelKey && baseStates.has(topLevelKey)) {
        baseStates.set(topLevelKey, "invalid");
      } else {
        baseStates.set(error.params.missingProperty, "invalid");
      }
      continue;
    }

    if (error.keyword === "additionalProperties" && typeof error.params.additionalProperty === "string") {
      const topLevelKey = getTopLevelKey(error.instancePath);
      if (topLevelKey && baseStates.has(topLevelKey)) {
        baseStates.set(topLevelKey, "invalid");
      } else {
        pushUniqueResult(extraResults, `extra.${error.params.additionalProperty}`, "invalid");
      }
      continue;
    }

    const topLevelKey = getTopLevelKey(error.instancePath);
    if (topLevelKey && baseStates.has(topLevelKey)) {
      baseStates.set(topLevelKey, "invalid");
      continue;
    }

    hasGlobalError = true;
  }

  if (hasGlobalError) {
    pushUniqueResult(extraResults, "$schema", "invalid");
  }

  return extraResults;
}

export function hasOutputSchemaProperties(schema?: JsonSchema): boolean {
  return Object.keys(getSchemaProperties(schema)).length > 0;
}

export function validateOutputAgainstSchema(
  outputValue: unknown,
  schema: JsonSchema,
): OutputSchemaValidationResult[] {
  const properties = getSchemaProperties(schema);
  const keys = Object.keys(properties);

  if (keys.length === 0) {
    return [];
  }

  const parsedOutput = normalizeOutputValue(outputValue);
  if (!isRecord(parsedOutput)) {
    return [
      ...keys.map((key) => ({ key, state: "invalid" as const })),
      { key: "$schema", state: "invalid" as const },
    ];
  }

  const requiredKeys = getRequiredKeys(schema);
  const validate = compileSchema(schema);
  if (!validate) {
    return [
      ...keys.map((key) => ({
        key,
        state: "invalid" as const,
      })),
      { key: "$schema", state: "invalid" as const },
    ];
  }

  validate(parsedOutput);

  const baseStates = new Map<string, OutputSchemaValidationState>(
    keys.map((key) => {
      if (!Object.prototype.hasOwnProperty.call(parsedOutput, key)) {
        return [key, requiredKeys.has(key) ? "invalid" : "missing"];
      }

      return [key, "valid"];
    }),
  );

  const extraResults = applyAjvErrors(validate.errors, baseStates);
  return [
    ...keys.map((key) => ({
      key,
      state: baseStates.get(key) ?? "missing",
    })),
    ...extraResults,
  ];
}

function scoreOutputCandidate(outputValue: unknown, schema: JsonSchema): OutputCandidateScore {
  const results = validateOutputAgainstSchema(outputValue, schema);
  const parsedOutput = normalizeOutputValue(outputValue);
  const schemaProperties = getSchemaProperties(schema);
  const outputKeys = isRecord(parsedOutput) ? Object.keys(parsedOutput) : [];
  const extraTopLevelCount = outputKeys.filter((key) => !(key in schemaProperties)).length;
  return {
    invalidCount: results.filter((result) => result.state === "invalid").length,
    missingCount: results.filter((result) => result.state === "missing").length,
    extraTopLevelCount,
    totalTopLevelCount: outputKeys.length,
  };
}

function isBetterCandidateScore(next: OutputCandidateScore, best: OutputCandidateScore): boolean {
  if (next.invalidCount !== best.invalidCount) {
    return next.invalidCount < best.invalidCount;
  }

  if (next.missingCount !== best.missingCount) {
    return next.missingCount < best.missingCount;
  }

  if (next.extraTopLevelCount !== best.extraTopLevelCount) {
    return next.extraTopLevelCount < best.extraTopLevelCount;
  }

  if (next.totalTopLevelCount !== best.totalTopLevelCount) {
    return next.totalTopLevelCount < best.totalTopLevelCount;
  }

  return false;
}

function getEventOutputCandidates(event: TraceEvent): ResolvedOutputCandidate[] {
  if (event.event_type === "on_chain_end") {
    return [{ value: event.payload?.outputs, eventId: event.event_id }];
  }

  if (event.event_type === "on_llm_end") {
    return [{ value: event.payload?.output_text ?? event.payload?.output, eventId: event.event_id }];
  }

  return [];
}

export function resolveOutputCandidateForSchema(
  events: TraceEvent[] | undefined,
  schema: JsonSchema,
  fallbackOutput?: unknown,
): ResolvedOutputCandidate {
  const candidates: ResolvedOutputCandidate[] = [];

  for (const event of [...(events ?? [])].reverse()) {
    for (const candidate of getEventOutputCandidates(event)) {
      if (candidate.value != null) {
        candidates.push(candidate);
      }
    }
  }

  if (fallbackOutput != null) {
    candidates.push({ value: fallbackOutput });
  }

  if (candidates.length === 0) {
    return { value: fallbackOutput };
  }

  let bestCandidate = candidates[0];
  let bestScore = scoreOutputCandidate(bestCandidate.value, schema);

  for (const candidate of candidates.slice(1)) {
    const nextScore = scoreOutputCandidate(candidate.value, schema);
    if (isBetterCandidateScore(nextScore, bestScore)) {
      bestCandidate = candidate;
      bestScore = nextScore;
    }
  }

  return bestCandidate;
}

export function resolveOutputValueForSchema(
  events: TraceEvent[] | undefined,
  schema: JsonSchema,
  fallbackOutput?: unknown,
): unknown {
  return resolveOutputCandidateForSchema(events, schema, fallbackOutput).value;
}
