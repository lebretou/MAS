import { describe, expect, it } from "vitest";
import type { JsonSchema } from "../types/schema";
import type { TraceEvent } from "../types/trace";
import {
  resolveOutputCandidateForSchema,
  resolveOutputValueForSchema,
  validateOutputAgainstSchema,
} from "./schema-validation";

const sampleSchema: JsonSchema = {
  type: "object",
  required: ["answer", "confidence"],
  additionalProperties: false,
  properties: {
    answer: { type: "string" },
    confidence: { type: "number" },
    notes: { type: "string" },
  },
};

const nestedSchema: JsonSchema = {
  type: "object",
  required: ["analysis"],
  additionalProperties: false,
  properties: {
    analysis: {
      type: "object",
      required: ["findings"],
      additionalProperties: false,
      properties: {
        findings: {
          type: "array",
          items: { type: "string" },
        },
      },
    },
  },
};

function makeEvent(overrides: Partial<TraceEvent>): TraceEvent {
  return {
    event_id: overrides.event_id ?? crypto.randomUUID(),
    trace_id: overrides.trace_id ?? "trace-1",
    execution_id: overrides.execution_id ?? "exec-1",
    timestamp: overrides.timestamp ?? new Date().toISOString(),
    sequence: overrides.sequence ?? 0,
    event_type: overrides.event_type ?? "on_chain_start",
    refs: overrides.refs ?? {},
    payload: overrides.payload ?? {},
    agent_id: overrides.agent_id ?? null,
    span_id: overrides.span_id ?? null,
    parent_span_id: overrides.parent_span_id ?? null,
  };
}

describe("validateOutputAgainstSchema", () => {
  it("marks matching properties as valid", () => {
    const results = validateOutputAgainstSchema(
      {
        answer: "done",
        confidence: 0.82,
      },
      sampleSchema,
    );

    expect(results).toEqual([
      { key: "answer", state: "valid" },
      { key: "confidence", state: "valid" },
      { key: "notes", state: "missing" },
    ]);
  });

  it("marks missing required properties as invalid", () => {
    const results = validateOutputAgainstSchema(
      {
        answer: "done",
      },
      sampleSchema,
    );

    expect(results).toEqual([
      { key: "answer", state: "valid" },
      { key: "confidence", state: "invalid" },
      { key: "notes", state: "missing" },
    ]);
  });

  it("surfaces unexpected properties when additionalProperties is false", () => {
    const results = validateOutputAgainstSchema(
      {
        answer: "done",
        confidence: 0.82,
        extra: true,
      },
      sampleSchema,
    );

    expect(results).toEqual([
      { key: "answer", state: "valid" },
      { key: "confidence", state: "valid" },
      { key: "notes", state: "missing" },
      { key: "extra.extra", state: "invalid" },
    ]);
  });

  it("treats invalid json strings as schema failures", () => {
    const results = validateOutputAgainstSchema('{"answer":"done"', sampleSchema);

    expect(results).toEqual([
      { key: "answer", state: "invalid" },
      { key: "confidence", state: "invalid" },
      { key: "notes", state: "invalid" },
      { key: "$schema", state: "invalid" },
    ]);
  });

  it("marks a top-level field invalid when nested required fields are missing", () => {
    const results = validateOutputAgainstSchema(
      {
        analysis: {},
      },
      nestedSchema,
    );

    expect(results).toEqual([
      { key: "analysis", state: "invalid" },
    ]);
  });

  it("marks a top-level field invalid when nested additional properties exist", () => {
    const results = validateOutputAgainstSchema(
      {
        analysis: {
          findings: ["done"],
          extra: true,
        },
      },
      nestedSchema,
    );

    expect(results).toEqual([
      { key: "analysis", state: "invalid" },
    ]);
  });

  it("fails safely when the saved schema cannot compile", () => {
    const invalidSchema: JsonSchema = {
      type: "object",
      properties: {
        answer: { $ref: "missing-schema.json#/answer" },
      },
    };

    const results = validateOutputAgainstSchema(
      {
        answer: "done",
      },
      invalidSchema,
    );

    expect(results).toEqual([
      { key: "answer", state: "invalid" },
      { key: "$schema", state: "invalid" },
    ]);
  });

  it("prefers the best schema-matching trace output candidate", () => {
    const decisionSchema: JsonSchema = {
      type: "object",
      required: ["decision", "response", "reasoning", "dataset_observations"],
      additionalProperties: false,
      properties: {
        decision: { type: "string" },
        response: { type: "string" },
        reasoning: { type: "string" },
        dataset_observations: {
          type: "array",
          items: { type: "string" },
        },
      },
    };
    const validOutput = {
      decision: "relevant",
      response: "Use age and annual_salary_usd.",
      reasoning: "The query requires analysis.",
      dataset_observations: ["age is numeric", "annual_salary_usd is numeric"],
    };
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "llm-end",
        event_type: "on_llm_end",
        payload: {
          output_text: "{\"decision\":\"relevant\",\"response\":\"truncated",
        },
      }),
      makeEvent({
        event_id: "structured-output",
        event_type: "on_chain_end",
        payload: {
          outputs: validOutput,
        },
      }),
      makeEvent({
        event_id: "final-state",
        event_type: "on_chain_end",
        payload: {
          outputs: {
            dataset: "sample",
            next_agent: "planner",
            ...validOutput,
          },
        },
      }),
    ];

    expect(resolveOutputValueForSchema(events, decisionSchema)).toEqual(validOutput);
  });

  it("prefers the more specific candidate for permissive schemas", () => {
    const permissiveSchema: JsonSchema = {
      type: "object",
      required: ["decision", "response"],
      properties: {
        decision: { type: "string" },
        response: { type: "string" },
      },
    };
    const validOutput = {
      decision: "relevant",
      response: "Use age and annual_salary_usd.",
    };
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "specific-output",
        event_type: "on_chain_end",
        payload: {
          outputs: validOutput,
        },
      }),
      makeEvent({
        event_id: "broader-state",
        event_type: "on_chain_end",
        payload: {
          outputs: {
            dataset: "sample",
            next_agent: "planner",
            ...validOutput,
          },
        },
      }),
    ];

    expect(resolveOutputValueForSchema(events, permissiveSchema)).toEqual(validOutput);
  });

  it("returns the source event id for the resolved candidate", () => {
    const decisionSchema: JsonSchema = {
      type: "object",
      required: ["decision", "response"],
      additionalProperties: false,
      properties: {
        decision: { type: "string" },
        response: { type: "string" },
      },
    };
    const validOutput = {
      decision: "relevant",
      response: "ok",
    };
    const events: TraceEvent[] = [
      makeEvent({
        event_id: "llm-end",
        event_type: "on_llm_end",
        payload: {
          output_text: "{\"decision\":\"relevant\",\"response\":\"truncated",
        },
      }),
      makeEvent({
        event_id: "state-update",
        event_type: "on_chain_end",
        payload: {
          outputs: validOutput,
        },
      }),
    ];

    expect(resolveOutputCandidateForSchema(events, decisionSchema)).toEqual({
      value: validOutput,
      eventId: "state-update",
    });
  });
});
