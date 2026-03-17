import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  getMock: vi.fn(),
}));

vi.mock("./client", () => ({
  default: {
    get: mocks.getMock,
  },
}));

import { fetchTraceEvents, fetchTraces, fetchTraceSummary } from "./traces";

describe("traces api", () => {
  beforeEach(() => {
    mocks.getMock.mockReset();
  });

  it("fetches traces with pagination params", async () => {
    const traces = [{ trace_id: "trace-1" }];
    mocks.getMock.mockResolvedValue({ data: traces });

    const result = await fetchTraces(50, 10);

    expect(mocks.getMock).toHaveBeenCalledWith("/traces", {
      params: { limit: 50, offset: 10 },
    });
    expect(result).toEqual(traces);
  });

  it("fetches trace events by trace id", async () => {
    const events = [{ event_id: "event-1", event_type: "on_chain_start" }];
    mocks.getMock.mockResolvedValue({ data: events });

    const result = await fetchTraceEvents("trace-1");

    expect(mocks.getMock).toHaveBeenCalledWith("/traces/trace-1");
    expect(result).toEqual(events);
  });

  it("fetches a trace summary by trace id", async () => {
    const summary = { trace_id: "trace-1", agents: ["planner"] };
    mocks.getMock.mockResolvedValue({ data: summary });

    const result = await fetchTraceSummary("trace-1");

    expect(mocks.getMock).toHaveBeenCalledWith("/traces/trace-1/summary");
    expect(result).toEqual(summary);
  });
});
