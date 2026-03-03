import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  getMock: vi.fn(),
}));

vi.mock("./client", () => ({
  default: {
    get: mocks.getMock,
  },
}));

import { fetchCognitionCached } from "./traces";

describe("fetchCognitionCached", () => {
  it("deduplicates in-flight requests", async () => {
    mocks.getMock.mockResolvedValue({
      data: {
        agent_id: "interaction",
        summary: "[llm] done",
        hint: "[llm] done",
        cached: false,
      },
    

  it("fetches fresh cognition when forceRefresh is true", async () => {
    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] stale",
        hint: "[llm] stale",
        cached: true,
      },
    });

    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] fresh",
        hint: "[llm] fresh",
        cached: false,
      },
    });

    const first = await fetchCognitionCached("trace-refresh", "interaction");
    const second = await fetchCognitionCached("trace-refresh", "interaction", { forceRefresh: true });

    expect(first.summary).toBe("[llm] stale");
    expect(second.summary).toBe("[llm] fresh");
    expect(mocks.getMock).toHaveBeenCalledTimes(2);
  });
});

    const p1 = fetchCognitionCached("trace-1", "interaction");
    const p2 = fetchCognitionCached("trace-1", "interaction");
    const [r1, r2] = await Promise.all([p1, p2]);

    expect(mocks.getMock).toHaveBeenCalledTimes(1);
    expect(r1.hint).toBe("[llm] done");
    expect(r2.hint).toBe("[llm] done");
  

  it("fetches fresh cognition when forceRefresh is true", async () => {
    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] stale",
        hint: "[llm] stale",
        cached: true,
      },
    });

    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] fresh",
        hint: "[llm] fresh",
        cached: false,
      },
    });

    const first = await fetchCognitionCached("trace-refresh", "interaction");
    const second = await fetchCognitionCached("trace-refresh", "interaction", { forceRefresh: true });

    expect(first.summary).toBe("[llm] stale");
    expect(second.summary).toBe("[llm] fresh");
    expect(mocks.getMock).toHaveBeenCalledTimes(2);
  });
});


  it("fetches fresh cognition when forceRefresh is true", async () => {
    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] stale",
        hint: "[llm] stale",
        cached: true,
      },
    });

    mocks.getMock.mockResolvedValueOnce({
      data: {
        agent_id: "interaction",
        summary: "[llm] fresh",
        hint: "[llm] fresh",
        cached: false,
      },
    });

    const first = await fetchCognitionCached("trace-refresh", "interaction");
    const second = await fetchCognitionCached("trace-refresh", "interaction", { forceRefresh: true });

    expect(first.summary).toBe("[llm] stale");
    expect(second.summary).toBe("[llm] fresh");
    expect(mocks.getMock).toHaveBeenCalledTimes(2);
  });
});
