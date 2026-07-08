import { describe, it, expect } from "vitest";
import { searchKnowledge } from "./knowledge";

describe("searchKnowledge", () => {
  it("finds a relevant doc for a policy query", () => {
    const out = searchKnowledge("reorder policy safety stock");
    expect(out.length).toBeGreaterThan(0);
    expect(out).not.toContain("No relevant");
  });
  it("returns a no-match message for gibberish", () => {
    expect(searchKnowledge("zzzz qqqq")).toContain("No relevant");
  });
});
