import { describe, it, expect } from "vitest";
import { getInventoryStatus } from "./inventory";

describe("getInventoryStatus", () => {
  it("returns details for a known part", () => {
    const out = getInventoryStatus({ partId: "PART_001" });
    expect(out).toContain("PART_001");
    expect(out).toMatch(/Risk: (CRITICAL|WARNING|OK)/);
  });
  it("handles unknown part", () => {
    expect(getInventoryStatus({ partId: "NOPE" })).toContain("not found");
  });
  it("lists at-risk parts when no partId", () => {
    const out = getInventoryStatus({});
    expect(out.length).toBeGreaterThan(0);
  });
});
