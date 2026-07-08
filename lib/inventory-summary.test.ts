import { describe, it, expect } from "vitest";
import { summarizeInventory } from "./inventory-summary";

describe("summarizeInventory", () => {
  it("enriches all 50 parts and counts sum to 50", () => {
    const { parts, counts } = summarizeInventory();
    expect(parts.length).toBe(50);
    expect(counts.critical + counts.warning + counts.ok).toBe(50);
  });

  it("enriches each part with daysOfSupply and a valid risk", () => {
    const { parts } = summarizeInventory();
    for (const p of parts) {
      expect(typeof p.daysOfSupply).toBe("number");
      expect(["CRITICAL", "WARNING", "OK"]).toContain(p.risk);
    }
  });
});
