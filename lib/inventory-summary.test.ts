import { describe, it, expect } from "vitest";
import { summarizeInventory } from "./inventory-summary";

describe("summarizeInventory", () => {
  it("enriches every part and risk counts partition the total", () => {
    const { parts, counts } = summarizeInventory();
    expect(parts.length).toBeGreaterThan(0);
    expect(counts.critical + counts.warning + counts.ok).toBe(parts.length);
  });

  it("enriches each part with daysOfSupply and a valid risk", () => {
    const { parts } = summarizeInventory();
    for (const p of parts) {
      expect(typeof p.daysOfSupply).toBe("number");
      expect(["CRITICAL", "WARNING", "OK"]).toContain(p.risk);
    }
  });
});
