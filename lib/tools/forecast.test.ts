import { describe, it, expect } from "vitest";
import { getForecast } from "./forecast";

describe("getForecast", () => {
  it("returns ordered quantiles for a known part", () => {
    const f = getForecast("PART_001");
    expect(f.p10).toBeLessThanOrEqual(f.p50);
    expect(f.p50).toBeLessThanOrEqual(f.p90);
    expect(f.p10).toBeGreaterThanOrEqual(0);
    expect(["TFT model", "statistical baseline"]).toContain(f.source);
  });
  it("handles unknown part", () => {
    expect(getForecast("NOPE").text).toContain("No data");
  });
});
