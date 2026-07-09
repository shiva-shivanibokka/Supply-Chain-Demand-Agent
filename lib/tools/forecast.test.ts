import { describe, it, expect } from "vitest";
import { getForecast } from "./forecast";

describe("getForecast", () => {
  it("returns ordered quantiles for a known part", () => {
    const f = getForecast("PART_001");
    expect(f.p10).toBeLessThanOrEqual(f.p50);
    expect(f.p50).toBeLessThanOrEqual(f.p90);
    expect(f.p10).toBeGreaterThanOrEqual(0);
    expect(["TFT model", "statistical baseline"]).toContain(f.source);
    // Daily series drives the chart — must be a non-empty, equal-length series.
    expect(f.daily.p50.length).toBeGreaterThan(0);
    expect(f.daily.p10.length).toBe(f.daily.p50.length);
    expect(f.daily.p90.length).toBe(f.daily.p50.length);
  });
  it("handles unknown part", () => {
    expect(getForecast("NOPE").text).toContain("No data");
  });
});
