import { describe, it, expect } from "vitest";
import {
  getForecast,
  TFT_FORECASTS_ARE_IN_SAMPLE,
  IN_SAMPLE_NOTICE,
} from "./forecast";

describe("getForecast", () => {
  it("returns ordered quantiles for a known part", () => {
    const f = getForecast("PART_001");
    expect(f.p10).toBeLessThanOrEqual(f.p50);
    expect(f.p50).toBeLessThanOrEqual(f.p90);
    expect(f.p10).toBeGreaterThanOrEqual(0);
    expect([
      "TFT model",
      "TFT model (in-sample fit, not a forecast)",
      "statistical baseline",
    ]).toContain(f.source);
    // Daily series drives the chart — must be a non-empty, equal-length series.
    expect(f.daily.p50.length).toBeGreaterThan(0);
    expect(f.daily.p10.length).toBe(f.daily.p50.length);
    expect(f.daily.p90.length).toBe(f.daily.p50.length);
  });
  it("handles unknown part", () => {
    expect(getForecast("NOPE").text).toContain("No data");
  });

  // The committed forecasts.json predicts days the model had already seen, one
  // window of which it was early-stopped on. While that is true of the data,
  // nothing the user reads may call it a forecast of future days.
  it("never presents the in-sample TFT series as a forecast", () => {
    if (!TFT_FORECASTS_ARE_IN_SAMPLE) return; // regenerated; nothing to label
    const f = getForecast("PART_001");
    if (!f.source.startsWith("TFT model")) return; // fell through to baseline
    expect(f.source).toContain("in-sample");
    expect(f.text).toContain(IN_SAMPLE_NOTICE);
    expect(IN_SAMPLE_NOTICE).toContain("not a forecast of future days");
  });
});
