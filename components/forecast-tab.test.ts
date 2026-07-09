import { describe, it, expect } from "vitest";
import { buildForecastSeries } from "./forecast-tab";

// A 30-day daily series with p10 < p50 < p90 that varies day to day.
function mkForecast(source: string) {
  const p50 = Array.from({ length: 30 }, (_, i) => 20 + i * 0.2);
  return {
    p10: 300,
    p50: 600,
    p90: 900,
    p50Daily: 20,
    source,
    daily: {
      p10: p50.map((v) => v - 5),
      p50,
      p90: p50.map((v) => v + 5),
    },
  };
}

describe("buildForecastSeries", () => {
  it("computes 30 consecutive future dates starting the day after lastDate", () => {
    const series = buildForecastSeries("2024-12-31", mkForecast("statistical baseline"));
    expect(series).toHaveLength(30);
    expect(series[0].date).toBe("2025-01-01");
    expect(series[29].date).toBe("2025-01-30");
  });

  it("plots the per-day series with p10 <= p50 <= p90 and band as the gap", () => {
    const series = buildForecastSeries("2024-01-01", mkForecast("TFT model"));
    for (const point of series) {
      expect(point.p10).toBeLessThanOrEqual(point.p50);
      expect(point.p50).toBeLessThanOrEqual(point.p90);
      expect(point.band).toBeCloseTo(point.p90 - point.p10);
    }
    // Confirms the curve varies day to day (not flat).
    expect(series[0].p50).not.toBe(series[29].p50);
  });
});
