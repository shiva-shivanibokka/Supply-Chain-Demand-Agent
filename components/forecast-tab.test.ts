import { describe, it, expect } from "vitest";
import { buildForecastSeries } from "./forecast-tab";

describe("buildForecastSeries", () => {
  it("computes 30 consecutive future dates starting the day after lastDate", () => {
    const series = buildForecastSeries("2024-12-31", { p10: 300, p50: 600, p90: 900, p50Daily: 20, source: "statistical baseline" });
    expect(series).toHaveLength(30);
    expect(series[0].date).toBe("2025-01-01");
    expect(series[29].date).toBe("2025-01-30");
  });

  it("keeps p10 <= p50 <= p90 per day and derives band as the gap", () => {
    const series = buildForecastSeries("2024-01-01", { p10: 300, p50: 600, p90: 900, p50Daily: 20, source: "TFT model" });
    for (const point of series) {
      expect(point.p10).toBeLessThanOrEqual(point.p50);
      expect(point.p50).toBeLessThanOrEqual(point.p90);
      expect(point.band).toBeCloseTo(point.p90 - point.p10);
    }
  });
});
