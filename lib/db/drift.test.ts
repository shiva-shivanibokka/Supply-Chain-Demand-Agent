import { describe, it, expect } from "vitest";
import { computeDrift, type DriftInputRow, type PartLike } from "./drift";

// Fixed part catalog: avg_daily_demand of 10/20/30/40 -> global mean 25,
// baseline MAE (predict global mean for every part) = mean(15,5,5,15) = 10.
const PARTS: PartLike[] = [
  { part_id: "P1", avg_daily_demand: 10 },
  { part_id: "P2", avg_daily_demand: 20 },
  { part_id: "P3", avg_daily_demand: 30 },
  { part_id: "P4", avg_daily_demand: 40 },
];

describe("computeDrift", () => {
  it("returns NO-DATA when fewer than 3 rows are logged", () => {
    const rows: DriftInputRow[] = [
      { partId: "P1", p50Daily: 10, p10Total: 150, p90Total: 450, horizonDays: 30 },
      { partId: "P2", p50Daily: 20, p10Total: 450, p90Total: 750, horizonDays: 30 },
    ];
    const result = computeDrift(rows, PARTS);
    expect(result.status).toBe("NO-DATA");
    expect(result.nPredictions).toBe(0);
    expect(result.mae).toBeNull();
    expect(result.baselineMae).toBeNull();
    expect(result.calibration).toBeNull();
    expect(result.driftFlag).toBe(false);
  });

  it("reports no drift when forecasts exactly match actuals and fall inside the band", () => {
    // p50Daily == actual for every part -> MAE 0; band is actual +/- 5/day -> 100% calibrated.
    const rows: DriftInputRow[] = [
      { partId: "P1", p50Daily: 10, p10Total: 150, p90Total: 450, horizonDays: 30 },
      { partId: "P2", p50Daily: 20, p10Total: 450, p90Total: 750, horizonDays: 30 },
      { partId: "P3", p50Daily: 30, p10Total: 750, p90Total: 1050, horizonDays: 30 },
      { partId: "P4", p50Daily: 40, p10Total: 1050, p90Total: 1350, horizonDays: 30 },
    ];
    const result = computeDrift(rows, PARTS);
    expect(result.status).toBe("OK");
    expect(result.driftFlag).toBe(false);
    expect(result.nPredictions).toBe(4);
    expect(result.baselineMae).toBe(10);
    expect(result.mae).toBe(0);
    expect(result.calibration).toBe(100);
  });

  it("flags drift when forecasts are far off and outside the calibration band", () => {
    // p50Daily = 100 for every part regardless of actual -> big MAE, 0% calibration.
    const rows: DriftInputRow[] = [
      { partId: "P1", p50Daily: 100, p10Total: 2970, p90Total: 3030, horizonDays: 30 },
      { partId: "P2", p50Daily: 100, p10Total: 2970, p90Total: 3030, horizonDays: 30 },
      { partId: "P3", p50Daily: 100, p10Total: 2970, p90Total: 3030, horizonDays: 30 },
      { partId: "P4", p50Daily: 100, p10Total: 2970, p90Total: 3030, horizonDays: 30 },
    ];
    const result = computeDrift(rows, PARTS);
    expect(result.status).toBe("WARNING");
    expect(result.driftFlag).toBe(true);
    expect(result.nPredictions).toBe(4);
    expect(result.baselineMae).toBe(10);
    expect(result.mae).toBe(75);
    expect(result.calibration).toBe(0);
  });

  it("keeps MAE non-negative and calibration within [0, 100] for arbitrary input", () => {
    const rows: DriftInputRow[] = [
      { partId: "P1", p50Daily: 12, p10Total: 200, p90Total: 400, horizonDays: 30 },
      { partId: "P2", p50Daily: 18, p10Total: 400, p90Total: 700, horizonDays: 30 },
      { partId: "P3", p50Daily: 35, p10Total: 700, p90Total: 1000, horizonDays: 30 },
    ];
    const result = computeDrift(rows, PARTS);
    expect(result.mae).not.toBeNull();
    expect(result.mae as number).toBeGreaterThanOrEqual(0);
    expect(result.calibration as number).toBeGreaterThanOrEqual(0);
    expect(result.calibration as number).toBeLessThanOrEqual(100);
    expect(typeof result.driftFlag).toBe("boolean");
  });

  it("drops logged rows whose partId is not in the part catalog before scoring", () => {
    const rows: DriftInputRow[] = [
      { partId: "P1", p50Daily: 10, p10Total: 150, p90Total: 450, horizonDays: 30 },
      { partId: "P2", p50Daily: 20, p10Total: 450, p90Total: 750, horizonDays: 30 },
      { partId: "UNKNOWN", p50Daily: 999, p10Total: 1, p90Total: 2, horizonDays: 30 },
    ];
    const result = computeDrift(rows, PARTS);
    expect(result.nPredictions).toBe(2);
  });
});
