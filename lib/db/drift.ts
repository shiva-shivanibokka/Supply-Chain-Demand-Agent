// Ported from mlops/mlops_cloud.py `compute_drift_metrics`.
// Takes the log rows as an argument (no DB access here) so it's unit-testable.

export type DriftInputRow = {
  partId: string;
  p50Daily: number;
  p10Total: number;
  p90Total: number;
  horizonDays: number;
};

export type PartLike = { part_id: string; avg_daily_demand: number };

export type DriftResult = {
  nPredictions: number;
  mae: number | null;
  calibration: number | null;
  driftFlag: boolean;
  baselineMae: number | null;
  status: "OK" | "WARNING" | "NO-DATA";
};

const NO_DATA: DriftResult = {
  nPredictions: 0,
  mae: null,
  calibration: null,
  driftFlag: false,
  baselineMae: null,
  status: "NO-DATA",
};

const mean = (xs: number[]): number => xs.reduce((s, x) => s + x, 0) / xs.length;

export function computeDrift(rows: DriftInputRow[], parts: PartLike[]): DriftResult {
  if (rows.length < 3) return NO_DATA;

  // Baseline MAE: predict the global mean demand for every part.
  const actuals = parts.map((p) => p.avg_daily_demand);
  const globalMean = mean(actuals);
  const baselineMae = mean(actuals.map((a) => Math.abs(a - globalMean)));

  // Match logged predictions to actuals by part.
  const actualByPart = new Map(parts.map((p) => [p.part_id, p.avg_daily_demand]));
  const matched = rows
    .filter((r) => actualByPart.has(r.partId))
    .map((r) => ({ ...r, actualAvg: actualByPart.get(r.partId) as number }));
  if (matched.length === 0) return NO_DATA;

  // MAE: p50_daily vs actual average daily demand.
  const mae = mean(matched.map((r) => Math.abs(r.p50Daily - r.actualAvg)));

  // Calibration: % of actuals inside [p10_total/horizon, p90_total/horizon].
  const inside = matched.filter((r) => {
    const p10Daily = r.p10Total / r.horizonDays;
    const p90Daily = r.p90Total / r.horizonDays;
    return r.actualAvg >= p10Daily && r.actualAvg <= p90Daily;
  });
  const calibration = (inside.length / matched.length) * 100;

  const driftFlag = baselineMae > 0 && mae > 1.5 * baselineMae;

  return {
    nPredictions: matched.length,
    mae: Math.round(mae * 100) / 100,
    calibration: Math.round(calibration * 10) / 10,
    driftFlag,
    baselineMae: Math.round(baselineMae * 100) / 100,
    status: driftFlag ? "WARNING" : "OK",
  };
}
