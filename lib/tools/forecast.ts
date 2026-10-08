import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";
// forecasts.json holds a 30-day daily quantile series per part, produced by
// forecasting/export_forecasts.py. Placeholder is {} until the TFT is exported.
import forecasts from "@/lib/data/forecasts.json";

const DATA = parts as PartRecord[];

/**
 * The committed `forecasts.json` is an IN-SAMPLE FIT, not a forecast.
 *
 * `export_forecasts.py` called `from_dataset(..., predict=True)` on the full
 * observed data, so its decoder covered time_idx 1431-1460 = 2024-12-02 to
 * 2024-12-31 -- the last 30 *observed* days, which are also the window
 * `train.py` used for early stopping. Measured against those days it scores
 * p50 MAE 6.72 with 80.6% p10-p90 coverage, numbers that mean nothing as a
 * forecast because the model selected on them.
 *
 * The export bug is fixed in the code (a regression test fails on 0f94fd1 and
 * passes after), but THIS DATA FILE was never regenerated, because doing so
 * needs a trained checkpoint that is not in git. So the deployed app is still
 * serving predictions of days it had already seen, and until the file is
 * replaced it must not be presented as "the next 30 days".
 *
 * See RESULTS.md sections on the export bug. When the file IS regenerated with
 * the fixed export, set this to false and the labelling disappears.
 */
export const TFT_FORECASTS_ARE_IN_SAMPLE = true;

/** Shown wherever the TFT series is displayed, while the flag above is true. */
export const IN_SAMPLE_NOTICE =
  "These TFT values are an in-sample fit of the last 30 already-observed days " +
  "(2024-12-02 to 2024-12-31), not a forecast of future days. The export bug " +
  "is fixed in code but this data file has not been regenerated.";

type Daily = { p10: number[]; p50: number[]; p90: number[] };
const TFT = forecasts as Record<string, Daily>;

export type ForecastResult = {
  text: string;
  source: string;
  p10: number; // 30-day total
  p50: number;
  p90: number;
  p50Daily: number;
  daily: Daily; // per-day p10/p50/p90 (length = horizon)
};

const HORIZON = 30;
const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0);
const mean = (xs: number[]) => (xs.length ? sum(xs) / xs.length : 0);

function std(xs: number[], m: number): number {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
}

export function getForecast(partId: string): ForecastResult {
  const p = DATA.find((d) => d.part_id === partId);
  if (!p) {
    return {
      text: `No data found for part '${partId}'.`,
      source: "none",
      p10: 0,
      p50: 0,
      p90: 0,
      p50Daily: 0,
      daily: { p10: [], p50: [], p90: [] },
    };
  }

  const pre = TFT[partId];
  let daily: Daily;
  let source: string;
  if (
    pre &&
    Array.isArray(pre.p50) &&
    pre.p50.length &&
    Array.isArray(pre.p10) &&
    Array.isArray(pre.p90) &&
    pre.p10.length === pre.p50.length &&
    pre.p90.length === pre.p50.length
  ) {
    daily = { p10: pre.p10, p50: pre.p50, p90: pre.p90 };
    source = TFT_FORECASTS_ARE_IN_SAMPLE
      ? "TFT model (in-sample fit, not a forecast)"
      : "TFT model";
  } else {
    // Statistical baseline: mean + gentle upward trend, constant-width band.
    const recent = p.history.slice(-60).map((h) => h.demand);
    const avg = mean(recent);
    const sd = std(recent, avg);
    const trendMax = avg * 0.05;
    const p50arr = Array.from({ length: HORIZON }, (_, i) =>
      Math.max(avg + (trendMax * i) / (HORIZON - 1), 0),
    );
    daily = {
      p50: p50arr,
      p10: p50arr.map((v) => Math.max(v - 1.65 * sd, 0)),
      p90: p50arr.map((v) => v + 1.65 * sd),
    };
    source = "statistical baseline";
  }

  const p10t = Math.round(sum(daily.p10));
  const p50t = Math.round(sum(daily.p50));
  const p90t = Math.round(sum(daily.p90));
  const p50Daily = mean(daily.p50);

  const usingInSampleTft =
    source.startsWith("TFT model") && TFT_FORECASTS_ARE_IN_SAMPLE;

  const text =
    `30-day demand forecast for ${partId} (${source}):\n` +
    `  Daily demand (median): ${p50Daily.toFixed(1)} units/day\n` +
    `  Total 30-day demand: ${p50t} units (p50)\n` +
    `  Lower bound (p10): ${p10t} units\n` +
    `  Upper bound (p90): ${p90t} units\n` +
    `  Recommendation: Order at least ${p90t} units for 90% service level` +
    (usingInSampleTft ? `\n  NOTE: ${IN_SAMPLE_NOTICE}` : "");
  return { text, source, p10: p10t, p50: p50t, p90: p90t, p50Daily, daily };
}
