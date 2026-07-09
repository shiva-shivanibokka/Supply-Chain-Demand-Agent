import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";
// forecasts.json holds a 30-day daily quantile series per part, produced by
// forecasting/export_forecasts.py. Placeholder is {} until the TFT is exported.
import forecasts from "@/lib/data/forecasts.json";

const DATA = parts as PartRecord[];

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
  if (pre && Array.isArray(pre.p50) && pre.p50.length) {
    daily = { p10: pre.p10, p50: pre.p50, p90: pre.p90 };
    source = "TFT model";
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

  const text =
    `30-day demand forecast for ${partId} (${source}):\n` +
    `  Daily demand (median): ${p50Daily.toFixed(1)} units/day\n` +
    `  Total 30-day demand: ${p50t} units (p50)\n` +
    `  Lower bound (p10): ${p10t} units\n` +
    `  Upper bound (p90): ${p90t} units\n` +
    `  Recommendation: Order at least ${p90t} units for 90% service level`;
  return { text, source, p10: p10t, p50: p50t, p90: p90t, p50Daily, daily };
}
