import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";
// forecasts.json is generated locally by forecasting/export_forecasts.py.
// Placeholder is {} until the TFT is trained + exported (Task 9 overwrites it).
import forecasts from "@/lib/data/forecasts.json";

const DATA = parts as PartRecord[];

type TFTEntry = { p10: number; p50: number; p90: number; p50Daily: number };
const TFT = forecasts as Record<string, TFTEntry>;

export type ForecastResult = {
  text: string;
  source: string;
  p10: number;
  p50: number;
  p90: number;
  p50Daily: number;
};

function std(xs: number[], mean: number): number {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length);
}

export function getForecast(partId: string): ForecastResult {
  const p = DATA.find((d) => d.part_id === partId);
  if (!p) return { text: `No data found for part '${partId}'.`, source: "none", p10: 0, p50: 0, p90: 0, p50Daily: 0 };

  const pre = TFT[partId];
  let p10t: number, p50t: number, p90t: number, p50Daily: number, source: string;
  if (pre) {
    ({ p10: p10t, p50: p50t, p90: p90t, p50Daily } = pre);
    source = "TFT model";
  } else {
    const recent = p.history.slice(-60).map((h) => h.demand);
    const avg = recent.reduce((s, x) => s + x, 0) / (recent.length || 1);
    const sd = std(recent, avg);
    const horizon = 30;
    p50Daily = avg * 1.025; // midpoint of 0..5% trend
    p50t = Math.round(p50Daily * horizon);
    p10t = Math.round(Math.max(p50Daily - 1.65 * sd, 0) * horizon);
    p90t = Math.round((p50Daily + 1.65 * sd) * horizon);
    source = "statistical baseline";
  }
  const text =
    `30-day demand forecast for ${partId} (${source}):\n` +
    `  Daily demand (median): ${p50Daily.toFixed(1)} units/day\n` +
    `  Total 30-day demand: ${p50t} units (p50)\n` +
    `  Lower bound (p10): ${p10t} units\n` +
    `  Upper bound (p90): ${p90t} units\n` +
    `  Recommendation: Order at least ${p90t} units for 90% service level`;
  return { text, source, p10: p10t, p50: p50t, p90: p90t, p50Daily };
}
