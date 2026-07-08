import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";

const DATA = parts as PartRecord[];

function risk(p: PartRecord): "CRITICAL" | "WARNING" | "OK" {
  const dos = p.inventory / Math.max(p.avg_daily_demand, 0.1);
  if (dos < p.lead_time_days) return "CRITICAL";
  if (dos < 2 * p.lead_time_days) return "WARNING";
  return "OK";
}

export function daysOfSupply(p: PartRecord): number {
  return Math.round((p.inventory / Math.max(p.avg_daily_demand, 0.1)) * 10) / 10;
}

export function riskOf(p: PartRecord) {
  return risk(p);
}

export function getInventoryStatus(opts: { partId?: string; topN?: number } = {}): string {
  const { partId, topN = 10 } = opts;
  if (partId) {
    const p = DATA.find((d) => d.part_id === partId);
    if (!p) return `Part '${partId}' not found in dataset.`;
    return (
      `Part: ${p.part_id} | Category: ${p.category} | Supplier: ${p.supplier} | Region: ${p.region}\n` +
      `Inventory: ${Math.round(p.inventory)} units | Avg daily demand: ${p.avg_daily_demand.toFixed(1)} units/day\n` +
      `Days of supply: ${daysOfSupply(p)} days | Lead time: ${p.lead_time_days} days | Risk: ${risk(p)}`
    );
  }
  const atRisk = DATA.filter((p) => risk(p) !== "OK")
    .sort((a, b) => daysOfSupply(a) - daysOfSupply(b))
    .slice(0, topN);
  if (atRisk.length === 0) return "All parts have sufficient inventory levels.";
  return (
    `Top ${atRisk.length} at-risk parts:\n` +
    atRisk
      .map(
        (p) =>
          `  [${risk(p)}] ${p.part_id} (${p.category}, ${p.supplier}) - ${daysOfSupply(p)} days supply (lead time: ${p.lead_time_days} days)`,
      )
      .join("\n")
  );
}
