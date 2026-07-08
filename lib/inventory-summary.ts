import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";
import { daysOfSupply, riskOf } from "@/lib/tools/inventory";

export type RiskLevel = "CRITICAL" | "WARNING" | "OK";
export type EnrichedPart = PartRecord & { daysOfSupply: number; risk: RiskLevel };

export function summarizeInventory(): {
  parts: EnrichedPart[];
  counts: { critical: number; warning: number; ok: number };
} {
  const enriched: EnrichedPart[] = (parts as PartRecord[]).map((p) => ({
    ...p,
    daysOfSupply: daysOfSupply(p),
    risk: riskOf(p),
  }));

  const counts = { critical: 0, warning: 0, ok: 0 };
  for (const p of enriched) {
    if (p.risk === "CRITICAL") counts.critical++;
    else if (p.risk === "WARNING") counts.warning++;
    else counts.ok++;
  }

  return { parts: enriched, counts };
}
