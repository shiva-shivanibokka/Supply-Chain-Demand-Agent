import { getPredictionLog } from "@/lib/db/log";
import { computeDrift, type DriftInputRow, type PartLike } from "@/lib/db/drift";
import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";

const PARTS: PartLike[] = (parts as PartRecord[]).map((p) => ({
  part_id: p.part_id,
  avg_daily_demand: p.avg_daily_demand,
}));

export async function GET() {
  const log = await getPredictionLog(100);
  const drift = computeDrift(log as DriftInputRow[], PARTS);
  return Response.json({ log, drift });
}
