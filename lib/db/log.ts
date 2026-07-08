import { desc } from "drizzle-orm";
import { db } from "./client";
import { predictions } from "./schema";

export type PredictionRow = {
  partId: string;
  source: string;
  p50Daily: number;
  p50Total: number;
  p10Total: number;
  p90Total: number;
  horizonDays: number;
};

export async function logPrediction(row: PredictionRow): Promise<void> {
  if (!db) {
    console.warn("[predictions] DATABASE_URL unset — skipping log");
    return;
  }
  await db.insert(predictions).values(row);
}

export async function getPredictionLog(limit = 100) {
  if (!db) return [];
  return db.select().from(predictions).orderBy(desc(predictions.ts)).limit(limit);
}
