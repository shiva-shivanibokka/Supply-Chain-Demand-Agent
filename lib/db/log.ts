import { desc, sql } from "drizzle-orm";
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

// Lightweight self-bootstrap: create the table on first use (cached per warm
// instance). Vercel injects Neon's connection string only at runtime and hides
// it from `vercel env pull`, so `drizzle-kit push` from a laptop isn't possible;
// this keeps the schema in sync without a manual migration step. `drizzle-kit
// push` (npm run db:push) remains the migration path if you have the URL.
let ensured: Promise<void> | null = null;
function ensureTable(): Promise<void> {
  if (!db) return Promise.resolve();
  if (!ensured) {
    ensured = db
      .execute(sql`
        CREATE TABLE IF NOT EXISTS predictions (
          id serial PRIMARY KEY,
          ts timestamptz NOT NULL DEFAULT now(),
          part_id text NOT NULL,
          source text NOT NULL,
          p50_daily double precision NOT NULL,
          p50_total double precision NOT NULL,
          p10_total double precision NOT NULL,
          p90_total double precision NOT NULL,
          horizon_days integer NOT NULL
        )
      `)
      .then(() => undefined)
      .catch((err) => {
        ensured = null; // let a later call retry
        throw err;
      });
  }
  return ensured;
}

export async function logPrediction(row: PredictionRow): Promise<void> {
  if (!db) {
    console.warn("[predictions] no database configured — skipping log");
    return;
  }
  try {
    await ensureTable();
    await db.insert(predictions).values(row);
  } catch (err) {
    console.error("[predictions] insert failed:", err);
  }
}

export async function getPredictionLog(limit = 100) {
  if (!db) return [];
  try {
    await ensureTable();
    return await db.select().from(predictions).orderBy(desc(predictions.ts)).limit(limit);
  } catch (err) {
    console.error("[predictions] read failed:", err);
    return [];
  }
}
