import { pgTable, serial, timestamp, text, doublePrecision, integer } from "drizzle-orm/pg-core";

export const predictions = pgTable("predictions", {
  id: serial("id").primaryKey(),
  ts: timestamp("ts", { withTimezone: true }).defaultNow().notNull(),
  partId: text("part_id").notNull(),
  source: text("source").notNull(),
  p50Daily: doublePrecision("p50_daily").notNull(),
  p50Total: doublePrecision("p50_total").notNull(),
  p10Total: doublePrecision("p10_total").notNull(),
  p90Total: doublePrecision("p90_total").notNull(),
  horizonDays: integer("horizon_days").notNull(),
});
