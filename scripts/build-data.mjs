import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const CSV = join(process.cwd(), "data", "supply_chain_data.csv");
const OUT = join(process.cwd(), "lib", "data", "parts.json");

// ponytail: CSV has CRLF line endings; split on \r?\n so the trailing \r
// doesn't corrupt the last header/column ("price_usd") via exact-match lookup.
const lines = readFileSync(CSV, "utf8").trim().split(/\r?\n/);
const header = lines[0].split(",");
const idx = (name) => header.indexOf(name);
const col = {
  date: idx("date"), part: idx("part_id"), category: idx("category"),
  supplier: idx("supplier"), region: idx("region"), demand: idx("demand"),
  inventory: idx("inventory"), lead: idx("lead_time_days"), price: idx("price_usd"),
};

const rows = lines.slice(1).map((l) => l.split(","));
const maxDate = rows.reduce((m, r) => (r[col.date] > m ? r[col.date] : m), "");
const cutoff = new Date(maxDate);
cutoff.setDate(cutoff.getDate() - 30);
const cutoffStr = cutoff.toISOString().slice(0, 10);

const byPart = new Map();
for (const r of rows) {
  const p = r[col.part];
  if (!byPart.has(p)) byPart.set(p, []);
  byPart.get(p).push(r);
}

const parts = [];
for (const [part, prows] of byPart) {
  prows.sort((a, b) => a[col.date].localeCompare(b[col.date]));
  const last = prows[prows.length - 1];
  const last30 = prows.filter((r) => r[col.date] >= cutoffStr);
  const avg = last30.reduce((s, r) => s + Number(r[col.demand]), 0) / (last30.length || 1);
  const history = prows.slice(-90).map((r) => ({
    date: r[col.date], demand: Number(r[col.demand]),
  }));
  parts.push({
    part_id: part,
    category: last[col.category],
    supplier: last[col.supplier],
    region: last[col.region],
    lead_time_days: Number(last[col.lead]),
    price_usd: Number(last[col.price]),
    inventory: Number(last[col.inventory]),
    avg_daily_demand: Math.round(avg * 100) / 100,
    history,
  });
}
parts.sort((a, b) => a.part_id.localeCompare(b.part_id));
mkdirSync(join(process.cwd(), "lib", "data"), { recursive: true });
writeFileSync(OUT, JSON.stringify(parts));
console.log(`Wrote ${parts.length} parts to ${OUT}`);
