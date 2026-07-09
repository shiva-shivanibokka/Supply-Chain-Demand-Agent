"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { summarizeInventory, type RiskLevel } from "@/lib/inventory-summary";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// Fixed status palette (never themed) — same steps hold contrast in light and dark.
const RISK_COLOR: Record<RiskLevel, string> = {
  CRITICAL: "#d03b3b",
  WARNING: "#fab219",
  OK: "#0ca30c",
};

const RISK_BADGE_CLASS: Record<RiskLevel, string> = {
  CRITICAL: "bg-[#d03b3b]/15 text-[#d03b3b]",
  WARNING: "bg-[#fab219]/20 text-[#9a6b06] dark:text-[#fab219]",
  OK: "bg-[#0ca30c]/15 text-[#0ca30c]",
};

// Category chips use the cool/brand family so they never collide with the
// red/amber/green risk colors.
const CATEGORY_CHIP: Record<string, string> = {
  Controller: "bg-[#6366f1]/15 text-[#4338ca] ring-1 ring-inset ring-[#6366f1]/30",
  Filter: "bg-[#06b6d4]/15 text-[#0e7490] ring-1 ring-inset ring-[#06b6d4]/30",
  Pump: "bg-[#d946ef]/15 text-[#a21caf] ring-1 ring-inset ring-[#d946ef]/30",
  Sensor: "bg-[#3b82f6]/15 text-[#1d4ed8] ring-1 ring-inset ring-[#3b82f6]/30",
  Valve: "bg-[#a855f7]/15 text-[#7e22ce] ring-1 ring-inset ring-[#a855f7]/30",
};
const catChip = (c: string) =>
  CATEGORY_CHIP[c] ?? "bg-muted text-foreground ring-1 ring-inset ring-foreground/20";

const usd = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

// Categorical slots 1 & 2 from the reference palette — distinct from status colors.
const INVENTORY_COLOR = "#2a78d6";
const DEMAND_COLOR = "#1baf7a";

const tickStyle = { fontSize: 11, fill: "var(--muted-foreground)" };
const axisLine = { stroke: "var(--border)" };
const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

const summary = summarizeInventory();
const CATEGORIES = Array.from(new Set(summary.parts.map((p) => p.category))).sort();
const CATEGORY_OPTIONS = ["All", ...CATEGORIES];
// Charts stay legible at any scale by showing only the most at-risk slice; the table lists all.
const CHART_LIMIT = 20;

export function InventoryDashboard() {
  const [category, setCategory] = useState<string>("All");

  const filtered = useMemo(
    () =>
      category === "All"
        ? summary.parts
        : summary.parts.filter((p) => p.category === category),
    [category],
  );

  const chartData = useMemo(
    () => [...filtered].sort((a, b) => a.daysOfSupply - b.daysOfSupply).slice(0, CHART_LIMIT),
    [filtered],
  );

  const avgLeadTime = useMemo(() => {
    if (filtered.length === 0) return 0;
    const sum = filtered.reduce((acc, p) => acc + p.lead_time_days, 0);
    return Math.round((sum / filtered.length) * 10) / 10;
  }, [filtered]);

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KpiCard label="Total parts" value={summary.parts.length} color="#6366f1" />
        <KpiCard label="Critical" value={summary.counts.critical} color={RISK_COLOR.CRITICAL} />
        <KpiCard label="Warning" value={summary.counts.warning} color={RISK_COLOR.WARNING} />
        <KpiCard label="OK" value={summary.counts.ok} color={RISK_COLOR.OK} />
      </div>

      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Category</span>
        <Select value={category} onValueChange={(value) => value && setCategory(value)}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CATEGORY_OPTIONS.map((c) => (
              <SelectItem key={c} value={c}>
                {c}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>
              Days of supply — {chartData.length} most at-risk in {category}
            </CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="part_id"
                  tick={tickStyle}
                  tickLine={false}
                  axisLine={axisLine}
                  interval={0}
                  angle={-35}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  tick={tickStyle}
                  tickLine={false}
                  axisLine={axisLine}
                  label={{ value: "Days", angle: -90, position: "insideLeft", fill: "var(--muted-foreground)", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => [`${value} days`, "Days of supply"]}
                />
                <ReferenceLine
                  y={avgLeadTime}
                  stroke="var(--foreground)"
                  strokeDasharray="4 4"
                  label={{
                    value: `Avg lead time: ${avgLeadTime}d`,
                    position: "insideTopRight",
                    fill: "var(--foreground)",
                    fontSize: 11,
                  }}
                />
                <Bar dataKey="daysOfSupply" name="Days of supply" radius={[4, 4, 0, 0]}>
                  {chartData.map((p) => (
                    <Cell key={p.part_id} fill={RISK_COLOR[p.risk]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              Inventory vs. demand — {chartData.length} most at-risk in {category}
            </CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis
                  dataKey="part_id"
                  tick={tickStyle}
                  tickLine={false}
                  axisLine={axisLine}
                  interval={0}
                  angle={-35}
                  textAnchor="end"
                  height={50}
                />
                <YAxis tick={tickStyle} tickLine={false} axisLine={axisLine} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }} />
                <Bar dataKey="inventory" name="Inventory (units)" fill={INVENTORY_COLOR} radius={[4, 4, 0, 0]} />
                <Bar
                  dataKey="avg_daily_demand"
                  name="Avg daily demand (units/day)"
                  fill={DEMAND_COLOR}
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Parts — {category} ({filtered.length})</CardTitle>
        </CardHeader>
        <CardContent className="max-h-[26rem] overflow-auto p-0">
          <Table>
            <TableHeader className="[&_th]:sticky [&_th]:top-0 [&_th]:z-10 [&_th]:h-9 [&_th]:bg-card [&_th]:text-[11px] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground [&_tr]:border-b [&_tr]:border-border">
              <TableRow>
                <TableHead className="pl-4">Part</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Supplier</TableHead>
                <TableHead>Region</TableHead>
                <TableHead className="text-right">Inventory</TableHead>
                <TableHead className="text-right">Demand/day</TableHead>
                <TableHead className="text-right">Unit price</TableHead>
                <TableHead className="text-right">Lead time</TableHead>
                <TableHead className="text-right">Days supply</TableHead>
                <TableHead className="pr-4 text-right">Risk</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((p, i) => (
                <TableRow
                  key={p.part_id}
                  className={i % 2 === 1 ? "bg-muted/20" : undefined}
                >
                  <TableCell className="pl-4 font-mono text-xs font-medium">
                    {p.part_id}
                  </TableCell>
                  <TableCell>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-semibold ${catChip(p.category)}`}
                    >
                      {p.category}
                    </span>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{p.supplier}</TableCell>
                  <TableCell className="text-muted-foreground">{p.region}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {Math.round(p.inventory).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {p.avg_daily_demand.toFixed(1)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {usd(p.price_usd)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{p.lead_time_days}d</TableCell>
                  <TableCell
                    className="text-right font-medium tabular-nums"
                    style={{ color: RISK_COLOR[p.risk] }}
                  >
                    {p.daysOfSupply}
                  </TableCell>
                  <TableCell className="pr-4 text-right">
                    <Badge variant="outline" className={RISK_BADGE_CLASS[p.risk]}>
                      {p.risk}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function KpiCard({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-normal text-muted-foreground">
          {color && (
            <span className="inline-block size-2 rounded-full" style={{ background: color }} />
          )}
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="text-3xl font-bold tabular-nums"
          style={color ? { color } : undefined}
        >
          {value}
        </div>
      </CardContent>
    </Card>
  );
}
