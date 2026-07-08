"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import parts from "@/lib/data/parts.json";
import type { PartRecord } from "@/lib/data/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const DATA = parts as PartRecord[];
const HISTORY_COLOR = "#2a78d6";
const FORECAST_COLOR = "#1baf7a";

const tickStyle = { fontSize: 11, fill: "var(--muted-foreground)" };
const axisLine = { stroke: "var(--border)" };
const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

type ForecastResponse = { p10: number; p50: number; p90: number; p50Daily: number; source: string };

// Distributes the 30-day totals evenly across future days — the API only
// exposes aggregate quantiles, not a per-day series.
function futureDates(lastDate: string, days: number): string[] {
  const start = new Date(lastDate + "T00:00:00Z");
  return Array.from({ length: days }, (_, i) => {
    const d = new Date(start);
    d.setUTCDate(d.getUTCDate() + i + 1);
    return d.toISOString().slice(0, 10);
  });
}

export function buildForecastSeries(lastDate: string, forecast: ForecastResponse, days = 30) {
  const dates = futureDates(lastDate, days);
  const p10 = forecast.p10 / days;
  const p90 = forecast.p90 / days;
  return dates.map((date) => ({
    date,
    p10,
    p90,
    band: Math.max(p90 - p10, 0),
    p50: forecast.p50Daily,
  }));
}

const SOURCE_BADGE_CLASS: Record<string, string> = {
  "TFT model": "bg-[#2a78d6]/15 text-[#2a78d6]",
  "statistical baseline": "bg-muted text-muted-foreground",
};

export function ForecastTab() {
  const [partId, setPartId] = useState(DATA[0].part_id);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const part = useMemo(() => DATA.find((p) => p.part_id === partId)!, [partId]);

  useEffect(() => {
    let cancelled = false;
    setForecast(null);
    setError(null);
    fetch("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ partId }),
    })
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json()).error ?? "Forecast request failed");
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setForecast(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Forecast request failed");
      });
    return () => {
      cancelled = true;
    };
  }, [partId]);

  const forecastSeries = useMemo(() => {
    if (!forecast) return [];
    const lastDate = part.history[part.history.length - 1].date;
    return buildForecastSeries(lastDate, forecast);
  }, [forecast, part]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Part</span>
        <Select value={partId} onValueChange={(value) => value && setPartId(value)}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DATA.map((p) => (
              <SelectItem key={p.part_id} value={p.part_id}>
                {p.part_id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {forecast && (
          <Badge variant="outline" className={SOURCE_BADGE_CLASS[forecast.source]}>
            {forecast.source}
          </Badge>
        )}
      </div>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm font-normal text-muted-foreground">Part details</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Field label="Category" value={part.category} />
          <Field label="Supplier" value={part.supplier} />
          <Field label="Region" value={part.region} />
          <Field label="Lead time" value={`${part.lead_time_days}d`} />
          <Field label="Price" value={`$${part.price_usd.toFixed(2)}`} />
        </CardContent>
      </Card>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>History — {partId} (last 90 days)</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={part.history} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="date" tick={tickStyle} tickLine={false} axisLine={axisLine} minTickGap={30} />
                <YAxis tick={tickStyle} tickLine={false} axisLine={axisLine} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="demand" name="Demand" stroke={HISTORY_COLOR} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Forecast — {partId} (next 30 days)</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            {forecast ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={forecastSeries} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="date" tick={tickStyle} tickLine={false} axisLine={axisLine} minTickGap={30} />
                  <YAxis tick={tickStyle} tickLine={false} axisLine={axisLine} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Area dataKey="p10" stackId="band" stroke="none" fill="none" />
                  <Area
                    dataKey="band"
                    stackId="band"
                    name="p10-p90 range"
                    stroke="none"
                    fill={FORECAST_COLOR}
                    fillOpacity={0.2}
                  />
                  <Line
                    type="monotone"
                    dataKey="p50"
                    name="p50 (median)"
                    stroke={FORECAST_COLOR}
                    dot={false}
                    strokeWidth={2}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              !error && <p className="text-sm text-muted-foreground">Loading forecast...</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium text-foreground">{value}</span>
    </div>
  );
}
