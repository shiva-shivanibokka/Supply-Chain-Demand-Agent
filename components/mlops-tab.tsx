"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { DriftResult } from "@/lib/db/drift";

type LogRow = {
  id: number;
  ts: string;
  partId: string;
  source: string;
  p50Daily: number;
  p50Total: number;
  p10Total: number;
  p90Total: number;
  horizonDays: number;
};

type MlopsResponse = { log: LogRow[]; drift: DriftResult };

const STATUS_BADGE_CLASS: Record<DriftResult["status"], string> = {
  OK: "bg-[#0ca30c]/15 text-[#0ca30c]",
  WARNING: "bg-[#fab219]/20 text-[#9a6b06] dark:text-[#fab219]",
  "NO-DATA": "bg-muted text-muted-foreground",
};

const CHART_COLOR = "#2a78d6";
const tickStyle = { fontSize: 11, fill: "var(--muted-foreground)" };
const axisLine = { stroke: "var(--border)" };
const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
};

export function MlopsTab() {
  const [data, setData] = useState<MlopsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const triggerRetrain = useCallback(() => {
    setRetraining(true);
    setRetrainMsg(null);
    fetch("/api/retrain", { method: "POST" })
      .then(async (res) => ({ ok: res.ok, body: await res.json() }))
      .then(({ ok, body }) => setRetrainMsg({ ok, text: ok ? body.message : body.error }))
      .catch((err) =>
        setRetrainMsg({
          ok: false,
          text: err instanceof Error ? err.message : "Failed to dispatch retraining",
        }),
      )
      .finally(() => setRetraining(false));
  }, []);

  const refresh = useCallback(() => {
    setError(null);
    fetch("/api/mlops")
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json()).error ?? "Failed to load MLOps data");
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load MLOps data"));
  }, []);

  useEffect(() => {
    // Fetch on mount; `refresh` itself resets error/data state before its fetch.
    refresh(); // eslint-disable-line react-hooks/set-state-in-effect
  }, [refresh]);

  const topParts = useMemo(() => {
    if (!data) return [];
    const counts = new Map<string, number>();
    for (const row of data.log) counts.set(row.partId, (counts.get(row.partId) ?? 0) + 1);
    return Array.from(counts, ([partId, count]) => ({ partId, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 15);
  }, [data]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-foreground">Prediction log &amp; drift monitoring</h2>
        <Button variant="outline" size="sm" onClick={refresh}>
          Refresh
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {data && (
        <>
          <Card size="sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm font-normal text-muted-foreground">
                Drift status
                <Badge variant="outline" className={STATUS_BADGE_CLASS[data.drift.status]}>
                  {data.drift.status}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Field label="Predictions scored" value={String(data.drift.nPredictions)} />
              <Field label="MAE" value={data.drift.mae !== null ? data.drift.mae.toFixed(2) : "—"} />
              <Field
                label="Baseline MAE"
                value={data.drift.baselineMae !== null ? data.drift.baselineMae.toFixed(2) : "—"}
              />
              <Field
                label="Calibration"
                value={data.drift.calibration !== null ? `${data.drift.calibration.toFixed(1)}%` : "—"}
              />
            </CardContent>
          </Card>

          {data.log.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No predictions logged yet — run some forecasts, and set DATABASE_URL to persist them.
            </p>
          ) : (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Most-queried parts</CardTitle>
                </CardHeader>
                <CardContent className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topParts} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                      <CartesianGrid stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="partId"
                        tick={tickStyle}
                        tickLine={false}
                        axisLine={axisLine}
                        interval={0}
                        angle={-35}
                        textAnchor="end"
                        height={50}
                      />
                      <YAxis tick={tickStyle} tickLine={false} axisLine={axisLine} allowDecimals={false} />
                      <Tooltip contentStyle={tooltipStyle} formatter={(value) => [value, "Predictions"]} />
                      <Bar dataKey="count" name="Predictions" fill={CHART_COLOR} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Prediction log (latest {data.log.length})</CardTitle>
                </CardHeader>
                <CardContent className="max-h-72 overflow-y-auto overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Part</TableHead>
                        <TableHead>Source</TableHead>
                        <TableHead className="text-right">p50/day</TableHead>
                        <TableHead className="text-right">p10 total</TableHead>
                        <TableHead className="text-right">p50 total</TableHead>
                        <TableHead className="text-right">p90 total</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {data.log.map((row) => (
                        <TableRow key={row.id}>
                          <TableCell className="text-muted-foreground">
                            {new Date(row.ts).toLocaleString()}
                          </TableCell>
                          <TableCell className="font-medium">{row.partId}</TableCell>
                          <TableCell>{row.source}</TableCell>
                          <TableCell className="text-right">{row.p50Daily.toFixed(1)}</TableCell>
                          <TableCell className="text-right">{Math.round(row.p10Total)}</TableCell>
                          <TableCell className="text-right">{Math.round(row.p50Total)}</TableCell>
                          <TableCell className="text-right">{Math.round(row.p90Total)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm font-semibold text-foreground">
            Automated retraining pipeline
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-muted-foreground">
          <p className="leading-relaxed">
            When drift appears, this fires a GitHub Actions pipeline that retrains the TFT,
            exports fresh forecasts, and commits them straight to main — Vercel then
            auto-deploys the new model. Fully automated: monitor → retrain → ship, no manual
            merge.
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={triggerRetrain} disabled={retraining}>
              {retraining ? "Dispatching…" : "Trigger retraining"}
            </Button>
            {retrainMsg && (
              <span className={`text-sm ${retrainMsg.ok ? "text-[#0ca30c]" : "text-destructive"}`}>
                {retrainMsg.text}
              </span>
            )}
          </div>

          <div className="space-y-3 rounded-lg border border-border bg-background/40 p-4 text-xs leading-relaxed">
            <p>
              <strong className="text-foreground">Why CPU + GitHub Actions?</strong>{" "}
              Vercel&rsquo;s serverless functions have no GPU and can&rsquo;t run PyTorch, so
              training can&rsquo;t happen in the app. On the free tier we retrain on
              GitHub&rsquo;s CPU runners — slower than a GPU, so epochs are capped for time. It
              keeps full 200-part coverage and auto-deploys, so the model stays a little less
              trained than a local GPU run (retrain locally on a GPU for best quality).
            </p>
            <p>
              <strong className="text-foreground">With a paid tier</strong> we&rsquo;d swap in
              GPU runners or a managed training service (SageMaker, Vertex AI, Modal, Replicate)
              for full-speed retraining, auto-evaluate the new model against the baseline, and
              canary-promote it through the model registry — no manual step, no scoping.
            </p>
          </div>
        </CardContent>
      </Card>
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
