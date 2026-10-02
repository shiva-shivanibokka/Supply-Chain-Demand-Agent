"""Scores all forecasts in eval_sop/results against realized demand.

Metrics per (model, seed, origin, part) over the 30-day window:
  MAE        mean |y - q50|
  MASE       MAE / in-sample MAE of the lag-1 naive forecast on history <= origin
  pinball    mean pinball loss averaged over q in {0.1, 0.5, 0.9}
  cov80      fraction of days with q10 <= y <= q90 (nominal 0.80)
  width80    mean (q90 - q10)
Aggregate = mean over 200 parts x 4 origins (n = 800 part-windows, 24,000 days).

TFT: one aggregate per seed -> mean +- std over 3 seeds. Bootstrap CIs (2,000
resamples of the 200 PARTS, with all their origins, seed 12345) are computed on
the per-part metric averaged over the 3 seeds. Paired bootstrap of the
difference vs each baseline uses the same resampled parts.

Usage: python -m eval_sop.metrics
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_sop.common import DATA, RESULTS, ORIGINS  # noqa: E402

METRICS = ["MAE", "MASE", "pinball", "cov80", "width80"]
B = 2000


def load_actuals():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["part_id", "date"])
    df["time_idx"] = df.groupby("part_id").cumcount()
    scales = []
    for o in ORIGINS:
        h = df[df["time_idx"] <= o]
        s = h.groupby("part_id")["demand"].apply(lambda v: np.abs(np.diff(v.to_numpy())).mean())
        scales.append(pd.DataFrame({"part_id": s.index, "origin": o, "scale": s.values}))
    return df[["part_id", "time_idx", "demand"]], pd.concat(scales)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score(preds, actuals, scales):
    m = preds.merge(actuals, on=["part_id", "time_idx"], how="left")
    assert m["demand"].notna().all()
    y = m["demand"].to_numpy(float)
    m["ae"] = np.abs(y - m["q50"])
    m["pin"] = (pinball(y, m["q10"], 0.1) + pinball(y, m["q50"], 0.5) + pinball(y, m["q90"], 0.9)) / 3
    m["in80"] = ((y >= m["q10"]) & (y <= m["q90"])).astype(float)
    m.loc[m["q10"].isna() | m["q90"].isna(), "in80"] = np.nan  # point-only models (Croston/TSB)
    m["w"] = m["q90"] - m["q10"]
    m["crossed"] = ((m["q10"] > m["q50"]) | (m["q50"] > m["q90"])).astype(float)
    g = m.groupby(["model", "seed", "origin", "part_id"]).agg(
        MAE=("ae", "mean"), pinball=("pin", "mean"), cov80=("in80", "mean"),
        width80=("w", "mean"), crossed=("crossed", "mean"), n=("ae", "size")).reset_index()
    g = g.merge(scales, on=["part_id", "origin"])
    g["MASE"] = g["MAE"] / g["scale"]
    return g


def boot_ci(per_part, rng_idx):
    """per_part: array [n_parts] ; rng_idx: [B, n_parts] resample indices."""
    bs = per_part[rng_idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    actuals, scales = load_actuals()
    tft_files = sorted(glob.glob(os.path.join(RESULTS, "tft_preds", "*.csv*")))
    preds = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RESULTS, "baseline_preds", "o*.csv.gz")))]
    preds += [pd.read_csv(f) for f in tft_files]
    if os.path.exists(os.path.join(RESULTS, "oracle_preds.csv")):
        preds.append(pd.read_csv(os.path.join(RESULTS, "oracle_preds.csv")))
    preds = pd.concat(preds, ignore_index=True)
    preds["model"] = preds["model"].replace({"full": "TFT", "no_meta": "TFT_no_meta"})
    g = score(preds, actuals, scales)
    g.to_csv(os.path.join(RESULTS, "per_part_window_metrics.csv.gz"), index=False,
             float_format="%.5f", compression="gzip")

    # completeness: every model must cover all 4 origins x 200 parts
    cover = g.groupby(["model", "seed"]).size()
    print("part-windows per model/seed:\n", cover)
    incomplete = cover[cover != 200 * len(ORIGINS)]
    if len(incomplete):
        print("WARNING: INCOMPLETE runs (expected 800 part-windows each):\n", incomplete)

    parts = np.array(sorted(g["part_id"].unique()))
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(parts), size=(B, len(parts)))

    # per-part (averaged over origins, and over seeds for TFT) for bootstrap
    pp = g.groupby(["model", "part_id"])[METRICS].mean()

    rows = []
    for model, gm in g.groupby("model"):
        seeds = sorted(gm["seed"].unique())
        per_seed = gm.groupby("seed")[METRICS].mean()
        r = {"model": model, "n_seeds": len(seeds),
             "n_part_windows_per_seed": int(len(gm) / len(seeds)),
             "n_days_per_seed": int(gm["n"].sum() / len(seeds)),
             "quantile_crossing_rate": float(gm["crossed"].mean())}
        ppm = pp.loc[model].reindex(parts)
        for k in METRICS:
            r[k] = float(per_seed[k].mean())
            r[k + "_seed_std"] = float(per_seed[k].std(ddof=1)) if len(seeds) > 1 else None
            lo, hi = boot_ci(ppm[k].to_numpy(), idx)
            r[k + "_ci95"] = [lo, hi]
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("MAE")

    # paired differences TFT - baseline
    diffs = []
    for tft in [m for m in ["TFT", "TFT_no_meta"] if m in pp.index.get_level_values(0)]:
        a = pp.loc[tft].reindex(parts)
        for other in sorted(set(pp.index.get_level_values(0)) - {tft}):
            b = pp.loc[other].reindex(parts)
            for k in ["MAE", "MASE", "pinball"]:
                d = (a[k] - b[k]).to_numpy()
                lo, hi = boot_ci(d, idx)
                diffs.append({"a": tft, "b": other, "metric": k, "mean_diff_a_minus_b": float(d.mean()),
                              "ci95": [lo, hi], "rel_diff_pct": float(100 * d.mean() / b[k].mean())})
    diffs = pd.DataFrame(diffs)

    per_origin = g.groupby(["model", "origin"])[["MAE", "MASE", "pinball", "cov80"]].mean().reset_index()

    summary.to_json(os.path.join(RESULTS, "summary.json"), orient="records", indent=1)
    diffs.to_json(os.path.join(RESULTS, "paired_diffs.json"), orient="records", indent=1)
    per_origin.to_csv(os.path.join(RESULTS, "per_origin.csv"), index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["model", "n_seeds", "MAE", "MAE_seed_std", "MAE_ci95", "MASE", "MASE_ci95",
            "pinball", "pinball_ci95", "cov80", "cov80_ci95", "width80", "quantile_crossing_rate"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(diffs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(per_origin.pivot(index="model", columns="origin", values="MAE").round(3))


if __name__ == "__main__":
    main()
