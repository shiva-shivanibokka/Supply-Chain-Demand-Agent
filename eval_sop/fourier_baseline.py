"""Yearly-Fourier regression baseline -- the baseline §3.3 said was untested.

ADDED 2026-10-04 (round-2 review). The statistical baselines in
eval_sop/baselines.py were given season_length=7 on data with NO weekly
pattern, while the TFT receives `month` as a known-future covariate, i.e. the
generator's only deterministic cycle. That asymmetry favours the TFT. This
script closes it with the baseline the generator actually calls for: per part,
an OLS regression of demand on

    1, t, sin(2*pi*k*t/365.25), cos(2*pi*k*t/365.25)   for k = 1..K  (K = 2)

fit on history up to and including the origin only, forecasting the next 30
days. q10/q90 are Gaussian quantiles from the in-sample residual sd (the same
convention the statsforecast baselines use), and q50/q10 are floored at 0.

This is reported SEPARATELY (eval_sop/results/fourier_baseline.json) and is
deliberately NOT merged into the main table in RESULTS.md §2, so the committed
summary.json / paired_diffs.json / per_origin.csv and the 22x3 pairwise
comparison count behind the Bonferroni check in §1 stay exactly as audited.

Not run: AutoARIMA with season_length=365. statsforecast's seasonal ARIMA at
m=365 over 1,341-1,431 observations x 200 parts was not tractable on this
laptop within the compute budget.

Usage: python -m eval_sop.fourier_baseline
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_sop.common import DATA, RESULTS, HORIZON, ORIGINS  # noqa: E402
from eval_sop.metrics import boot_ci, load_actuals, score  # noqa: E402

K = 2            # yearly harmonics (the generator has one; 2 is a mild hedge)
PERIOD = 365.25
Z = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}
B = 2000


def design(t):
    cols = [np.ones_like(t, dtype=float), t.astype(float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / PERIOD), np.cos(2 * np.pi * k * t / PERIOD)]
    return np.column_stack(cols)


def fit_predict(df):
    rows = []
    for origin in ORIGINS:
        fut = np.arange(origin + 1, origin + 1 + HORIZON)
        Xf = design(fut)
        for pid, g in df.groupby("part_id", sort=True):
            h = g[g["time_idx"] <= origin]
            t, y = h["time_idx"].to_numpy(), h["demand"].to_numpy(float)
            beta, *_ = np.linalg.lstsq(design(t), y, rcond=None)
            resid = y - design(t) @ beta
            sd = float(np.sqrt((resid ** 2).sum() / max(len(y) - design(t).shape[1], 1)))
            mu = Xf @ beta
            for i in range(HORIZON):
                rows.append(("FourierYearly", -1, origin, pid, i + 1, int(fut[i]),
                             max(mu[i] + Z[0.1] * sd, 0.0), max(mu[i], 0.0), mu[i] + Z[0.9] * sd))
        print("fitted origin", origin, flush=True)
    return pd.DataFrame(rows, columns=["model", "seed", "origin", "part_id", "h",
                                       "time_idx", "q10", "q50", "q90"])


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["part_id", "date"])
    df["time_idx"] = df.groupby("part_id").cumcount()
    preds = fit_predict(df)
    out_csv = os.path.join(RESULTS, "fourier_preds.csv.gz")
    preds.to_csv(out_csv, index=False, float_format="%.4f", compression="gzip")

    # Score with the SAME code path as the main table, then compare against the
    # already-committed per-part-window metrics of TFT / AutoARIMA / Oracle.
    actuals, scales = load_actuals()
    g_new = score(preds, actuals, scales)
    g_old = pd.read_csv(os.path.join(RESULTS, "per_part_window_metrics.csv.gz"))
    g = pd.concat([g_old, g_new], ignore_index=True)

    parts = np.array(sorted(g["part_id"].unique()))
    rng = np.random.default_rng(12345)          # same seed/scheme as metrics.py
    idx = rng.integers(0, len(parts), size=(B, len(parts)))
    METRICS = ["MAE", "MASE", "pinball", "cov80", "width80"]
    pp = g.groupby(["model", "part_id"])[METRICS].mean()

    res = {"model": "FourierYearly", "harmonics": K, "period_days": PERIOD,
           "n_part_windows": int(len(g_new)),   # g_new is one row per part-window
           "note": "reported separately; main table in RESULTS.md section 2 unchanged"}
    ppm = pp.loc["FourierYearly"].reindex(parts)
    for k in METRICS:
        res[k] = float(g_new.groupby("origin")[k].mean().mean())
        res[k + "_ci95"] = list(boot_ci(ppm[k].to_numpy(), idx))
    res["per_origin_MAE"] = {int(o): float(v) for o, v in g_new.groupby("origin")["MAE"].mean().items()}

    res["paired_vs"] = {}
    for other in ("TFT", "AutoARIMA", "AutoETS", "Oracle"):
        b = pp.loc[other].reindex(parts)
        entry = {}
        for k in ("MAE", "pinball"):
            d = (ppm[k] - b[k]).to_numpy()                       # Fourier minus other
            lo, hi = boot_ci(d, idx)
            av, bv = ppm[k].to_numpy(), b[k].to_numpy()
            rel = 100 * (av[idx].mean(axis=1) / bv[idx].mean(axis=1) - 1)
            entry[k] = {"mean_diff": float(d.mean()), "ci95": [lo, hi],
                        "rel_pct": float(100 * d.mean() / b[k].mean()),
                        "rel_pct_ci95": [float(np.percentile(rel, 2.5)),
                                         float(np.percentile(rel, 97.5))]}
        res["paired_vs"][other] = entry

    # TFT vs Fourier stated the usual way round (TFT minus Fourier).
    tft = pp.loc["TFT"].reindex(parts)
    res["TFT_minus_Fourier"] = {}
    for k in ("MAE", "pinball"):
        d = (tft[k] - ppm[k]).to_numpy()
        lo, hi = boot_ci(d, idx)
        av, bv = tft[k].to_numpy(), ppm[k].to_numpy()
        rel = 100 * (av[idx].mean(axis=1) / bv[idx].mean(axis=1) - 1)
        res["TFT_minus_Fourier"][k] = {"mean_diff": float(d.mean()), "ci95": [lo, hi],
                                       "rel_pct": float(100 * d.mean() / ppm[k].mean()),
                                       "rel_pct_ci95": [float(np.percentile(rel, 2.5)),
                                                        float(np.percentile(rel, 97.5))]}

    with open(os.path.join(RESULTS, "fourier_baseline.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
