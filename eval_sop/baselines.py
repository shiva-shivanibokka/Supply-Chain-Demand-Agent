"""Statistical baselines on the same rolling origins as eval_sop/tft_backtest.py.

Needs statsforecast (installed in a SEPARATE venv so the project venv is not
modified; see RESULTS.md for exact versions). Uses only history <= origin.

Models (q10/q90 = 80% prediction interval, q50 = point forecast):
  Naive, SeasonalNaive7, SeasonalNaive365, AutoETS(season 7), AutoARIMA(season 7),
  MSTL(season 365, ETS trend), AppBaseline (port of the app's own fallback in
  lib/tools/forecast.ts: mean of last 60 days + linear 0..5% trend, +-1.65 sd).

CrostonClassic and TSB(alpha_d=0.1, alpha_p=0.1) are run for completeness
(requested), but the series have 0 zero-demand days, i.e. they are NOT
intermittent, so these methods are outside their intended use. statsforecast
gives them no intervals: point forecast only (q10/q90 = NaN, MAE/MASE only).

Resumable: one file per origin in eval_sop/results/baseline_preds/.
Run with OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 SF_JOBS=1 (footprint limit).

Usage:  python -m eval_sop.baselines
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    TSB, AutoARIMA, AutoETS, CrostonClassic, MSTL, Naive, SeasonalNaive,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_sop.common import DATA, RESULTS, HORIZON, ORIGINS  # noqa: E402


def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["part_id", "date"])
    df["time_idx"] = df.groupby("part_id").cumcount()
    return df


def app_baseline(hist, origin):
    rows = []
    for pid, g in hist.groupby("part_id"):
        recent = g["demand"].to_numpy()[-60:].astype(float)
        avg, sd = recent.mean(), recent.std()  # population sd, as in the TS/py code
        p50 = np.maximum(avg + np.linspace(0, avg * 0.05, HORIZON), 0)
        for h in range(HORIZON):
            rows.append(("AppBaseline", -1, origin, pid, h + 1, origin + 1 + h,
                         max(p50[h] - 1.65 * sd, 0), p50[h], p50[h] + 1.65 * sd))
    return rows


def main():
    df = load()
    sf_models = [
        Naive(),
        SeasonalNaive(season_length=7, alias="SeasonalNaive7"),
        SeasonalNaive(season_length=365, alias="SeasonalNaive365"),
        AutoETS(season_length=7, alias="AutoETS"),
        AutoARIMA(season_length=7, alias="AutoARIMA"),
        MSTL(season_length=[365], trend_forecaster=AutoETS(model="ZZN"), alias="MSTL365"),
    ]
    point_only = [CrostonClassic(alias="CrostonClassic"), TSB(alpha_d=0.1, alpha_p=0.1, alias="TSB")]
    outdir = os.path.join(RESULTS, "baseline_preds")
    os.makedirs(outdir, exist_ok=True)
    for origin in ORIGINS:
        path = os.path.join(outdir, f"o{origin}.csv.gz")
        if os.path.exists(path):
            print("skip (exists)", path)
            continue
        hist = df[df["time_idx"] <= origin]
        out_rows = app_baseline(hist, origin)
        y = hist.rename(columns={"part_id": "unique_id", "date": "ds", "demand": "y"})[
            ["unique_id", "ds", "y"]
        ]
        y["y"] = y["y"].astype(float)
        t0 = time.time()
        sf = StatsForecast(models=sf_models, freq="D", n_jobs=int(os.environ.get("SF_JOBS", 1)))
        fc = sf.forecast(df=y, h=HORIZON, level=[80])
        fc2 = StatsForecast(models=point_only, freq="D", n_jobs=1).forecast(df=y, h=HORIZON)
        secs = round(time.time() - t0, 1)
        print("origin", origin, "seconds", secs, flush=True)
        for f_, models, has_pi in ((fc, sf_models, True), (fc2, point_only, False)):
            f_ = f_.reset_index() if "unique_id" not in f_.columns else f_
            f_["h"] = f_.groupby("unique_id").cumcount() + 1
            for m in [mm.alias for mm in models]:
                part = pd.DataFrame({
                    "model": m, "seed": -1, "origin": origin, "part_id": f_["unique_id"],
                    "h": f_["h"], "time_idx": origin + f_["h"],
                    "q10": f_[f"{m}-lo-80"] if has_pi else np.nan, "q50": f_[m],
                    "q90": f_[f"{m}-hi-80"] if has_pi else np.nan,
                })
                out_rows += list(part.itertuples(index=False, name=None))
        out = pd.DataFrame(out_rows, columns=["model", "seed", "origin", "part_id", "h",
                                              "time_idx", "q10", "q50", "q90"])
        out.to_csv(path, index=False, float_format="%.4f", compression="gzip")
        with open(os.path.join(outdir, f"o{origin}_seconds.txt"), "w") as fh:
            fh.write(f"{secs}\n")
        print(out.groupby("model").size(), flush=True)


if __name__ == "__main__":
    main()
