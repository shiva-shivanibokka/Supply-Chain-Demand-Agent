"""Data provenance + oracle reference.

1. Re-runs data/generate_data.py (seed 42) in memory and checks it reproduces
   data/supply_chain_data.csv exactly.
2. Records each part's noise-free generating signal (trend + yearly sine) by
   wrapping generate_demand_signal, and writes an "Oracle" forecast:
       q_tau = floor(signal + z_tau * 0.15 * base)
   This uses the TRUE generator parameters, so it is NOT an achievable model.
   It is a floor reference: what an ideal forecaster that knew the generator
   would score (it ignores the unpredictable i.i.d. spikes).
3. Fits a quick check that category/supplier/region carry no signal:
   R^2 of per-part mean demand on those categoricals (one-hot OLS).

Usage: python -m eval_sop.provenance_oracle
"""
import json
import os

import numpy as np
import pandas as pd

from eval_sop.common import DATA, RESULTS, HORIZON, ORIGINS
import data.generate_data as gen

Z = {0.1: -1.2815516, 0.5: 0.0, 0.9: 1.2815516}


def main():
    captured = {}
    orig = gen.generate_demand_signal

    def wrapped(n_days, base, part_idx):
        days = np.arange(n_days)
        trend = base * (1 + 0.2 * days / n_days)
        season = base * 0.3 * np.sin(2 * np.pi * days / 365 + part_idx * (2 * np.pi / gen.NUM_PARTS))
        captured[f"PART_{part_idx + 1:03d}"] = (base, trend + season)
        return orig(n_days, base, part_idx)

    gen.generate_demand_signal = wrapped
    np.random.seed(42)
    regen = gen.generate_dataset()
    csv = pd.read_csv(DATA, parse_dates=["date"])
    same = (len(regen) == len(csv)) and bool(
        (regen["demand"].to_numpy() == csv["demand"].to_numpy()).all()
        and (regen["part_id"].to_numpy() == csv["part_id"].to_numpy()).all()
        and (regen["category"].to_numpy() == csv["category"].to_numpy()).all()
        and (regen["inventory"].to_numpy() == csv["inventory"].to_numpy()).all()
    )

    rows = []
    for o in ORIGINS:
        for pid, (base, sig) in captured.items():
            for h in range(HORIZON):
                t = o + 1 + h
                q = {tau: max(np.floor(sig[t] + z * 0.15 * base), 0) for tau, z in Z.items()}
                rows.append(("Oracle", -1, o, pid, h + 1, t, q[0.1], q[0.5], q[0.9]))
    pd.DataFrame(rows, columns=["model", "seed", "origin", "part_id", "h", "time_idx",
                                "q10", "q50", "q90"]).to_csv(
        os.path.join(RESULTS, "oracle_preds.csv.gz"), index=False)

    # covariate signal check
    pp = csv.groupby("part_id").agg(mean=("demand", "mean"), category=("category", "first"),
                                    supplier=("supplier", "first"), region=("region", "first"),
                                    lead=("lead_time_days", "first"), price=("price_usd", "first"))
    X = pd.get_dummies(pp[["category", "supplier", "region"]], drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), pp["mean"].to_numpy(), rcond=None)
    resid = pp["mean"].to_numpy() - X.to_numpy() @ beta
    r2 = 1 - resid.var() / pp["mean"].var(ddof=0)
    k = X.shape[1] - 1
    adj = 1 - (1 - r2) * (len(pp) - 1) / (len(pp) - k - 1)
    res = {
        "regenerated_csv_identical": same,
        "n_rows": len(csv), "n_parts": int(csv["part_id"].nunique()),
        "date_range": [str(csv["date"].min().date()), str(csv["date"].max().date())],
        "zero_demand_fraction": float((csv["demand"] == 0).mean()),
        "r2_part_mean_on_category_supplier_region": float(r2),
        "adj_r2_part_mean_on_category_supplier_region": float(adj),
        "corr_part_mean_vs_lead_time": float(pp["mean"].corr(pp["lead"])),
        "corr_part_mean_vs_price": float(pp["mean"].corr(pp["price"])),
    }
    with open(os.path.join(RESULTS, "provenance.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
