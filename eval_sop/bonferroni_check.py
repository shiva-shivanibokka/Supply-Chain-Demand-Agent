"""Multiplicity check for the headline comparisons (round-2 review, 2026-10-04).

RESULTS.md reports 66 pairwise comparisons (22 model pairs x 3 metrics: 2 TFT
variants x 11 other models) plus 60 marginal CIs (12 models x 5 metrics), all
as UNADJUSTED 95% percentile bootstraps. This script re-runs the paired,
part-clustered bootstrap for the two headline comparisons at the Bonferroni
level alpha/66 (two-sided, i.e. the 0.0379% and 99.9621% percentiles) and
reports whether the interval still excludes 0.

B is raised to 200,000 (in chunks) because alpha/66 percentiles are not
resolvable at B = 2,000. The resampling unit, the statistic and the seed scheme
are identical to eval_sop/metrics.py; only B and the percentiles differ, so
nothing in summary.json / paired_diffs.json is touched or regenerated.

Usage: python -m eval_sop.bonferroni_check
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_sop.common import RESULTS  # noqa: E402

N_COMPARISONS = 66          # 22 pairs x 3 metrics
ALPHA = 0.05
B = 200_000
CHUNK = 10_000
PAIRS = [("TFT", "AutoARIMA", "MAE"), ("TFT", "Oracle", "MAE")]


def main():
    g = pd.read_csv(os.path.join(RESULTS, "per_part_window_metrics.csv.gz"))
    parts = np.array(sorted(g["part_id"].unique()))
    pp = g.groupby(["model", "part_id"])[["MAE", "MASE", "pinball"]].mean()
    rng = np.random.default_rng(12345)
    q_lo, q_hi = 100 * (ALPHA / N_COMPARISONS) / 2, 100 * (1 - (ALPHA / N_COMPARISONS) / 2)

    out = {"n_comparisons": N_COMPARISONS, "alpha": ALPHA, "B": B,
           "percentiles": [q_lo, q_hi], "marginal_cis_reported": 60, "pairs": {}}
    for a, b, metric in PAIRS:
        av = pp.loc[a].reindex(parts)[metric].to_numpy()
        bv = pp.loc[b].reindex(parts)[metric].to_numpy()
        d = av - bv
        diffs, rels = [], []
        for _ in range(B // CHUNK):
            idx = rng.integers(0, len(parts), size=(CHUNK, len(parts)))
            diffs.append(d[idx].mean(axis=1))
            rels.append(100 * (av[idx].mean(axis=1) / bv[idx].mean(axis=1) - 1))
        diffs, rels = np.concatenate(diffs), np.concatenate(rels)
        key = f"{a}_minus_{b}_{metric}"
        out["pairs"][key] = {
            "mean_diff": float(d.mean()),
            "bonferroni_ci": [float(np.percentile(diffs, q_lo)), float(np.percentile(diffs, q_hi))],
            "bonferroni_rel_pct_ci": [float(np.percentile(rels, q_lo)), float(np.percentile(rels, q_hi))],
            "boot_min_max_diff": [float(diffs.min()), float(diffs.max())],
            "excludes_zero": bool(np.percentile(diffs, q_lo) > 0 or np.percentile(diffs, q_hi) < 0),
        }
    with open(os.path.join(RESULTS, "bonferroni_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
