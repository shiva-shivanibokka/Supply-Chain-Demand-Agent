"""Verifies what forecasting/export_forecasts.py (ORIGINAL version, commit 0f94fd1)
actually forecasts.

The original code does, per part:
    TimeSeriesDataSet.from_dataset(training_ds, part_df, predict=True)
on the FULL observed data. With predict=True pytorch-forecasting uses the LAST
max_prediction_length rows of each series as the decoder. This script prints
the decoder time_idx / dates of that dataset and compares the committed
lib/data/forecasts.json against the last 30 OBSERVED days.

Output: eval_sop/results/export_bug_check.json
"""
import json
import os

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from eval_sop.common import DATA, RESULTS, ROOT
from forecasting.model import load_and_prepare, build_dataset


def main():
    df = load_and_prepare(DATA)
    training_ds, _ = build_dataset(df)
    part_df = df[df["part_id"] == "PART_001"]
    pred_ds = TimeSeriesDataSet.from_dataset(training_ds, part_df, predict=True)
    x, _ = next(iter(pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
    dec_t = x["decoder_time_idx"][0].numpy()
    t2date = dict(zip(part_df["time_idx"], part_df["date"].dt.strftime("%Y-%m-%d")))
    res = {
        "last_observed_time_idx": int(df["time_idx"].max()),
        "last_observed_date": str(df["date"].max().date()),
        "original_export_decoder_time_idx_first_last": [int(dec_t[0]), int(dec_t[-1])],
        "original_export_decoder_dates_first_last": [t2date[int(dec_t[0])], t2date[int(dec_t[-1])]],
    }
    # Committed forecasts.json vs the last 30 observed days (in-sample window).
    with open(os.path.join(ROOT, "lib", "data", "forecasts.json")) as f:
        fc = json.load(f)
    last = df[df["time_idx"] > df["time_idx"].max() - 30].sort_values(["part_id", "time_idx"])
    errs, cov = [], []
    for pid, g in last.groupby("part_id"):
        if pid not in fc:
            continue
        y = g["demand"].to_numpy()
        p50 = np.array(fc[pid]["p50"])
        errs.append(np.abs(y - p50).mean())
        cov.append(((y >= np.array(fc[pid]["p10"])) & (y <= np.array(fc[pid]["p90"]))).mean())
    res["committed_forecasts_json_parts"] = len(fc)
    res["committed_p50_MAE_vs_last30_observed_days"] = float(np.mean(errs))
    res["committed_p10p90_coverage_on_last30_observed_days"] = float(np.mean(cov))
    res["conclusion"] = (
        "Decoder covers the last 30 OBSERVED days (in-sample window used as validation "
        "during training), not the 30 days after the data ends."
        if int(dec_t[-1]) == int(df["time_idx"].max()) else "Decoder extends beyond observed data."
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "export_bug_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
