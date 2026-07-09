"""
forecasting/export_forecasts.py
-------------------------------
Runs the trained TFT for every part and writes lib/data/forecasts.json, which
the Vercel web app serves as real "TFT model" forecasts (it falls back to the
statistical baseline when this file is empty/absent).

Run locally after training:
    python -m forecasting.train
    python -m forecasting.export_forecasts

Requires requirements-local.txt and a checkpoint in forecasting/saved_model/.
Loads the model and dataset once, then predicts per part (fast, no re-loads).
"""

import glob
import json
import math
import os


DATA = "data/supply_chain_data.csv"
OUT = "lib/data/forecasts.json"
CKPT_DIR = "forecasting/saved_model"


def _sanity_check(part_id: str, p10: list, p50: list, p90: list) -> None:
    """Guards the auto-committed forecasts.json: fail loudly instead of
    letting a broken forecast (NaN/inf, negative, or crossed quantiles)
    get pushed straight to main with no human review."""
    for name, series in (("p10", p10), ("p50", p50), ("p90", p90)):
        for x in series:
            if not math.isfinite(x) or x < 0:
                raise SystemExit(
                    f"Forecast sanity check failed for {part_id}: {name} has "
                    f"non-finite or negative value ({x})"
                )
    for i, (lo, mid, hi) in enumerate(zip(p10, p50, p90)):
        if not (lo <= mid <= hi):
            raise SystemExit(
                f"Forecast sanity check failed for {part_id}: day {i} quantiles "
                f"not ordered (p10={lo}, p50={mid}, p90={hi})"
            )


def main() -> None:
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/*.ckpt"))
    if not ckpts:
        raise SystemExit("No checkpoint found. Train first: python -m forecasting.train")
    ckpt = ckpts[0]
    print(f"Using checkpoint: {ckpt}")

    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

    from forecasting.model import load_and_prepare, build_dataset, DECODER_LENGTH

    full_df = load_and_prepare(DATA)
    training_ds, _ = build_dataset(full_df)
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    model.eval()

    out: dict[str, dict[str, float]] = {}
    part_ids = sorted(full_df["part_id"].unique())
    for i, part_id in enumerate(part_ids, 1):
        part_df = full_df[full_df["part_id"] == part_id]
        pred_ds = TimeSeriesDataSet.from_dataset(training_ds, part_df, predict=True)
        loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        with torch.no_grad():
            preds = model.predict(loader, mode="quantiles", return_y=False)
        p10 = preds[:, :, 0].cpu().numpy().flatten()[:DECODER_LENGTH]
        p50 = preds[:, :, 1].cpu().numpy().flatten()[:DECODER_LENGTH]
        p90 = preds[:, :, 2].cpu().numpy().flatten()[:DECODER_LENGTH]
        # Store the full 30-day daily quantile series so the UI can plot the real
        # forecast shape (trend/seasonality), not a flat aggregate. Totals and the
        # daily median are derived from these arrays downstream.
        p10_r = [round(float(x), 1) for x in p10]
        p50_r = [round(float(x), 1) for x in p50]
        p90_r = [round(float(x), 1) for x in p90]
        _sanity_check(part_id, p10_r, p50_r, p90_r)
        out[part_id] = {"p10": p10_r, "p50": p50_r, "p90": p90_r}
        if i % 25 == 0 or i == len(part_ids):
            print(f"  {i}/{len(part_ids)} parts forecast")

    os.makedirs("lib/data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} TFT forecasts to {OUT}")


if __name__ == "__main__":
    main()
