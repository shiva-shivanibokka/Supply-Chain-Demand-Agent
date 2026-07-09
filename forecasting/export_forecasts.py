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
import os


DATA = "data/supply_chain_data.csv"
OUT = "lib/data/forecasts.json"
CKPT_DIR = "forecasting/saved_model"


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
        out[part_id] = {
            "p10": [round(float(x), 1) for x in p10],
            "p50": [round(float(x), 1) for x in p50],
            "p90": [round(float(x), 1) for x in p90],
        }
        if i % 25 == 0 or i == len(part_ids):
            print(f"  {i}/{len(part_ids)} parts forecast")

    os.makedirs("lib/data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} TFT forecasts to {OUT}")


if __name__ == "__main__":
    main()
