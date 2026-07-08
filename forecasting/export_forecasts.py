"""
forecasting/export_forecasts.py
Runs the trained TFT for every part and writes lib/data/forecasts.json,
which the Vercel app serves as real TFT forecasts. Run locally after training:
    python -m forecasting.export_forecasts
Requires requirements-local.txt and a checkpoint in forecasting/saved_model/.
"""
import glob
import json
import os

import pandas as pd

DATA = "data/supply_chain_data.csv"
OUT = "lib/data/forecasts.json"
CKPT_DIR = "forecasting/saved_model"


def main() -> None:
    ckpts = glob.glob(f"{CKPT_DIR}/*.ckpt")
    if not ckpts:
        raise SystemExit("No checkpoint found. Train first: python -m forecasting.train")

    from agent.agent import _forecast_with_tft  # reuse the existing TFT path

    df = pd.read_csv(DATA, parse_dates=["date"])
    out = {}
    for part_id in sorted(df["part_id"].unique()):
        part_data = df[df["part_id"] == part_id].sort_values("date")
        text = _forecast_with_tft(part_id, part_data, ckpts[0])
        # Same split(":")[0]/[1] parsing agent.py's _log_forecast_to_mlflow uses
        # against _format_forecast's output - keys below match that function
        # exactly: "Daily demand (median)", "Total 30-day demand",
        # "Lower bound (p10)", "Upper bound (p90)".
        lines = {
            line.split(":")[0].strip(): line.split(":")[1].strip()
            for line in text.split("\n")
            if ":" in line
        }
        p50_daily = float(lines["Daily demand (median)"].split()[0])
        p50 = float(lines["Total 30-day demand"].split()[0].replace(",", ""))
        p10 = float(lines["Lower bound (p10)"].split()[0].replace(",", ""))
        p90 = float(lines["Upper bound (p90)"].split()[0].replace(",", ""))
        out[part_id] = {
            "p10": round(p10),
            "p50": round(p50),
            "p90": round(p90),
            "p50Daily": round(p50_daily, 2),
        }
        print(f"{part_id}: p50={p50:.0f}")

    os.makedirs("lib/data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} forecasts to {OUT}")


if __name__ == "__main__":
    main()
