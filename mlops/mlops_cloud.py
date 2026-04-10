"""
mlops/mlops_cloud.py
--------------------
Lightweight MLOps monitoring for Hugging Face Spaces deployment.
No MLflow required — uses a CSV file on HF persistent storage (/data).

On HF Spaces, /data is a persistent volume that survives restarts.
On local machines, falls back to a local logs/ directory.

Public API (mirrors mlops/monitor.py interface):
    log_prediction(part_id, p50_daily, p50_total, p10_total, p90_total, horizon_days, source)
    get_prediction_log(limit)   -> pd.DataFrame
    compute_drift_metrics()     -> dict
"""

import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Storage path — /data on HF Spaces, logs/ locally
# ---------------------------------------------------------------------------
_HF_DATA = "/data"
_LOCAL_DATA = os.path.join(os.path.dirname(__file__), "..", "logs")


def _log_dir() -> str:
    if os.path.isdir(_HF_DATA) and os.access(_HF_DATA, os.W_OK):
        return _HF_DATA
    os.makedirs(_LOCAL_DATA, exist_ok=True)
    return _LOCAL_DATA


def _log_path() -> str:
    return os.path.join(_log_dir(), "predictions.csv")


# Thread lock so concurrent Gradio requests don't corrupt the CSV
_lock = threading.Lock()

# Column schema
_COLUMNS = [
    "timestamp",
    "part_id",
    "source",  # "statistical" or "tft"
    "p50_daily",
    "p50_total",
    "p10_total",
    "p90_total",
    "horizon_days",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_prediction(
    part_id: str,
    p50_daily: float,
    p50_total: float,
    p10_total: float,
    p90_total: float,
    horizon_days: int = 30,
    source: str = "statistical",
) -> None:
    """Append one forecast to the prediction log CSV. Thread-safe."""
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "part_id": part_id,
        "source": source,
        "p50_daily": round(float(p50_daily), 2),
        "p50_total": round(float(p50_total), 1),
        "p10_total": round(float(p10_total), 1),
        "p90_total": round(float(p90_total), 1),
        "horizon_days": int(horizon_days),
    }
    try:
        with _lock:
            path = _log_path()
            file_exists = os.path.isfile(path)
            df_new = pd.DataFrame([row], columns=_COLUMNS)
            df_new.to_csv(
                path,
                mode="a",
                header=not file_exists,
                index=False,
            )
    except Exception:
        pass  # never crash the app over a logging failure


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------
def get_prediction_log(limit: int = 100) -> pd.DataFrame:
    """Return the last `limit` prediction log rows, newest first."""
    path = _log_path()
    if not os.path.isfile(path):
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(path)
        df = df.tail(limit).iloc[::-1].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=_COLUMNS)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def compute_drift_metrics(data_path: str = "data/supply_chain_data.csv") -> dict:
    """
    Compare logged p50 forecasts against actual demand from the CSV.

    Returns a dict with:
        n_predictions   int     total predictions logged
        mae             float   mean absolute error (forecast p50/day vs actual avg)
        calibration     float   % of actuals inside p10/30-p90/30 daily band
        drift_flag      bool    True if MAE > 1.5x the naive baseline MAE
        baseline_mae    float   naive baseline MAE (predict mean demand for all parts)
        status          str     "OK" / "WARNING" / "NO DATA"
    """
    empty = {
        "n_predictions": 0,
        "mae": None,
        "calibration": None,
        "drift_flag": False,
        "baseline_mae": None,
        "status": "NO DATA",
    }

    log = get_prediction_log(limit=500)
    if log.empty or len(log) < 3:
        return empty

    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
    except Exception:
        return empty

    latest = df["date"].max()
    last_30 = df[df["date"] >= latest - pd.Timedelta(days=30)]
    actual_avg = last_30.groupby("part_id")["demand"].mean().rename("actual_avg")

    # Baseline MAE: predict the global mean for every part
    global_mean = float(actual_avg.mean())
    baseline_mae = float((actual_avg - global_mean).abs().mean())

    # Match logged predictions to actuals
    log = log.merge(actual_avg.reset_index(), on="part_id", how="inner")
    if log.empty:
        return empty

    # MAE: compare p50_daily to actual_avg
    log["error"] = (log["p50_daily"] - log["actual_avg"]).abs()
    mae = float(log["error"].mean())

    # Calibration: % of actuals inside [p10/30, p90/30] daily band
    log["p10_daily"] = log["p10_total"] / log["horizon_days"]
    log["p90_daily"] = log["p90_total"] / log["horizon_days"]
    inside = (log["actual_avg"] >= log["p10_daily"]) & (
        log["actual_avg"] <= log["p90_daily"]
    )
    calibration = float(inside.mean() * 100)

    # Drift flag: MAE more than 50% worse than baseline
    drift_flag = bool(baseline_mae > 0 and mae > 1.5 * baseline_mae)
    status = "WARNING" if drift_flag else "OK"

    return {
        "n_predictions": int(len(log)),
        "mae": round(mae, 2),
        "calibration": round(calibration, 1),
        "drift_flag": drift_flag,
        "baseline_mae": round(baseline_mae, 2),
        "status": status,
    }
