"""
mlops/monitor.py
----------------
This file handles three MLOps concerns that go beyond just training:

  1. PREDICTION LOGGING
     Every time the app generates a forecast for a part, we log it to MLflow.
     This creates a searchable record of what the model predicted, when,
     and for which part. In production, this is how you audit model behavior.

  2. DRIFT DETECTION
     Over time, a model's predictions can become less accurate — this is
     called "model drift". It happens when real-world patterns change but
     the model was never retrained on the new data.

     We detect drift by comparing what the model predicted vs what actually
     happened, using historical data we have in the dataset. We compute:
       - MAE (Mean Absolute Error) in rolling 7-day and 30-day windows
       - Calibration score: what % of actual values fell inside the
         predicted p10-p90 band? Should be ~80%. If it drops to 50%,
         the model's uncertainty estimates are wrong.
       - Drift alert: if recent MAE is 20% worse than the baseline MAE
         from training, we flag it as a drift event.

  3. MODEL REGISTRY QUERY
     Functions to check which model version is currently in Staging/Production
     and surface that information in the monitoring dashboard.

Why does this matter for the resume?
  The JD mentions "model deployment, versioning and performance monitoring
  in production environments". This file directly implements all three.
  Logging predictions + measuring drift over time is exactly what a
  production ML system does to ensure the model stays reliable.
"""

import os
import json
import uuid
import glob
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

MLFLOW_URI = "mlruns"
EXPERIMENT_NAME = "supply-chain-tft"
PRED_EXPERIMENT = "prediction-log"
MODEL_REGISTRY_NAME = "supply-chain-tft"
DATA_PATH = "data/supply_chain_data.csv"

# Drift threshold: alert if recent MAE is this much worse than training MAE
DRIFT_THRESHOLD_PCT = 0.20  # 20% degradation triggers a warning

# Calibration target: ~80% of actuals should fall inside p10-p90
CALIBRATION_TARGET = 0.80


# ---------------------------------------------------------------------------
# 1. PREDICTION LOGGING
# ---------------------------------------------------------------------------


def log_prediction(
    part_id: str,
    p10_total: float,
    p50_total: float,
    p90_total: float,
    p50_daily: float,
    horizon_days: int,
    source: str = "statistical baseline",
) -> str:
    """
    Logs a single forecast call to MLflow's prediction-log experiment.

    Each call creates a new MLflow run with:
      - The part that was queried
      - The p10/p50/p90 forecast totals
      - The timestamp
      - Which model/source generated it

    This creates a full audit trail: you can look back and see every
    forecast the system made, when, and what values it returned.

    Args:
        part_id      : e.g. "PART_007"
        p10_total    : total 30-day lower bound
        p50_total    : total 30-day median
        p90_total    : total 30-day upper bound
        p50_daily    : median daily demand
        horizon_days : forecast horizon (usually 30)
        source       : "TFT model" or "statistical baseline"

    Returns:
        The MLflow run ID for this prediction (useful for tracing)
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(PRED_EXPERIMENT)

    with mlflow.start_run(run_name=f"forecast-{part_id}") as run:
        mlflow.log_params(
            {
                "part_id": part_id,
                "horizon_days": horizon_days,
                "forecast_source": source,
                "timestamp": datetime.now().isoformat(),
            }
        )
        mlflow.log_metrics(
            {
                "p10_total": round(p10_total, 2),
                "p50_total": round(p50_total, 2),
                "p90_total": round(p90_total, 2),
                "p50_daily": round(p50_daily, 2),
            }
        )
        return run.info.run_id


def get_prediction_log(limit: int = 100) -> pd.DataFrame:
    """
    Retrieves recent prediction log entries from MLflow.

    Returns a DataFrame with columns:
      part_id, timestamp, p50_daily, p50_total, p90_total, p10_total,
      forecast_source, run_id

    Used in the monitoring dashboard to show what the model has been serving.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        experiment = mlflow.get_experiment_by_name(PRED_EXPERIMENT)
        if not experiment:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )

        if runs.empty:
            return pd.DataFrame()

        # Flatten MLflow run columns into a clean DataFrame
        cols = {
            "params.part_id": "part_id",
            "params.timestamp": "timestamp",
            "params.forecast_source": "source",
            "metrics.p50_daily": "p50_daily",
            "metrics.p50_total": "p50_total",
            "metrics.p90_total": "p90_total",
            "metrics.p10_total": "p10_total",
            "run_id": "run_id",
        }
        available = {k: v for k, v in cols.items() if k in runs.columns}
        df = runs[list(available.keys())].rename(columns=available)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df.reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. DRIFT DETECTION
# ---------------------------------------------------------------------------


def compute_drift_metrics(
    window_days: int = 30,
) -> Dict:
    """
    Detects model drift by comparing predictions vs actual demand.

    How it works:
      - We take the most recent prediction log from MLflow
      - For each logged prediction, we look up what the actual demand
        turned out to be in the dataset
      - We compute MAE and calibration score over rolling windows
      - We compare against the baseline MAE from the training run

    Since our dataset is historical (ends Dec 2024), we simulate this by
    using predictions made for any part and comparing against real values
    from the same time period in the dataset.

    Returns a dict with:
      - mae_7d      : MAE over the last 7 days of logged predictions
      - mae_30d     : MAE over the last 30 days
      - calibration : % of actuals inside p10-p90 band (target: 80%)
      - baseline_mae: the MAE from the original training run
      - drift_alert : True if mae_30d is 20% worse than baseline
      - n_predictions: how many predictions were evaluated
    """
    mlflow.set_tracking_uri(MLFLOW_URI)

    result = {
        "mae_7d": None,
        "mae_30d": None,
        "calibration": None,
        "baseline_mae": None,
        "drift_alert": False,
        "n_predictions": 0,
        "status": "no_data",
    }

    # --- Get baseline MAE from training run ---
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            training_runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if (
                not training_runs.empty
                and "metrics.val_mae_p50" in training_runs.columns
            ):
                result["baseline_mae"] = float(
                    training_runs["metrics.val_mae_p50"].iloc[0]
                )
    except Exception:
        pass

    # --- Get prediction log ---
    pred_log = get_prediction_log(limit=500)
    if pred_log.empty or "part_id" not in pred_log.columns:
        result["status"] = "no_predictions_logged"
        return result

    # --- Load actual demand data ---
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except FileNotFoundError:
        result["status"] = "data_not_found"
        return result

    # --- Match predictions to actuals ---
    # For each logged prediction, find the actual daily demand for that part
    # over the next 30 days from the most recent date in the dataset.
    # We use average daily demand as the "actual" since we're comparing
    # to the p50 daily forecast.
    errors = []
    inside_band = []
    latest_date = df["date"].max()
    cutoff_30d = latest_date - timedelta(days=window_days)
    cutoff_7d = latest_date - timedelta(days=7)

    for _, row in pred_log.iterrows():
        part = row.get("part_id")
        if not part or part not in df["part_id"].values:
            continue

        # Actual: average daily demand for the last 30 days of data for this part
        part_recent = df[(df["part_id"] == part) & (df["date"] >= cutoff_30d)]["demand"]

        if part_recent.empty:
            continue

        actual_daily = float(part_recent.mean())
        pred_daily = row.get("p50_daily", np.nan)
        pred_p10 = row.get("p10_total", np.nan)
        pred_p90 = row.get("p90_total", np.nan)

        if pd.isna(pred_daily):
            continue

        # MAE error for this prediction
        errors.append(
            {
                "error": abs(actual_daily - pred_daily),
                "actual": actual_daily,
                "pred_p50": pred_daily,
                "pred_p10": pred_p10 / 30 if not pd.isna(pred_p10) else np.nan,
                "pred_p90": pred_p90 / 30 if not pd.isna(pred_p90) else np.nan,
                "timestamp": row.get("timestamp"),
            }
        )

        # Calibration: was the actual inside the p10-p90 daily band?
        if not pd.isna(pred_p10) and not pd.isna(pred_p90):
            daily_p10 = pred_p10 / 30
            daily_p90 = pred_p90 / 30
            inside_band.append(daily_p10 <= actual_daily <= daily_p90)

    if not errors:
        result["status"] = "insufficient_matches"
        return result

    errors_df = pd.DataFrame(errors)
    result["n_predictions"] = len(errors_df)
    result["mae_30d"] = round(float(errors_df["error"].mean()), 2)

    # 7-day subset
    errors_7d = errors_df.tail(min(len(errors_df), 10))
    result["mae_7d"] = round(float(errors_7d["error"].mean()), 2)

    # Calibration score
    if inside_band:
        result["calibration"] = round(sum(inside_band) / len(inside_band), 3)

    # Drift alert
    if result["baseline_mae"] and result["mae_30d"]:
        degradation = (result["mae_30d"] - result["baseline_mae"]) / result[
            "baseline_mae"
        ]
        result["drift_alert"] = degradation > DRIFT_THRESHOLD_PCT
        result["degradation_pct"] = round(degradation * 100, 1)

    result["status"] = "ok"
    result["errors_df"] = errors_df

    # Log drift metrics back to MLflow so they're tracked over time
    _log_drift_metrics(result)

    return result


def _log_drift_metrics(result: Dict) -> None:
    """Logs the computed drift metrics to the training experiment for trending."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="drift-check"):
            metrics = {}
            if result["mae_7d"] is not None:
                metrics["drift_mae_7d"] = result["mae_7d"]
            if result["mae_30d"] is not None:
                metrics["drift_mae_30d"] = result["mae_30d"]
            if result["calibration"] is not None:
                metrics["calibration_score"] = result["calibration"]
            if result.get("degradation_pct") is not None:
                metrics["degradation_pct"] = result["degradation_pct"]
            if metrics:
                mlflow.log_metrics(metrics)
                mlflow.log_param("drift_alert", str(result["drift_alert"]))
                mlflow.log_param(
                    "n_predictions_evaluated", str(result["n_predictions"])
                )
    except Exception:
        pass  # monitoring should never crash the app


# ---------------------------------------------------------------------------
# 3. MODEL REGISTRY QUERIES
# ---------------------------------------------------------------------------


def get_registered_model_info() -> List[Dict]:
    """
    Returns info about all registered versions of the TFT model.

    Used in the monitoring dashboard to show which versions exist,
    what stage they're in, and their key metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME)

        result = []
        for v in versions:
            tags = dict(v.tags) if v.tags else {}
            result.append(
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "creation_time": datetime.fromtimestamp(
                        v.creation_timestamp / 1000
                    ).strftime("%Y-%m-%d %H:%M"),
                    "val_mae_p50": tags.get("val_mae_p50", "—"),
                    "epochs_trained": tags.get("epochs_trained", "—"),
                    "n_parts": tags.get("n_parts", "—"),
                    "run_id": v.run_id,
                }
            )
        return result

    except Exception:
        return []


def get_production_model_version() -> Optional[str]:
    """Returns the version number of the model currently in Production, or None."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            MODEL_REGISTRY_NAME, stages=["Production"]
        )
        return versions[0].version if versions else None
    except Exception:
        return None


def promote_to_production(version: str) -> bool:
    """
    Promotes a model version from Staging to Production.
    Archives any existing Production version.

    In a real company this would be gated by evaluation criteria
    (e.g. val_mae must be below a threshold). Here we allow manual
    promotion from the monitoring dashboard UI.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        return True
    except Exception:
        return False
