"""
train.py
--------
This file trains the TFT model and logs everything to MLflow.

What is MLflow?
  MLflow is an experiment tracking tool. Every time you train a model,
  it automatically saves:
    - The hyperparameters you used (learning rate, hidden size, etc.)
    - The metrics at every epoch (training loss, validation loss)
    - The final trained model file
    - Plots of the predictions vs actual demand

  Why does this matter? Imagine training the model 5 times with different
  settings trying to improve accuracy. Without MLflow, you'd lose track of
  which run used which settings and which one performed best.
  MLflow gives you a dashboard at localhost:5000 where you can compare all
  your runs side by side. This is standard practice in industry.

What is PyTorch Lightning?
  Training a neural network requires a lot of boilerplate:
    - Loop over batches
    - Zero gradients
    - Forward pass
    - Compute loss
    - Backward pass
    - Update weights
    - Validate
    - Save checkpoints
    - Handle GPU/CPU automatically

  Writing all that every time is tedious and error-prone.
  PyTorch Lightning handles all of it. You just define your model
  and call trainer.fit(). Lightning does the rest.

  pytorch-forecasting's TFT is built on top of Lightning, so this
  integration is seamless.

How to run this file:
  From the project root:
    python -m forecasting.train

  Then open the MLflow dashboard:
    mlflow ui
  Go to http://localhost:5000 in your browser.
"""

import os
import warnings
import pandas as pd
import torch
import mlflow
import mlflow.pytorch

# pytorch-forecasting 1.7 builds on the standalone `lightning` package,
# NOT the older `pytorch_lightning` package. They share the same API but
# are different Python packages. Using pytorch_lightning here causes the
# "must be a LightningModule" TypeError because TFT inherits from
# lightning.pytorch.core.module, not pytorch_lightning.core.module.
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader
from pytorch_forecasting.data import TimeSeriesDataSet

from forecasting.model import load_and_prepare, build_dataset, build_model

# Suppress some verbose warnings from pytorch-forecasting internals
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------------------------

DATA_PATH = "data/supply_chain_data.csv"
MODEL_DIR = "forecasting/saved_model"
MLFLOW_URI = "mlruns"  # local folder where MLflow stores experiment data
EXPERIMENT_NAME = "supply-chain-tft"

BATCH_SIZE = 64  # number of training windows processed together in one step
MAX_EPOCHS = 30  # maximum training epochs (EarlyStopping may stop sooner)
NUM_WORKERS = 0  # data loading workers - keep 0 on Windows to avoid issues


def train():
    """
    Full training pipeline:
      1. Load and prepare data
      2. Build train/val datasets
      3. Build TFT model
      4. Train with PyTorch Lightning
      5. Log everything to MLflow
      6. Save the trained model
    """

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = load_and_prepare(DATA_PATH)
    print(f"  {len(df):,} rows loaded, {df['part_id'].nunique()} unique parts")

    # ------------------------------------------------------------------
    # STEP 2: Build datasets
    # ------------------------------------------------------------------
    print("Building TimeSeriesDataSet...")
    training_dataset, validation_dataset = build_dataset(df)

    # DataLoader wraps the dataset and feeds it to the model in batches.
    # shuffle=True on training means each epoch sees data in a different order,
    # which helps the model generalize rather than memorize sequence order.
    # Use pytorch-forecasting's built-in to_dataloader() instead of raw DataLoader.
    # This uses the correct collate function that handles target_scale numpy arrays
    # mixed with tensors - which is what causes the NoneType collate error.
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # STEP 3: Build model
    # ------------------------------------------------------------------
    print("Building TFT model...")
    model = build_model(training_dataset)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # STEP 4: Set up training callbacks
    # ------------------------------------------------------------------

    # EarlyStopping: monitors validation loss.
    # If it doesn't improve for 5 consecutive epochs, stop training.
    # This prevents wasting time and overfitting.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    # LearningRateMonitor: logs the current learning rate to MLflow at each epoch.
    # Useful to see if the scheduler is reducing LR as expected.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ModelCheckpoint: saves the model weights whenever validation loss improves.
    # So even if training continues for more epochs, we always keep the best version.
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # only keep the single best checkpoint
        verbose=True,
    )

    # ------------------------------------------------------------------
    # STEP 5: Configure MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # STEP 6: Train inside an MLflow run
    # ------------------------------------------------------------------
    # Everything inside `with mlflow.start_run()` is tracked automatically.
    with mlflow.start_run(run_name="tft-training") as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"View at: mlflow ui  →  http://localhost:5000\n")

        # Log our configuration so we can reproduce this run later
        mlflow.log_params(
            {
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "encoder_length": 90,
                "decoder_length": 30,
                "hidden_size": 64,
                "attention_heads": 4,
                "dropout": 0.1,
                "learning_rate": 3e-3,
                "n_parts": df["part_id"].nunique(),
                "n_training_rows": len(training_dataset),
            }
        )

        # The Trainer is PyTorch Lightning's main orchestrator.
        # It handles the training loop, GPU placement, gradient clipping,
        # and calling all the callbacks.
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=0.1,  # clip gradients to prevent exploding gradients
            callbacks=[early_stop, lr_monitor, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=5,
        )

        # This is where the actual training happens.
        # Lightning loops through epochs, calls forward/backward/optimizer steps,
        # runs validation, and fires all the callbacks automatically.
        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)

        # ------------------------------------------------------------------
        # STEP 7: Evaluate on validation set and log metrics
        # ------------------------------------------------------------------
        print("\nEvaluating best model on validation set...")

        # Load the best checkpoint (lowest val_loss)
        best_model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint.best_model_path
        )
        best_model.eval()

        # Get predictions on the validation set
        predictions = best_model.predict(
            val_loader,
            mode="quantiles",
            return_y=True,
        )

        # Calculate MAE on the median (p50) predictions
        # MAE = Mean Absolute Error = average of |predicted - actual|
        # Lower is better. Units are the same as demand (units/day).
        actuals = predictions.y[0].cpu().numpy().flatten()
        pred_p50 = predictions.output[:, :, 1].cpu().numpy().flatten()  # index 1 = p50

        mae = float(abs(actuals - pred_p50).mean())
        print(f"  Validation MAE (p50): {mae:.2f} units/day")

        # Log final metrics to MLflow
        mlflow.log_metrics(
            {
                "val_mae_p50": mae,
                "best_val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
                "epochs_trained": trainer.current_epoch,
            }
        )

        # ------------------------------------------------------------------
        # STEP 8: Log model artifact + register in MLflow Model Registry
        #
        # The Model Registry is MLflow's way of managing model versions.
        # Instead of just saving a file, you register it with a name and
        # a version number. You can then transition versions through stages:
        #   None → Staging → Production → Archived
        #
        # In a real company: a data scientist trains a new model (Staging),
        # it gets evaluated, and only then promoted to Production — the
        # version that actually serves predictions in the app.
        # ------------------------------------------------------------------
        MODEL_REGISTRY_NAME = "supply-chain-tft"

        # Log the model as an artifact first (required before registering)
        model_info = mlflow.pytorch.log_model(
            best_model,
            artifact_path="tft_model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        # Transition the newly registered version to Staging automatically.
        # A human (or CI pipeline) would then promote to Production after review.
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            MODEL_REGISTRY_NAME, stages=["None"]
        )
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"  Model registered: {MODEL_REGISTRY_NAME} v{version} → Staging")

            # Tag the version with key metadata so it's traceable
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME, version, "val_mae_p50", f"{mae:.4f}"
            )
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME,
                version,
                "epochs_trained",
                str(trainer.current_epoch),
            )
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME, version, "n_parts", str(df["part_id"].nunique())
            )

        print(f"\nTraining complete.")
        print(f"  Best model saved to: {checkpoint.best_model_path}")
        print(f"  MLflow run ID: {run.info.run_id}")
        print(f"  Model registered as: {MODEL_REGISTRY_NAME}")
        print(f"  Run: mlflow ui  →  http://localhost:5000")

    return checkpoint.best_model_path


# Avoid circular import issue with Lightning's multiprocessing
from pytorch_forecasting import TemporalFusionTransformer

if __name__ == "__main__":
    best_path = train()
    print(f"\nDone. Best checkpoint: {best_path}")
