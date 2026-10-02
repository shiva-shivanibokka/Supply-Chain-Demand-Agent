"""Rolling-origin backtest of the project's TFT.

For each origin T (last observed day) and seed:
  * train on time_idx <= T-30, early-stop on the inner window T-29..T
    (so the test window is never used for model selection),
  * forecast T+1..T+30 using only history <= T plus known calendar features.

Same architecture/hyper-parameters as forecasting/model.py (hidden 64, 4 heads,
dropout 0.1, lr 3e-3, QuantileLoss [0.1,0.5,0.9], encoder 90, decoder 30,
softplus GroupNormalizer). Training budget is SHORTER than forecasting/train.py
(see --max-epochs / --batches-per-epoch); this is documented in RESULTS.md.

Variants:
  full     : static_categoricals = part_id, category, supplier, region;
             static_reals = lead_time_days, price_usd   (as in forecasting/model.py)
  no_meta  : static_categoricals = part_id only; no static reals
             (ablation: removes category/supplier/region/lead_time/price)

Usage:
  python -m eval_sop.tft_backtest --variant full --seeds 0 1 2
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from eval_sop.common import DATA, RESULTS, HORIZON, ORIGINS, QUANTILES
from forecasting.model import load_and_prepare, ENCODER_LENGTH, DECODER_LENGTH

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
# Footprint limit (laptop shared with the user): 2 CPU threads by default.
THREADS = int(os.environ.get("SOP_THREADS", "2"))
torch.set_num_threads(THREADS)


def make_datasets(df, origin, variant):
    if variant == "full":
        static_cats = ["part_id", "category", "supplier", "region"]
        static_reals = ["lead_time_days", "price_usd"]
    elif variant == "no_meta":
        static_cats = ["part_id"]
        static_reals = []
    else:
        raise ValueError(variant)
    hist = df[df["time_idx"] <= origin]
    inner_cut = origin - HORIZON
    training = TimeSeriesDataSet(
        data=hist[hist["time_idx"] <= inner_cut],
        group_ids=["part_id"],
        time_idx="time_idx",
        target="demand",
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=DECODER_LENGTH,
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_categoricals=["month", "day_of_week", "quarter"],
        time_varying_unknown_reals=["demand", "inventory"],
        target_normalizer=GroupNormalizer(groups=["part_id"], transformation="softplus"),
    )
    # inner validation: decoder = origin-29..origin (all <= origin, no test leakage)
    val = TimeSeriesDataSet.from_dataset(training, hist, predict=True, stop_randomization=True)
    # test: decoder = origin+1..origin+30. Decoder rows only feed KNOWN covariates
    # (month/dow/quarter); demand/inventory are "unknown" reals and are not
    # given to the decoder, so including those rows does not leak the target.
    test_df = df[df["time_idx"] <= origin + HORIZON]
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
    return training, val, test


def run_one(datasets, origin, seed, variant, args):
    pl.seed_everything(seed, workers=True)
    if args.accel == "gpu":
        torch.cuda.reset_peak_memory_stats()
    training, val, test = datasets
    tl = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    vl = val.to_dataloader(train=False, batch_size=256, num_workers=0)
    te = test.to_dataloader(train=False, batch_size=256, num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    ckdir = os.path.join(args.ckpt_dir, f"{variant}_o{origin}_s{seed}")
    ck = ModelCheckpoint(dirpath=ckdir, monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=args.batches_per_epoch,
        accelerator=args.accel,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min"), ck],
        logger=CSVLogger(args.ckpt_dir, name=f"log_{variant}_o{origin}_s{seed}"),
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=False,
    )
    t0 = time.time()
    trainer.fit(model, tl, vl)
    train_s = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if args.accel == "gpu" else None
    best = TemporalFusionTransformer.load_from_checkpoint(ck.best_model_path)
    best.eval()
    pred = best.predict(te, mode="quantiles", return_index=True,
                        trainer_kwargs=dict(accelerator=args.accel,
                                            enable_progress_bar=False, logger=False))
    q = pred.output.cpu().numpy()  # [n_series, 30, 3]
    idx = pred.index.reset_index(drop=True)
    assert (idx["time_idx"] == origin + 1).all(), "test decoder must start at origin+1"
    rows = []
    for i, part in enumerate(idx["part_id"]):
        for h in range(HORIZON):
            rows.append((variant, seed, origin, part, h + 1, origin + 1 + h,
                         float(q[i, h, 0]), float(q[i, h, 1]), float(q[i, h, 2])))
    out = pd.DataFrame(rows, columns=["model", "seed", "origin", "part_id", "h", "time_idx",
                                      "q10", "q50", "q90"])
    meta = dict(variant=variant, seed=seed, origin=origin, train_seconds=round(train_s, 1),
                epochs_run=int(trainer.current_epoch),
                best_inner_val_loss=float(ck.best_model_score.cpu()),
                best_ckpt=os.path.basename(ck.best_model_path),
                n_train_samples=len(training), batches_per_epoch=args.batches_per_epoch,
                batch_size=args.batch_size, max_epochs=args.max_epochs,
                device=args.accel, torch_threads=THREADS, cuda_peak_alloc_mb=peak_mb, torch_version=torch.__version__)
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "no_meta"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--origins", type=int, nargs="+", default=ORIGINS)
    ap.add_argument("--max-epochs", type=int, default=6)
    ap.add_argument("--batches-per-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ckpt-dir", default=os.environ.get("SOP_CKPT_DIR", "eval_sop_ckpts"))
    ap.add_argument("--accel", default="cpu", choices=["cpu", "gpu"],
                    help="runs dated 2026-10-01 used gpu; later runs cpu (see RESULTS.md)")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.25,
                    help="cap on this process's share of VRAM (GPU shared with Ollama)")
    args = ap.parse_args()
    if args.accel == "gpu":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)

    df = load_and_prepare(DATA)
    os.makedirs(RESULTS, exist_ok=True)
    for origin in args.origins:
        datasets = None
        for seed in args.seeds:
            tag = f"tft_{args.variant}_o{origin}_s{seed}"
            path = os.path.join(RESULTS, "tft_preds", tag + ".csv.gz")
            if os.path.exists(path) or os.path.exists(path[:-3]):
                print("skip (exists)", tag)
                continue
            if datasets is None:
                datasets = make_datasets(df, origin, args.variant)
            try:
                out, meta = run_one(datasets, origin, seed, args.variant, args)
            except torch.cuda.OutOfMemoryError:
                # VRAM is shared with Ollama (priority): fall back to CPU for this run only.
                print("CUDA OOM -> rerunning on CPU:", tag, flush=True)
                torch.cuda.empty_cache()
                accel, args.accel = args.accel, "cpu"
                out, meta = run_one(datasets, origin, seed, args.variant, args)
                args.accel = accel
                meta["oom_fallback_to_cpu"] = True
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out.to_csv(path, index=False, float_format="%.4f")
            with open(os.path.join(RESULTS, "tft_preds", tag + ".json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
