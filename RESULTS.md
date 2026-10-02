# SOP evaluation: Supply-Chain-Demand-Agent

Branch `sop-eval` (base `0f94fd1`). Everything here can be reproduced from `eval_sop/`.
Raw outputs are in `eval_sop/results/`. Dates: 2026-10-01 to 2026-10-02. Machine: Windows 11 laptop, RTX 4060 Laptop 8 GB, shared with other jobs.

**Headline:** on this repo's **synthetic** data, the TFT beats every statistical baseline tried. Its MAE is 14% lower than AutoARIMA, the strongest baseline, and it sits 5.5% above an oracle that knows the data generator. Its 80% intervals are close to calibrated (79.6% coverage). Removing the static covariates (category, supplier, region, lead time, price) makes **no measurable difference** (ΔMAE = +0.004, 95% CI [−0.019, +0.026]). That is what the data generator implies. None of this tells us anything about real demand data.

---

## 1. Setup

### Data (provenance)
- `data/supply_chain_data.csv`: 200 parts × 1,461 days (2021-01-01 to 2024-12-31), 292,200 rows. It is **synthetic**, made by `data/generate_data.py` with `np.random.seed(42)`.
- `eval_sop/provenance_oracle.py` re-runs the generator in memory and confirms the committed CSV has identical `demand`, `part_id`, `category` and `inventory` columns (other columns were not compared) (`results/provenance.json`: `regenerated_csv_identical: true`).
- How demand is generated: base × (1 + 0.2·t/T) trend, plus a yearly sine (amplitude 0.3·base, with a different phase for each part), plus Gaussian noise (sd 0.15·base). About 20 i.i.d. spikes per part (×2 to ×4) are added, then the series is floored to an integer. There is **no weekly pattern**, and there are **0 zero-demand days** (not intermittent).
- Category, supplier and region are drawn independently of demand. A regression of each part's mean demand on them gives R² = 0.022, adjusted R² = −0.029. The correlation of mean demand with lead time is 0.02, and with price −0.08 (`provenance.json`).

### Protocol: rolling origin
- **Origins** (last observed `time_idx`): 1340, 1370, 1400, 1430. Each one forecasts the next 30 days. The four test windows are disjoint and cover the last 120 days (2024-09-03 to 2024-12-31).
- **n** = 200 parts × 4 origins = **800 part-windows (24,000 point forecasts) per model and seed**.
- Models see only history up to and including the origin. The TFT is **early-stopped on an inner window (T−29…T)** that lies entirely before the test window, so the test window is never used to select the model. The original `train.py` early-stopped on the same window it reported.
- **Metrics** (`eval_sop/metrics.py`): MAE of q50. MASE, which is MAE divided by the in-sample lag-1 naive MAE on history up to the origin, per part-window. Mean pinball loss over q ∈ {0.1, 0.5, 0.9}, which are the quantiles the TFT outputs. Coverage of [q10, q90], with 0.80 nominal. Mean interval width.
- **Uncertainty**: the TFT is reported as mean ± std over **3 seeds (0, 1, 2)**. The 95% CIs are a **part-level bootstrap** (2,000 resamples of the 200 parts, RNG seed 12345) of the metric averaged over origins and seeds. Model comparisons use a *paired* bootstrap over the same resampled parts.

### Models
| Model | Details |
|---|---|
| **TFT** (`full`) | Same architecture as `forecasting/model.py`: hidden 64, 4 heads, dropout 0.1, lr 3e-3, QuantileLoss [0.1, 0.5, 0.9], encoder 90, decoder 30, softplus GroupNormalizer. Static covariates: part_id, category, supplier, region, lead_time_days, price_usd. |
| **TFT_no_meta** | Ablation. Only part_id is kept as a static covariate. category, supplier, region, lead_time_days and price_usd are removed. |
| Naive, SeasonalNaive7, SeasonalNaive365 | statsforecast 2.1.1, 80% intervals |
| AutoETS (season 7), AutoARIMA (season 7) | statsforecast, fit on full history up to the origin |
| MSTL365 | statsforecast MSTL(season 365) with an AutoETS(ZZN) trend |
| CrostonClassic, TSB(0.1, 0.1) | Run because they were requested. **They are not appropriate here**, because the series are not intermittent. They are point forecasts only, so they have no pinball or coverage. With no zeros they reduce to SES, which is why their numbers are identical. |
| AppBaseline | Port of the app's own fallback (`lib/tools/forecast.ts`): mean of the last 60 days, a linear 0→5% trend, ±1.65·sd |
| Oracle (reference, **not achievable**) | floor(true trend + seasonality + z_τ·0.15·base), built from the generator's true parameters. It ignores spikes. It is a floor reference only. |

**TFT training budget (shorter than `forecasting/train.py`, on purpose):** at most 6 epochs × 150 batches × batch size 128, which is about 115k windows, or 0.4 of one full pass. Early stopping has patience 3 on the inner window, and the best inner-val checkpoint is used. The original trains for up to 30 epochs on full passes at batch size 64. A longer budget might do better; this was not tested.

### Devices and timings (all 24 TFT runs, from `results/tft_preds/*.json`)
| Runs | Device | Train time per run |
|---|---|---|
| full o1340, o1370 (s0–2); no_meta o1340, o1370 (s0–2), o1400 s0 | GPU, 2026-10-01. These JSONs have no `device` field because the flag was added later; they ran with `accelerator="gpu"`. | 448–907 s (the GPU was heavily shared) |
| full o1400 (s0–2) | CPU, 2 torch threads | 481–740 s |
| full o1430 (s0–2); no_meta o1400 s1–2, o1430 (s0–2) | GPU, `set_per_process_memory_fraction(0.25)`, peak about 253 MB allocated | 196–372 s |

CPU and GPU are not bit-identical. Seed std of MAE is 0.03 to 0.05. See the CPU replicate check in §3.4. Baselines took 955, 1188, 1447 and 1715 s per origin on CPU with 2 threads and n_jobs=1.

---

## 2. Results (n = 800 part-windows per model per seed)

| Model | MAE (± seed std) [95% CI] | MASE [95% CI] | Pinball (q.1/.5/.9) [95% CI] | 80% coverage | Width |
|---|---|---|---|---|---|
| Oracle (reference) | 6.504 [6.03, 6.95] | 0.656 [0.640, 0.671] | 2.257 [2.09, 2.43] | 0.829 | 17.1 |
| **TFT (full)** | **6.862 ± 0.027** [6.37, 7.33] | **0.693 ± 0.003** [0.676, 0.708] | **2.368 ± 0.007** [2.19, 2.54] | **0.796 ± 0.005** [0.790, 0.802] | 18.4 |
| **TFT_no_meta** | 6.859 ± 0.049 [6.37, 7.33] | 0.693 ± 0.005 [0.677, 0.709] | 2.367 ± 0.014 [2.19, 2.54] | 0.797 ± 0.004 [0.791, 0.802] | 18.4 |
| AutoARIMA | 8.012 [7.43, 8.61] | 0.809 [0.785, 0.833] | 2.998 [2.78, 3.22] | 0.961 | 38.6 |
| CrostonClassic / TSB | 8.021 [7.42, 8.64] | 0.808 [0.785, 0.833] | n/a | n/a | n/a |
| AutoETS | 8.155 [7.56, 8.76] | 0.823 [0.797, 0.850] | 3.097 [2.87, 3.32] | 0.961 | 40.9 |
| MSTL365 | 8.543 [7.94, 9.13] | 0.862 [0.844, 0.879] | 3.117 [2.89, 3.34] | 0.901 | 30.5 |
| AppBaseline (the app's fallback) | 9.521 [8.79, 10.22] | 0.958 [0.928, 0.985] | 3.484 [3.22, 3.75] | 0.860 | 41.0 |
| Naive | 9.997 [9.06, 11.15] | 1.012 [0.942, 1.094] | 8.186 [7.64, 8.78] | 0.986 | 189.8 |
| SeasonalNaive365 | 10.232 [9.50, 10.94] | 1.026 [1.003, 1.049] | 4.134 [3.83, 4.42] | 0.964 | 51.1 |
| SeasonalNaive7 | 10.297 [9.48, 11.13] | 1.036 [0.994, 1.078] | 4.926 [4.56, 5.31] | 0.974 | 80.5 |

Single-run baselines have no seed std. The CIs are wide in absolute units because MAE scales with each part's base demand (5 to 80 units/day). Paired differences are much tighter:

**Paired differences, TFT (full) minus other model** (`results/paired_diffs.json`):
| vs | ΔMAE [95% CI] | rel. | Δpinball [95% CI] |
|---|---|---|---|
| AutoARIMA | −1.150 [−1.332, −0.989] | −14.4% | −0.629 [−0.686, −0.577] (−21.0%) |
| AutoETS | −1.292 [−1.484, −1.106] | −15.8% | −0.728 [−0.797, −0.662] |
| Croston/TSB | −1.159 [−1.347, −0.995] | −14.4% | n/a |
| AppBaseline | −2.659 [−2.974, −2.353] | −27.9% | −1.116 [−1.228, −1.008] |
| SeasonalNaive7 | −3.434 [−3.919, −3.005] | −33.4% | −2.558 |
| Oracle | +0.359 [+0.313, +0.406] | +5.5% | +0.111 [+0.097, +0.126] |
| **TFT_no_meta (ablation)** | **+0.004 [−0.019, +0.026]** | +0.05% | +0.001 [−0.005, +0.007] |

TFT MAE by origin: 6.98 / 6.75 / 7.02 / 6.70. It is the best non-oracle model at every origin (`results/per_origin.csv`).

### 2.1 Export bug (verified, fixed)
- **Reproduced first.** `eval_sop/check_export_bug.py` builds the export's exact dataset, `from_dataset(..., part_df, predict=True)` on the full observed data. The decoder covers **time_idx 1431–1460 = 2024-12-02 to 2024-12-31, the last 30 *observed* days**, not the 30 days after the data ends (`results/export_bug_check.json`).
- Those days are also the validation window that `train.py` used for early stopping. So the committed `lib/data/forecasts.json` is an in-sample fit of already-observed data: p50 MAE 6.72 against those days, p10–p90 coverage 80.6%. The app shows it as "the next 30 days".
- A second bug was found in the same place: `ckpts[0]` after `sorted()` loads `epoch=00-val_loss=4.6037` instead of the better `epoch=01-val_loss=4.5668`. Both checkpoints exist in the original `forecasting/saved_model/`.
- `eval_sop/test_export_forecasts.py` runs the real `main()` with a stub model. It **FAILS on 0f94fd1** on both checks (`results/test_export_original_0f94fd1.txt`) and **PASSES after the fix** (`results/test_export_fixed.txt`). The fixed export was also run against the real local checkpoint, and its decoder is 1461–1490. `lib/data/forecasts.json` was **not** regenerated (see "Proposed, not done").

### 2.2 Agent harness (built; one run recorded on 2026-10-01)
- `eval_sop/agent_eval.py` is a **Python port** of the `app/api/chat/route.ts` loop. It keeps the same system prompt, the same 3 tools with the same descriptions and schemas, and the same 6-step cap, and it uses ports of `lib/tools/*.ts` over the same `lib/data/*.json`.
- `eval_sop/agent_questions.json` holds 30 templated questions: 20 inventory, 6 forecast and 4 knowledge-base. Answers are **computed by code from lib/data**, or copied verbatim from the human-written `docs.json`. **No LLM-generated labels.**
- Grading is automatic. A numeric answer passes if any number in the reply is within ±0.5 (integers) or ±0.05 (decimals). Risk labels must match exactly, and part-ID questions need all the IDs.
- **One run was done on 2026-10-01, before Ollama use was paused.** Model: `qwen2.5:7b`, Ollama 0.34.4, Q4_K_M, digest 845dbda0ea48, temperature 0, seed 0, on a local server.
  - Tool-selection accuracy was **30/30** (95% Clopper-Pearson [0.88, 1.00]). The first tool call was correct in 29/30.
  - Answer accuracy was **30/30** [0.88, 1.00] (`results/agent_eval_qwen2_5_7b.json`).
- A `llama3.1:8b` run was started but **did not finish**: its server was stopped mid-run, so there is **no result**.
- These questions are easy single-lookup templates, so this is a ceiling result. It shows the loop and tool relaying work. It says nothing about planning, multi-step reasoning, or robustness to phrasing.
- **Paid or other runs (not done):**
  - The app's default provider is Groq (`openai/gpt-oss-20b/120b`). That would be **$0 on the free tier**, but no Groq key exists on this machine.
  - Anthropic, 30 questions, about 2 calls each: roughly 80k input + 7k output tokens per pass. That is about **$0.12 on claude-haiku-4-5** ($1/$5 per MTok), **$0.23 on claude-sonnet-5** ($2/$10) and **$0.58 on claude-opus-4-8** ($5/$25).
  - These are estimates of token counts, not measurements. ×3 repeats would triple them.

### 2.3 Drift check (verified by reading; not changed)
- `lib/db/drift.ts:36–62` compares each logged `p50Daily` to `parts.json`'s `avg_daily_demand`. That is the **historical average**, not the realized demand over the forecast horizon.
- "Calibration" checks whether that historical average falls inside [p10Total/h, p90Total/h].
- So this is a consistency check between forecast and history. It is not a drift or accuracy monitor. The file's own comment calls it "demo-grade".

---

## 3. What the numbers do and don't support

### 3.1 Supported
- On this synthetic dataset, under a leakage-free rolling-origin protocol with 4 origins, 200 series and 3 seeds, the TFT has 14–16% lower MAE and 21–24% lower pinball loss than AutoARIMA and AutoETS. The paired 95% CIs exclude 0.
- The TFT is about 28% better than the app's own fallback baseline.
- Its 80% intervals cover 79.6% of the time. The statistical baselines over-cover (90–99%) with intervals 2 to 10 times wider, because the i.i.d. spikes inflate their residual variance.
- The TFT gets within 5.5% (MAE) of an oracle that knows the generator.
- Static metadata covariates do not help: the ablation difference is +0.05% MAE, with a CI spanning 0. This matches the data generator, where they carry no signal.

### 3.2 Not supported
- Any claim about **real** supply-chain data. The data is synthetic, and the generator happens to be easy for a global model that sees calendar features. The yearly sine is a function of day-of-year, which the TFT gets through `month`. The ARIMA and ETS models used here have only weekly seasonality.
- The README's "learns supplier-specific patterns" / "Valve parts from SupplierA behave like X". There are none to learn, and the ablation shows no effect.
- Claims of "significantly more accurate" *before this eval*: there was no baseline anywhere in the repo. The claim now holds only in the narrow synthetic sense above.
- Agent quality beyond single-lookup relaying (§2.2).
- That the deployed web forecasts are forecasts. Until `forecasts.json` is regenerated with the fixed export, they are in-sample.

### 3.3 Threats to validity
- **Synthetic data with a simple generator.** Results may not transfer. The M5 / real-data check (optional plan item 4) was **not done**, because the laptop was shared and the compute was limited.
- **Baselines.** The statistical models got default statsforecast settings with weekly seasonality only. MSTL365 was the only one with explicit yearly seasonality, and it did worse than ETS here. A tuned yearly-Fourier ARIMAX or dynamic regression might close the gap. That is untested.
- **TFT budget.** Training is short (about 0.4 of an epoch per run, see §1). A larger budget could change the absolute numbers, probably for the better.
- **Mixed devices.** Runs were split across GPU and CPU, and the 2026-10-01 runs shared the GPU with other jobs. Seed std is small (0.03 to 0.05 MAE). See §3.4.
- **Stale checkpoint directories.** Runs killed on 2026-10-01 (full o1400 s0 and no_meta o1400 s1) left checkpoint files behind. The reruns loaded their *own* best checkpoint: Lightning wrote `...-v1.ckpt` on a name clash, and `best_model_path` points to the current run.
- **Bootstrap.** Parts are resampled, so dependence across origins within a part is kept. The 4 origins are not resampled, so uncertainty about time periods is understated.
- **The oracle ignores spikes and uses Gaussian quantiles.** It is a reference point, not a true lower bound. Its coverage is 0.83.
- **The agent eval is a Python port**, not the TypeScript runtime. The grading is lenient: a number anywhere in the reply counts.

### 3.4 CPU vs GPU replicate
I re-ran `full`, origin 1430, seed 0 on CPU with 2 threads; it took 1,220 s to train, against 196 s on GPU. The replicate is stored separately in `results/cpu_replicate/` and is **not** used in the main table.

| | GPU (main table) | CPU replicate |
|---|---|---|
| MAE | 6.720 | 6.721 |
| Pinball | 2.302 | 2.311 |
| Coverage | 0.804 | 0.798 |

- The MAE difference is 0.0007. For comparison, the other seeds at the same origin give 6.687 and 6.695.
- The two runs picked different best epochs (GPU epoch 3, CPU epoch 2).
- Individual q50 values differ by 1.13 units on average.

**Conclusion:** device effects are well inside seed noise for aggregate metrics, and the runs are not bit-identical. Details are in `results/cpu_gpu_replicate.json`.

---

## 4. SOP-ready sentences (strictly true as of this commit)
1. "On a 200-series synthetic spare-parts dataset, I evaluated a Temporal Fusion Transformer with a leakage-free rolling-origin backtest (4 origins × 30 days, 3 seeds). It reduced MAE by 14% versus AutoARIMA (paired 95% CI 12–17%) and kept 80% prediction intervals near nominal coverage (79.6%), while staying within 6% of an oracle that knows the data generator."
2. "An ablation showed the model's static supplier, region and category covariates contributed nothing (ΔMAE +0.05%, CI spanning zero). That is consistent with how the synthetic data was generated, and it led me to retract the project's claim that the model learns supplier-specific patterns."
3. "Evaluating the system end to end, I found and fixed a bug where the exported 'future' forecasts were in-sample predictions of the last 30 observed days. I added a regression test that fails on the original code and passes after the fix."

---

## 5. Change log (one entry per change on this branch)

1. **`forecasting/export_forecasts.py`: forecast the future, and load the best checkpoint** (commit `28552cd`).
   - *What:* added `_append_future_rows()`, which appends 30 rows after the last observed day carrying only known covariates (calendar features and static attributes; the unknown reals are placeholders the decoder never sees). Added `_best_checkpoint()`, which picks the lowest `val_loss` from the filename. Added env overrides `EXPORT_CKPT_DIR` / `EXPORT_OUT` so the export can be tested without touching `lib/data/` or `forecasting/saved_model/`. Added a print of the first decoder range.
   - *Why:* the export forecast the in-sample window (§2.1), and `sorted()[0]` picked the oldest checkpoint, not the best one.
   - *Evidence:* the original lines were `export_forecasts.py:67` `from_dataset(training_ds, part_df, predict=True)` with `part_df = full_df[...]`, and `:50` `ckpt = ckpts[0]`. `eval_sop/test_export_forecasts.py` fails on 0f94fd1 and passes after the fix. The real-checkpoint export logs `decoder time_idx 1461..1490`. vitest is 17/17 before the commit.
   - *Preserved:* every existing comment and docstring, `_sanity_check`, the output format, and the defaults (`lib/data/forecasts.json`, `forecasting/saved_model`).
2. **`eval_sop/` (new, evaluation only)** (commits `18f6320` and `d89540d`).
   - *What:* the backtest, baselines, metrics, provenance, export check and agent harness.
   - *Why:* the eval itself.
   - *Preserved:* no product code touched.
   - *Later edits to eval scripts:* a CPU/GPU flag, a 2-thread cap, a VRAM fraction cap with OOM→CPU fallback, per-origin resumable baseline files, Croston/TSB added on request, gzip outputs, and a `SOP_RESULTS` env override for the CPU replicate.

### Deviations from the requested rules, disclosed
- **statsforecast was installed into a separate scratchpad venv** (`venv-sf`: Python 3.12.3 from anaconda, statsforecast 2.1.1, numpy 2.5.3, pandas 2.3.3, numba 0.68.0), **not** the repo's `venv/`. statsforecast pulls newer numpy and pandas. Installing it into `venv/` (numpy 1.26.4, pandas 2.1.4, torch 2.5.1+cu121, pytorch-forecasting 1.7.0, lightning 2.2.5) would have upgraded the dependencies the TFT stack is pinned to. Nothing global or system-wide was changed.
- `npm ci --ignore-scripts` was run **inside the worktree** to run the existing vitest suite (node 24.14.0). `node_modules/` is gitignored and not committed.
- The agent run on 2026-10-01 used Ollama *before* the "no Ollama for now" instruction arrived. It is reported as is, and no further LLM calls were made.

## 6. Proposed, not done
- **Regenerate `lib/data/forecasts.json`** with the fixed export and a properly trained checkpoint. Not done: it is product data, and the checkpoint is not in git.
- **`agent/agent.py:_forecast_with_tft`** has the same `predict=True`-on-observed-data bug and also uses `checkpoint_files[0]`. It is the Streamlit path. The fix would be to reuse `_append_future_rows` and `_best_checkpoint`.
- **`lib/db/drift.ts`**: compare logged forecasts to *realized* demand over the forecast window, which means storing the forecast date. The current check compares against the historical average.
- **`forecasting/train.py`**: early-stopping on the validation window and then reporting MAE on that same window is optimistic. Add a held-out test window or a rolling-origin evaluation, and set a seed (`pl.seed_everything`).
- Add Python tests to CI (`eval_sop/test_export_forecasts.py` is a start).

### Proposed README corrections
- Line 239, "the TFT model is significantly more accurate", and line 263, "much more accurate for parts with ... supplier-specific patterns": replace with the measured numbers above, and state that the data is synthetic.
- Line 249, "Learns 'Valve parts from SupplierA behave like X'": state that in the synthetic data these attributes carry no signal (ablation ΔMAE ≈ 0).
- Line 217, "Accuracy: High / Moderate": cite MAE 6.86 vs 9.52, i.e. TFT vs the app baseline, on synthetic data.
- Lines 340–346, drift: describe it as a forecast-vs-historical-average consistency check, not drift detection.
- Line 368, "Loads the best checkpoint": this is true only after this branch's fix.

## 7. Reproduce
```bash
# project venv (torch 2.5.1+cu121, pytorch-forecasting 1.7.0, lightning 2.2.5)
PY=venv/Scripts/python.exe
$PY -m eval_sop.check_export_bug
$PY -m eval_sop.test_export_forecasts                    # fixed; pass a path to test the original file
$PY -m eval_sop.provenance_oracle
OMP_NUM_THREADS=2 SOP_CKPT_DIR=<scratch> $PY -m eval_sop.tft_backtest --variant full    --accel gpu   # or --accel cpu
OMP_NUM_THREADS=2 SOP_CKPT_DIR=<scratch> $PY -m eval_sop.tft_backtest --variant no_meta --accel gpu
# separate venv: pip install statsforecast==2.1.1
OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 SF_JOBS=1 venv-sf/Scripts/python.exe -m eval_sop.baselines
venv-sf/Scripts/python.exe -m eval_sop.metrics           # -> summary.json, paired_diffs.json, per_origin.csv
$PY -m eval_sop.agent_eval --build-questions             # question set (already committed)
$PY -m eval_sop.agent_eval --model qwen2.5:7b --base-url http://127.0.0.1:11434/v1   # needs Ollama
```
Seeds: data 42, TFT 0/1/2 (`pl.seed_everything`), bootstrap 12345, question sampling 2024.
