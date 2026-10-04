# SOP evaluation: Supply-Chain-Demand-Agent

Branch `sop-eval` (base `0f94fd1`). Everything here can be reproduced from `eval_sop/`.
Raw outputs are in `eval_sop/results/`. Dates: 2026-10-01 to 2026-10-02. Machine: Windows 11 laptop, RTX 4060 Laptop 8 GB, shared with other jobs.

**Headline:** on this repo's **synthetic** data, the TFT beats every statistical baseline tried. Its MAE is 14% lower than AutoARIMA, the strongest baseline (95% CI 13–16%). It sits about 5.5% above a spike-agnostic oracle built from the data generator (95% CI 4.9–6.1%). That oracle is not a lower bound: its own 80% coverage is 0.829. Its 80% intervals are close to calibrated (79.6% coverage). Removing the static covariates (category, supplier, region, lead time, price) makes **no measurable difference** (ΔMAE = +0.004, 95% CI [−0.019, +0.026]). That is what the data generator implies. None of this tells us anything about real demand data.

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
| vs | ΔMAE [95% CI] | rel. MAE [95% CI] | Δpinball [95% CI] (rel. [95% CI]) |
|---|---|---|---|
| AutoARIMA | −1.150 [−1.332, −0.989] | −14.4% [−15.9, −12.8] | −0.629 [−0.686, −0.577] (−21.0% [−21.9, −20.1]) |
| AutoETS | −1.292 [−1.484, −1.106] | −15.8% [−17.5, −14.3] | −0.728 [−0.797, −0.662] (−23.5% [−24.7, −22.3]) |
| Croston/TSB | −1.159 [−1.347, −0.995] | −14.4% [−16.0, −12.9] | n/a |
| AppBaseline | −2.659 [−2.974, −2.353] | −27.9% [−29.8, −26.1] | −1.116 [−1.228, −1.008] (−32.0% [−33.4, −30.6]; see the caveat below) |
| SeasonalNaive7 | −3.434 [−3.919, −3.005] | −33.4% [−35.7, −30.9] | −2.558 |
| Oracle | +0.359 [+0.313, +0.406] | +5.5% [+4.9, +6.1] | +0.111 [+0.097, +0.126] (+4.9% [+4.4, +5.5]) |
| **TFT_no_meta (ablation)** | **+0.004 [−0.019, +0.026]** | +0.05% [−0.28, +0.38] | +0.001 [−0.005, +0.007] |

Relative CIs come from a paired, part-clustered bootstrap of 100·(mean_TFT / mean_other − 1): 2,000 resamples of the 200 parts, seed 12345, the same resamples as the ΔMAE CIs (`rel_diff_pct_ci95` in `paired_diffs.json`).

**AppBaseline caveat:** the app's band is ±1.65·sd, which is nominally a **90%** interval (5th to 95th percentile). It is scored here as q10/q90, so its lower and upper quantiles sit too far out for pinball at 0.1 and 0.9. That penalises AppBaseline's pinball loss and inflates the −32% pinball gap; the MAE gap (−27.9%, which uses only the median) is not affected. Its coverage (0.860) is likewise against an 80% target it was not designed for.

TFT MAE by origin: 6.98 / 6.75 / 7.02 / 6.70. It is the best non-oracle model at every origin (`results/per_origin.csv`).

### 2.1 Export bug (verified, fixed)
- **Reproduced first.** `eval_sop/check_export_bug.py` builds the export's exact dataset, `from_dataset(..., part_df, predict=True)` on the full observed data. The decoder covers **time_idx 1431–1460 = 2024-12-02 to 2024-12-31, the last 30 *observed* days**, not the 30 days after the data ends (`results/export_bug_check.json`).
- Those days are also the validation window that `train.py` used for early stopping. So the committed `lib/data/forecasts.json` is an in-sample fit of already-observed data: p50 MAE 6.72 against those days, p10–p90 coverage 80.6%. The app shows it as "the next 30 days".
- A second bug was found in the same place: `ckpts[0]` after `sorted()` loads `epoch=00-val_loss=4.6037` instead of the better `epoch=01-val_loss=4.5668`. Both checkpoints exist in the original `forecasting/saved_model/`.
- `eval_sop/test_export_forecasts.py` runs the real `main()` with a stub model. It **FAILS on 0f94fd1** on both checks (`results/test_export_original_0f94fd1.txt`) and **PASSES after the fix** (`results/test_export_fixed.txt`). The fixed export was also run against the real local checkpoint, and its decoder is 1461–1490. **`lib/data/forecasts.json` was NOT regenerated**, so the deployed web app still serves the old predictions of the last 30 observed days (see "Proposed, not done").
- The **Streamlit/Python agent path** (`agent/agent.py`, `get_demand_forecast` at line 260 in `0f94fd1`, `_forecast_with_tft` at line 322) had the same two bugs. It is now fixed on this branch (change log 8). `eval_sop/test_agent_forecast_path.py` fails on the old code (decoder 1431–1460, epoch=00 checkpoint; `results/test_agent_forecast_path_before.txt`) and passes after the fix (decoder 1461–1490, best checkpoint; `results/test_agent_forecast_path_after.txt`).

### 2.2 Agent eval (local Ollama; 3 seeds)

**Harness.**
- `eval_sop/agent_eval.py` is a **Python port** of the `app/api/chat/route.ts` loop. It keeps the same system prompt, the same 3 tools with the same descriptions and schemas, and the same 6-step cap, and it uses ports of `lib/tools/*.ts` over the same `lib/data/*.json`.
- **Labels are computed by code** from `lib/data/*.json`, or copied verbatim from the human-written `docs.json`. **No LLM-generated labels.**

**Question sets.**
- **Original 30** (`agent_questions.json`, written before any model was run): 20 inventory, 6 forecast and 4 knowledge-base single-lookup templates.
- **Added hard 14** (`agent_questions_hard.json`, **added on 2026-10-02 after the ceiling result**, ids H1–H14):
  - inventory differences and sums across two parts (H1–H5);
  - stock vs median forecast (H6–H8) and order quantity = p90 − stock (H9–H10), which need **two tools and arithmetic**;
  - "who supplies X, and what is that supplier's on-time rate" (H11–H12), which needs the inventory tool **and** the knowledge base;
  - the minimum days of supply among 3 parts (H13);
  - a count of CRITICAL parts in the top 10 (H14).
- Forecast answers for H6–H10 use the **corrected export** (`results/forecasts_fixed_export.json`, the 30 days after 2024-12-31), and the forecast tool is pointed at it with `--forecasts`. With the committed in-sample `lib/data/forecasts.json`, the correct answers would be different numbers.

**Scoring.** The answer grader was written before the runs (this cannot be verified from git, because the harness commit `e406734` and the results commit `c9220ec` landed at the same time). It was corrected after the runs (see change log 5), and every number below is **re-graded from the raw final answers** by `eval_sop/agent_summary.py`:
- A numeric answer passes if any number in the reply is within ±0.5 (integers) or ±0.05 (decimals).
- For H1–H3 and H6–H8, the **direction must also be right**: either a signed number, or direction words in the sentence that holds the number ("more", "exceeds" vs "fewer", "short"), taking into account which part is the sentence's subject. The original grader compared only absolute values.
- Risk labels must match exactly. ID questions need every ID.

The grader is lenient. Most visibly, an answer that lists *every* supplier's rate passes H11 and H12. So I also report **joint** (all required tool calls made with the right part_id **and** the answer passes). Joint was added after seeing outputs, as a stricter view; it is not a re-grade.

**Settings.**
- Ollama 0.34.4, main server on :11434, one request at a time, native `/api/chat` with `num_ctx = 8192`. All prompts fit, with no truncation needed: the system prompt, tools and up to 3 KB documents total about 2–3k tokens.
- temperature 0.7, seeds 0/1/2. The model was unloaded afterwards with `keep_alive: 0`.
- Models: `llama3.1:8b` (Q4_K_M) and `qwen2.5:7b` (Q4_K_M, digest 845dbda0ea48).
- 95% CIs are a **question-level bootstrap** (2,000 resamples, seed 12345) of per-question scores averaged over seeds (`results/agent_summary.json`). With 14 or 30 questions, these CIs are wide.

| Model | Set | Tool selection (mean ± seed std) [95% CI] | Answer accuracy [95% CI] | Joint [95% CI] |
|---|---|---|---|---|
| llama3.1:8b | original 30 | 1.000 ± 0.000 [1.00, 1.00] | 0.922 ± 0.077 [0.83, 0.99] | 0.922 ± 0.077 [0.83, 0.99] |
| qwen2.5:7b | original 30 | 0.978 ± 0.019 [0.94, 1.00] | 0.978 ± 0.019 [0.94, 1.00] | 0.978 ± 0.019 [0.94, 1.00] |
| llama3.1:8b | **added hard 14** | 0.619 ± 0.041 [0.40, 0.83] | 0.524 ± 0.082 [0.29, 0.74] | **0.405 ± 0.109** [0.19, 0.64] |
| qwen2.5:7b | **added hard 14** | 0.762 ± 0.041 [0.55, 0.95] | 0.690 ± 0.041 [0.45, 0.90] | **0.619 ± 0.041** [0.36, 0.86] |

**Earlier run.** On 2026-10-01, qwen2.5:7b ran on the original 30 at temperature 0 / seed 0 through the OpenAI-compatible endpoint. It scored 30/30 on tools and 30/30 on answers (Clopper-Pearson [0.88, 1.00]); see `results/agent_eval_qwen2_5_7b.json`. A llama3.1:8b-instruct-q8_0 run from that day did not finish and has no result.

**Failure modes seen in the hard set** (read from the raw records):

*Skipped lookups:*
- **Supplier questions.** Both models went straight to the knowledge base without looking up the part's supplier. qwen then *asserted* a supplier, "SupplierA", for both parts; that was correct for PART_112 and **wrong for PART_108**, which is SupplierD. llama listed several suppliers.
- **Stock vs forecast (H6–H8).** llama often skipped the forecast tool and approximated the forecast as avg_daily × 30.

*Wrong arithmetic or reading:*
- **H2 (direction).** On all 3 seeds qwen answered "PART_102 has 351 more units than PART_167". The truth is 351 *fewer*. The original grader passed this; the corrected grader does not.
- **H9 and H10 (order quantity).** Both models often answered with the p90 total and did not subtract stock.
- **H14 (count of CRITICAL parts).** Both models miscounted, answering 6 instead of 7.

*Broken tool calls:*
- llama sometimes printed a tool call as plain text, or "called" a tool that does not exist.

**Original 30.** The errors were llama finishing with meta-text and no answer, mainly on the list and forecast questions.

**Interpretation.** Single-lookup relaying is near ceiling for both 7–8B local models. Multi-step questions that combine tools and arithmetic drop to 40–62% joint accuracy (llama 0.405, qwen 0.619). These are small local models; the app's intended providers (Groq gpt-oss, Claude) were not tested.

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
- The TFT's MAE is about 5.5% (95% CI 4.9–6.1%) above a spike-agnostic oracle built from the generator. The oracle is a reference point, not a lower bound (its coverage is 0.829).
- **Given part identity**, static metadata covariates add nothing: the ablation difference is +0.05% MAE (95% CI −0.28% to +0.38%). The ablation **keeps `part_id`**, so it only shows that category, supplier, region, lead time and price add nothing *on top of* a per-part embedding. It does not show that the TFT could not learn supplier effects without part_id. The stronger claim, that there are no supplier or category patterns to learn, rests on the **generator analysis**: attributes are drawn independently of demand, and R² of per-part mean demand on category/supplier/region is 0.022 (adjusted −0.029), see `provenance.json`.

### 3.2 Not supported
- Any claim about **real** supply-chain data. The data is synthetic, and the generator happens to be easy for a global model that sees calendar features. The yearly sine is a function of day-of-year, which the TFT gets through `month`. The ARIMA and ETS models used here have only weekly seasonality.
- The README's former "supplier-specific patterns" / "Valve parts from SupplierA behave like X" claims. There are none to learn (generator analysis), and the ablation shows the metadata adds nothing given part_id. These README lines were corrected on this branch (change log 7).
- Claims of "significantly more accurate" *before this eval*: there was no baseline anywhere in the repo. The claim now holds only in the narrow synthetic sense above.
- Agent quality claims beyond these two small local models on 44 templated questions (§2.2). For example, nothing is known about the deployed providers or about free-form user phrasing.
- That the deployed web forecasts are forecasts. Until `forecasts.json` is regenerated with the fixed export, they are in-sample.

### 3.3 Threats to validity
- **Synthetic data with a simple generator.** Results may not transfer. The M5 / real-data check (optional plan item 4) was **not done**, because the laptop was shared and the compute was limited.
- **Baselines.** The statistical models got default statsforecast settings with weekly seasonality only. MSTL365 was the only one with explicit yearly seasonality, and it did worse than ETS here. A tuned yearly-Fourier ARIMAX or dynamic regression might close the gap. That is untested.
- **TFT budget.** Training is short (about 0.4 of an epoch per run, see §1). A larger budget could change the absolute numbers, probably for the better.
- **Mixed devices.** Runs were split across GPU and CPU, and the 2026-10-01 runs shared the GPU with other jobs. Seed std is small (0.03 to 0.05 MAE). See §3.4.
- **Stale checkpoint directories.** Runs killed on 2026-10-01 (full o1400 s0 and no_meta o1400 s1) left checkpoint files behind. The reruns loaded their *own* best checkpoint: Lightning wrote `...-v1.ckpt` on a name clash, and `best_model_path` points to the current run.
- **Bootstrap.** Parts are resampled, so dependence across origins within a part is kept. The 4 origins are not resampled, so uncertainty about time periods is understated.
- **The oracle ignores spikes and uses Gaussian quantiles.** It is a reference point, not a true lower bound. Its coverage is 0.83.
- **The agent eval is a Python port**, not the TypeScript runtime. The grading is lenient: a number anywhere in the reply counts, apart from the direction check on H1–H3 and H6–H8. The hard set was written *after* seeing the ceiling result; its templates were designed to be harder, but no question was dropped or tuned after any model was run on it. Temperature 0.7 differs from the earlier temperature-0 run. n = 14 hard questions gives wide CIs.

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
1. "On a 200-series synthetic spare-parts dataset, I evaluated a Temporal Fusion Transformer with a leakage-free rolling-origin backtest (4 origins × 30 days, 3 seeds). It reduced MAE by 14% versus AutoARIMA (95% CI 13–16%) and kept 80% prediction intervals near nominal coverage (79.6%). Its MAE was about 5.5% above a spike-agnostic oracle built from the data generator."
2. "I checked the project's claim that the model learns supplier-specific patterns. The synthetic generator assigns supplier, region and category independently of demand (R² 0.022), and an ablation showed these covariates add nothing once part identity is known (ΔMAE +0.05%, 95% CI −0.28% to +0.38%). I corrected the claim in the README."
3. "Evaluating the system end to end, I found and fixed a bug where the exported 'future' forecasts were predictions of the last 30 already-observed days (the early-stopping window). I added regression tests that fail on the original code and pass after the fix."
   - Disclosure for sentence 3: the fix is in the code paths (export script and Streamlit agent). The committed web-app data file `lib/data/forecasts.json` has **not** been regenerated, so the deployed app still shows the old values.

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

3. **`eval_sop/agent_eval.py`: harder question set, seeds and native API** (results commit, 2026-10-02).
   - *What:* added `--build-hard` (14 added questions), `--seed`, `--temperature`, `--api native` (pins `num_ctx`), `--questions`, `--forecasts` and `--tag`. Added a `required_calls` check for multi-tool questions, and new `abs_number` / `id_and_number` grading kinds. Added `eval_sop/agent_summary.py`.
   - *Harness bug fix found during the run:* qwen passed `part_id` as a JSON list, which crashed the Python port with `TypeError: unhashable type` (`get_inventory_status`). This stopped the qwen hard-set runs for seeds 1 and 2.
     - The TS route validates inputs with zod, and the AI SDK returns a tool error to the model instead of crashing.
     - The port now catches tool exceptions and returns `Error: invalid input for tool ...` as the tool result.
     - qwen hard seeds 1 and 2 were re-run after the fix. Seed 0 and all llama runs never hit this path, so they are unaffected.
   - *Preserved:* the original 30 questions, their grader and `results/agent_eval_qwen2_5_7b.json` are unchanged.
4. **`eval_sop/results/forecasts_fixed_export.json`**: the output of the fixed export, run on the original repo's local (untracked) `epoch=01-val_loss=4.5668` checkpoint. Used only as the forecast source for H6–H10.

5. **`eval_sop/agent_eval.py`: the `abs_number` grader now checks direction** (fix phase, 2026-10-04, after an independent review).
   - *What:* `grade_any` / `claimed_sign` / `_subject_flip`. The magnitude must match **and** the claimed direction must match the truth. Direction comes from a signed number, or from direction words in the sentence that holds the number, adjusted for which part is the sentence's subject. If no direction can be determined, the answer fails. `agent_summary.py` now **re-grades every record from the raw final answers**; the raw run JSONs are untouched.
   - *Why:* the old check (`abs(abs(v) - abs(answer))`, at `agent_eval.py:299-300` in `c9220ec`) ignored direction. qwen's H2 answer, which got the direction wrong, was graded correct on all 3 seeds.
   - *Evidence:* `eval_sop/test_agent_grader.py` (11 cases) gives 5 FAIL on `c9220ec` (`results/test_agent_grader_before.txt`) and 11/11 PASS after the fix (`results/test_agent_grader_after.txt`). The re-grade changed exactly 3 flags, qwen H2 on seeds 0–2. qwen hard14 moved from answer 0.762 to **0.690**, and joint from 0.690 to **0.619 ± 0.041 [0.36, 0.86]**.
   - *Not a change for llama:* the review said llama is unchanged, which is true, but a first version of the fix would have wrongly failed llama's H2. llama wrote "PART_167 has 351 more than PART_102", which is the correct direction with the subject reversed. I added a subject-order rule and a test case for it, and llama's H2 stays correct.
   - *Preserved:* the original-30 grading kinds and all raw outputs.

6. **`eval_sop/metrics.py`: proper CI for the relative differences** (fix phase, 2026-10-04).
   - *What:* added `rel_diff_pct_ci95` to `paired_diffs.json`. It is a paired, part-clustered bootstrap of 100·(mean_a / mean_b − 1) on the same 2,000 resamples (seed 12345) as the ΔMAE CI.
   - *Why:* the earlier SOP text "(paired 95% CI 12–17%)" was the ΔMAE CI divided by AutoARIMA's point MAE, which is not a CI for the ratio.
   - *Evidence:* TFT vs AutoARIMA is −14.35% [−15.90, −12.81]; TFT vs Oracle is +5.51% [+4.91, +6.13]. Re-running `metrics.py` left every existing field of `paired_diffs.json` identical, as checked programmatically, and `per_part_window_metrics.csv.gz` identical in content (the file was restored to keep the committed bytes).
   - *Also disclosed:* the AppBaseline ±1.65·sd band is a nominal 90% band scored as q10/q90 (see the §2 caveat).

7. **`README.md`: minimal accuracy and supplier-pattern corrections** (fix phase, 2026-10-04).
   - *What:* 4 lines changed (`git diff` shows 4 insertions and 4 deletions):
     - line 217, the accuracy row: now "MAE 6.86 | MAE 9.52" on synthetic data;
     - line 239, "significantly more accurate": now the measured 28% / 14% lower MAE on synthetic data, untested on real data;
     - line 249, "Learns 'Valve parts from SupplierA behave like X'": now says the attributes carry no signal in the synthetic data and that removing them did not change accuracy;
     - line 263, "much more accurate for ... supplier-specific patterns": now gives the measured numbers and says there are no supplier-specific patterns in the synthetic data.
   - *Why:* those claims were unsupported (no baseline existed) or contradicted by the generator. SOP sentence 2 says the claim was corrected, so the README had to actually change.
   - *Evidence:* `results/summary.json`, `results/paired_diffs.json` and `results/provenance.json`. No code was touched, and vitest passes 17/17.
   - *Preserved:* every other README line, including the TFT architecture description and the drift and export sections, which are still listed under "Proposed".

8. **`agent/agent.py`: the Streamlit forecast path now forecasts the future with the best checkpoint** (fix phase, 2026-10-04).
   - *What:* `get_demand_forecast` now passes `_best_checkpoint(checkpoint_files)`; previously it passed `checkpoint_files[0]` in glob order (line 260 in `0f94fd1`). `_forecast_with_tft` now builds `part_df` with `_append_future_rows(...)`; previously it used the observed frame (line 322). Both helpers are imported from `forecasting/export_forecasts.py`. 11 lines were added and 2 changed.
   - *Why:* this is the same bug as change log 1. Here glob order on this machine returned `epoch=00` first.
   - *Evidence:* `eval_sop/test_agent_forecast_path.py` uses a stub model and stubs out MLflow logging. Before the fix it reported FAIL on the decoder (1431–1460) and FAIL on the checkpoint. After the fix it reports PASS on the decoder (1461–1490) and PASS on the checkpoint. vitest passes 17/17.
   - *Preserved:* the statistical-baseline fallback, the exception handling, MLflow logging and all comments.

9. **`forecasting/export_forecasts.py`: the `_best_checkpoint` regex now handles versioned names** (fix phase, 2026-10-04).
   - *What:* changed `val_loss=([0-9.]+?)(?:\.ckpt)?$` to `val_loss=(\d+\.\d+)`.
   - *Why:* Lightning names clashing checkpoints `...-v1.ckpt`. The old pattern scored those as `inf`, so a better versioned checkpoint was never chosen.
   - *Evidence:* `eval_sop/test_best_checkpoint.py` FAILs the `-v1` case before the fix (`results/test_best_checkpoint_before.txt`) and passes 3/3 after (`results/test_best_checkpoint_after.txt`). `test_export_forecasts` still passes, and vitest passes 17/17.
   - *Preserved:* all other export behaviour.

### Deviations from the requested rules, disclosed
- **statsforecast was installed into a separate scratchpad venv** (`venv-sf`: Python 3.12.3 from anaconda, statsforecast 2.1.1, numpy 2.5.3, pandas 2.3.3, numba 0.68.0), **not** the repo's `venv/`. statsforecast pulls newer numpy and pandas. Installing it into `venv/` (numpy 1.26.4, pandas 2.1.4, torch 2.5.1+cu121, pytorch-forecasting 1.7.0, lightning 2.2.5) would have upgraded the dependencies the TFT stack is pinned to. Nothing global or system-wide was changed.
- `npm ci --ignore-scripts` was run **inside the worktree** to run the existing vitest suite (node 24.14.0). `node_modules/` is gitignored and not committed.
- The agent run on 2026-10-01 used Ollama *before* the "no Ollama for now" instruction arrived, and it is reported as is. The 2026-10-02 runs were made after the Ollama slot was granted, on the main :11434 server, one request at a time, with `num_ctx` 8192.

## 6. Proposed, not done
- **Regenerate `lib/data/forecasts.json`** with the fixed export and a properly trained checkpoint. Not done: it is product data, and the checkpoint is not in git.
- ~~`agent/agent.py:_forecast_with_tft` has the same bug~~. **Done** in change log 8.
- **`lib/db/drift.ts`**: compare logged forecasts to *realized* demand over the forecast window, which means storing the forecast date. The current check compares against the historical average.
- **`forecasting/train.py`**: early-stopping on the validation window and then reporting MAE on that same window is optimistic. Add a held-out test window or a rolling-origin evaluation, and set a seed (`pl.seed_everything`).
- Add Python tests to CI (`eval_sop/test_export_forecasts.py` is a start).

### README corrections
- **Applied on this branch** (change log 7): lines 217, 239, 249 and 263. These are the accuracy row, "significantly more accurate", "Valve parts from SupplierA behave like X" and "much more accurate for ... supplier-specific patterns". They now cite the measured synthetic-data numbers and say that the attributes carry no signal in the synthetic data.
- **Still proposed, not done:**
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
$PY -m eval_sop.agent_eval --build-hard                  # added hard set (already committed)
for s in 0 1 2; do   # needs Ollama on :11434
  $PY -m eval_sop.agent_eval --model llama3.1:8b --api native --num-ctx 8192 --temperature 0.7 --seed $s --tag orig30_t07_s$s
  $PY -m eval_sop.agent_eval --model llama3.1:8b --api native --num-ctx 8192 --temperature 0.7 --seed $s \
      --questions eval_sop/agent_questions_hard.json --forecasts eval_sop/results/forecasts_fixed_export.json --tag hard14_t07_s$s
done   # same for --model qwen2.5:7b
$PY -m eval_sop.agent_summary
```
Seeds: data 42, TFT 0/1/2 (`pl.seed_everything`), bootstrap 12345, question sampling 2024.
