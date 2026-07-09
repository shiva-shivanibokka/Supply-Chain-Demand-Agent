# Supply Chain Demand Agent

An end-to-end agentic AI system for supply chain demand forecasting. Built for capital equipment and semiconductor manufacturing companies managing large inventories of spare parts.

The system forecasts 30-day part demand, answers natural language questions using a retrieval-augmented (RAG) tool over internal supply chain documents, and acts autonomously through an AI agent that decides which tools to call to answer a question.

The project has two layers:

1. **A live Next.js web app** (this repo's root) — a Vercel AI SDK–powered chat assistant with BYOK (bring your own key) across four LLM providers, an inventory dashboard, a demand forecast tab, and an MLOps monitor tab, backed by Neon Postgres.
2. **A local Python ML stack** (`forecasting/`, `rag/`, `mlops/`, `agent/`, `data/`) — where the actual Temporal Fusion Transformer model is trained, evaluated, and exported. Its output (`lib/data/forecasts.json`) is what the web app serves as real "TFT model" forecasts.

> **Bring Your Own Key:** The web app never uses a shared API key. You select a provider (Anthropic, OpenAI, Groq, or Google) and paste your own key in the top bar. It's held in browser state only for that session, sent per-request to the chat route, and never stored or logged server-side. See [BYOK](#llm-providers--bring-your-own-key) below for how to get a key.

---

## What problem does this solve?

In capital equipment manufacturing, thousands of spare parts sit in warehouses across multiple regions. Every day, supply chain teams need to know:

- Which parts are about to run out?
- How much should we order next month?
- What does our reorder policy say about this situation?

Answering these manually means opening spreadsheets, reading policy documents, and doing calculations by hand. This project automates all of that with an AI agent.

---

## Architecture

```
User question
      ↓
Chat route (app/api/chat) — Vercel AI SDK streamText + tool loop
      ↓                  ↓                    ↓
Inventory tool     Forecast tool        Knowledge-base tool
 reads parts.json   precomputed TFT      keyword search over
 (built from CSV)   JSON, else           docs.json (built from
                     statistical         rag/embeddings.npz)
                     baseline
      ↓                  ↓                    ↓
      Model (Anthropic / OpenAI / Groq / Google, user's key) writes final answer
                          ↓
      Next.js UI (Assistant / Inventory / Forecast / MLOps tabs)
                          ↓
      Neon Postgres (prediction log, optional — no-ops if DATABASE_URL unset)
```

The Python stack (`forecasting/`, `rag/`, `agent/agent.py`, `mlops/monitor.py`) is a separate, local-only pipeline: it trains the real TFT model, tracks experiments in MLflow, and builds the RAG knowledge base — then exports its outputs (`lib/data/forecasts.json`, `lib/data/docs.json`) for the web app to consume as static data. The web app itself has no Python or PyTorch dependency at runtime.

---

## Tech Stack

### Web app (root, live/Vercel)

| Layer | Tool |
|---|---|
| Framework | Next.js 16 (App Router) + TypeScript |
| Agent / chat | Vercel AI SDK (`ai`, `@ai-sdk/react`) — streaming + tool calling |
| LLM providers | Anthropic, OpenAI, Groq, Google (user's own key, per-request) |
| UI | shadcn/ui + Tailwind CSS v4 |
| Charts | Recharts |
| Database | Neon Postgres (serverless HTTP driver) + Drizzle ORM |
| Forecast data | Precomputed TFT output (`lib/data/forecasts.json`), statistical fallback |
| Knowledge search | Keyword/TF scoring over `lib/data/docs.json` (no vector DB at runtime) |
| Tests | Vitest |

### Python ML stack (local training layer)

| Layer | Tool |
|---|---|
| Forecasting model | Temporal Fusion Transformer (pytorch-forecasting) |
| Deep learning | PyTorch + Lightning |
| RAG embeddings | sentence-transformers (all-MiniLM-L6-v2) + ChromaDB |
| MLOps | MLflow (experiment tracking, model registry, prediction logging, drift detection) |
| Local UI (optional) | Streamlit + Plotly (`app.py`) |
| Data | Pandas, synthetic supply chain dataset |

---

## Project Structure

```
Supply-Chain-Demand-Agent/
├── app/                             ← Next.js App Router
│   ├── page.tsx                     ← main page: provider bar + 4 tabs
│   └── api/
│       ├── chat/route.ts            ← AI SDK streamText + tool loop, 4-provider BYOK
│       ├── forecast/route.ts        ← forecast lookup endpoint
│       └── mlops/route.ts           ← prediction log + drift endpoint
├── components/                      ← shadcn/ui + Recharts components (tabs, provider bar)
├── lib/
│   ├── providers.ts                 ← provider/model registry (BYOK)
│   ├── tools/                       ← inventory / forecast / knowledge tool implementations
│   ├── db/                          ← Drizzle schema, Neon client, prediction log, drift
│   └── data/                        ← generated JSON (parts, forecasts, docs) — checked in
├── scripts/
│   ├── build-data.mjs               ← CSV → lib/data/parts.json (runs pre-build/pre-dev)
│   └── build-docs.mjs               ← points at the one-off Python extractor for docs.json
├── docs/adr/                        ← architecture decision records
├── .github/workflows/ci.yml         ← lint, typecheck, test, build:data on push/PR
│
├── data/
│   ├── generate_data.py             ← synthetic dataset generator
│   └── supply_chain_data.csv        ← 73,050 rows, 50 parts × 4 years
├── forecasting/
│   ├── model.py                     ← TFT configuration + dataset builder
│   ├── train.py                     ← training loop + MLflow tracking
│   └── export_forecasts.py          ← trained model → lib/data/forecasts.json
├── rag/
│   ├── ingest.py                    ← embeds documents into ChromaDB (local)
│   ├── retriever.py                 ← semantic search via ChromaDB (local)
│   └── embeddings.npz               ← source docs for lib/data/docs.json
├── agent/
│   └── agent.py                     ← original multi-provider Python agent (local CLI/Streamlit)
├── mlops/
│   └── monitor.py                   ← prediction logging, drift detection, model registry
├── notebooks/
│   └── walkthrough.ipynb            ← step-by-step tutorial notebook
├── app.py                           ← Streamlit UI — optional local alternative to the Next.js app
└── requirements-local.txt           ← full local Python stack (torch, chromadb, mlflow, streamlit)
```

---

## Setup

### Option A — Run the web app locally

**1. Clone and install**
```bash
git clone https://github.com/shivani-shivanibokka/Supply-Chain-Demand-Agent.git
cd Supply-Chain-Demand-Agent
npm ci
```

**2. Run it**
```bash
npm run dev
# open http://localhost:3000
```

That's it — `data/supply_chain_data.csv` and the RAG document set are already checked into the repo, so `npm run dev` builds `lib/data/parts.json` and runs immediately. No API key is stored on the server: paste a key for any provider in the top bar to chat.

`DATABASE_URL` is **optional** for local dev. Without it, `lib/db/client.ts` exports `db = null` and `logPrediction()` no-ops (logs a warning, doesn't throw) — every other tab works normally, the MLOps tab will just show no prediction history.

### Option B — Deploy on Vercel

1. Import this repo on [vercel.com/new](https://vercel.com/new).
2. Add the **Neon** integration (Storage → Neon → Create) — this provisions a database and sets `DATABASE_URL` automatically.
3. Run the schema push once, against that database:
   ```bash
   npm run db:push
   ```
4. Deploy. No other environment variables are required — BYOK keys are supplied per-session by whoever uses the app.

### Option C — Train the TFT model and update the forecasts (local Python stack)

The web app ships with `lib/data/forecasts.json` already populated from a trained model. To retrain it yourself:

```bash
pip install -r requirements-local.txt
python -m data.generate_data          # regenerate the synthetic dataset (optional, already checked in)
python -m forecasting.train           # trains the TFT, ~5-20 minutes, logs to MLflow
python -m forecasting.export_forecasts  # writes lib/data/forecasts.json for the web app
```

If `forecasting.export_forecasts` hasn't been run (or a part has no entry), `lib/tools/forecast.ts` falls back to a statistical baseline computed from the last 60 days of demand — so the app always produces a forecast even without a trained model.

**Optional — run the original Streamlit app locally** (same agent logic, Python-native UI, includes an MLOps Monitor tab wired directly to MLflow):
```bash
streamlit run app.py
# open http://localhost:8501
```

---

## What the Web App Shows

Four tabs, all derived live from the dataset — nothing is hardcoded.

### Tab 1 — AI Assistant

A chat interface (`components/assistant.tsx`) built on `useChat` from `@ai-sdk/react`.

- **Provider bar** — pick a provider, a model, and paste your key. Held in component state only.
- **Live tool steps** — as the agent works, each tool call renders inline ("🔧 Checking inventory…" → "✅ Checking inventory") before the final answer streams in.
- **Clear conversation** — resets chat state.

The agent can answer questions like:
- Which parts are closest to a stockout?
- What does the reorder policy say for this category?
- What's the 30-day forecast for PART_014, and how much should I order?

### Tab 2 — Inventory Dashboard

KPI row (total parts, CRITICAL/WARNING/OK counts) computed by `lib/inventory-summary.ts`, a category selector, two Recharts bar charts (days-of-supply by risk, inventory vs. avg daily demand), and the full parts table with risk badges.

### Tab 3 — Demand Forecast

Pick a part, see its last-90-days demand history and a 30-day forecast chart (p10/p50/p90 band) fetched from `/api/forecast`. A badge shows whether the forecast came from the **TFT model** (precomputed JSON) or the **statistical baseline** (computed on the fly).

### Tab 4 — MLOps Monitor

Pulls from `/api/mlops`: a drift-status card (MAE, baseline MAE, calibration %, OK/WARNING/NO-DATA), a most-queried-parts chart, and the raw prediction log — all backed by the Neon `predictions` table. Every forecast the Assistant or Forecast tab requests is logged via `logPrediction()`.

---

## Forecasting Models — TFT vs Statistical Baseline

This project uses **two different forecasting paths**. Understanding the difference matters if you want to retrain the model or extend it with new data.

| | TFT (Temporal Fusion Transformer) | Statistical Baseline |
|---|---|---|
| **Where it runs** | Trained locally (`forecasting/train.py`), exported to static JSON, served by the web app | Computed on the fly, in `lib/tools/forecast.ts`, when no exported entry exists for a part |
| **Requires** | PyTorch, pytorch-forecasting (local training only — never installed at runtime on Vercel) | Nothing extra — pure TypeScript |
| **Accuracy** | High — learns trends, seasonality, part-specific patterns | Moderate — mean + trend extrapolation |
| **Training needed** | Yes — run `forecasting/train.py` + `forecasting/export_forecasts.py` once | No — always available |
| **Prediction intervals** | Learned quantiles (p10/p50/p90) from data | Computed from historical standard deviation |

**Why precompute instead of running the model on Vercel?** Vercel functions are ephemeral and don't ship PyTorch — training and inference happen locally, and only the resulting numbers (`lib/data/forecasts.json`) travel with the deploy. This keeps the production app dependency-free while still serving real model output, not just the baseline.

### How the statistical baseline works

Implemented in `lib/tools/forecast.ts` → `getForecast()`:

1. Takes the last 60 days of demand for the selected part
2. Computes the mean (`avg`) and standard deviation (`std`)
3. Adds a small upward trend (~2.5%) to the median forecast
4. Computes prediction intervals using `±1.65σ` (covers ~90% of a normal distribution)

```
p50Daily = avg × 1.025
p50 = p50Daily × 30
p10 = (p50Daily − 1.65 × std) × 30
p90 = (p50Daily + 1.65 × std) × 30
```

This is a well-known statistical method — an auto-regressive mean model with Gaussian uncertainty. It works well for parts with stable, low-volatility demand. For parts with strong seasonality or sudden spikes, the TFT model is significantly more accurate.

### How the TFT model works

Implemented in `forecasting/model.py`, trained via `forecasting/train.py`. It improves on the statistical baseline in three ways:

**1. It separates inputs by type**

| Input type | Examples | How TFT uses it |
|---|---|---|
| Static (never changes per part) | category, supplier, region | Learns "Valve parts from SupplierA behave like X" |
| Past (historical observations) | demand, inventory | Learns historical patterns |
| Future known | day of week, month, quarter | Uses the calendar to anticipate seasonality in advance |

The statistical baseline ignores all of this — it only looks at recent demand numbers.

**2. Variable Selection Network**

Before forecasting, TFT automatically learns which input features actually matter. If `day_of_week` turns out to be irrelevant for a particular part category, TFT learns to ignore it. This is built-in feature selection.

**3. Self-attention over the full history**

Like GPT/BERT, TFT uses attention to look back across the entire 90-day window and identify which historical time steps are most informative for the current prediction. A statistical model can only use recent summary statistics.

The result: TFT is much more accurate for parts with strong seasonality, supplier-specific patterns, or irregular spikes — which is exactly what real supply chain data looks like.

### How to retrain the TFT model and update the web app

```bash
python -m data.generate_data            # ensure data/supply_chain_data.csv exists
python -m forecasting.train              # trains, saves checkpoint to forecasting/saved_model/, logs to MLflow
mlflow ui                                # inspect loss curves, hyperparameters at http://localhost:5000
python -m forecasting.export_forecasts   # writes lib/data/forecasts.json from the trained checkpoint
```

Commit the updated `lib/data/forecasts.json` and redeploy — the web app will pick it up on the next `getForecast()` call, no code changes needed.

### Retraining with your own data

To use this project with real demand data instead of the synthetic dataset:

1. **Format your data** to match `data/supply_chain_data.csv`:

```
date,part_id,category,supplier,region,demand,inventory,lead_time_days,price_usd
2023-01-01,PART_001,Controller,SupplierA,North America,79,3420,23,822.29
...
```

Required columns: `date` (daily), `part_id`, `category`, `supplier`, `region`, `demand` (units/day), `inventory`, `lead_time_days`, `price_usd`.

2. **Place your file** at `data/supply_chain_data.csv` (overwrite the synthetic one).
3. **Rebuild the web app's data files**: `npm run build:data` regenerates `lib/data/parts.json` from the CSV.
4. **Rebuild the RAG knowledge base** if your policy documents changed: `python -m rag.ingest`, then re-run the extractor noted in `scripts/build-docs.mjs` to refresh `lib/data/docs.json`.
5. **Retrain the model**: `python -m forecasting.train` then `python -m forecasting.export_forecasts`.
6. **Redeploy** — the agent, dashboard, and forecast tab all pick up the new data automatically.

---

## LLM Providers — Bring Your Own Key

The web app supports four providers so you can use whichever key you have. Model choices come from `lib/providers.ts`:

| Provider | Models | Key format | Cost |
|---|---|---|---|
| **Anthropic** | `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5` | `sk-ant-...` | Paid |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | `sk-...` | Paid |
| **Groq** (default) | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | `gsk_...` | Free tier |
| **Google** | `gemini-2.0-flash`, `gemini-1.5-pro` | `AIza...` | Paid |

The provider bar at the top of the app lets you pick a provider, a model, and paste your key. The key lives in React state for that browser tab only — it's sent to `/api/chat` on each request and is never written to a database, log, or file.

**Why Groq is the default:** it offers a generous free API tier with no credit card required, running open-source Llama models at very fast inference speeds — the easiest way to try the app with zero cost.

**How to get each key:**
- **Anthropic:** [console.anthropic.com](https://console.anthropic.com/) → API Keys
- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Groq (free):** [console.groq.com/keys](https://console.groq.com/keys)
- **Google:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

Provider/model IDs move fast — check `lib/providers.ts` for the current list, and each vendor's docs if a model ID stops working.

---

## MLOps — Beyond Basic Training

This project implements the MLOps lifecycle end to end, not just training. Two implementations exist side by side: the original Python/MLflow stack (local training) and a TypeScript/Neon port (production, in the web app).

### Model Registry (local, Python/MLflow)

After training, the TFT model is registered in MLflow's Model Registry under `supply-chain-tft`. Each training run creates a new version, starting in **Staging**; promote it to **Production** from the app or the MLflow UI.

```
Training run → v1 (Staging) → review → v1 (Production)
New training  → v2 (Staging) → review → v2 (Production), v1 (Archived)
```

### Prediction Logging (production, web app)

Every forecast the Assistant or Forecast tab requests calls `logPrediction()` (`lib/db/log.ts`), writing part ID, p10/p50/p90 totals, daily median, and forecast source to the Neon `predictions` table. This is a real audit trail: "what did the model actually predict on a given day for PART_007?" — visible in the MLOps tab's prediction log.

### Drift Detection (production, web app)

`computeDrift()` (`lib/db/drift.ts`, a TypeScript port of the original `mlops/monitor.py` logic) compares logged predictions against actual average demand per part and computes:

- **MAE** — mean absolute error of `p50Daily` vs. actual average daily demand
- **Calibration** — the % of actuals falling inside the predicted p10–p90 band (target ~80%+)
- **Drift flag** — fires when MAE is more than 1.5× the baseline (predict-the-global-mean) MAE

Requires at least 3 logged predictions; otherwise the MLOps tab reports `NO-DATA`.

---

## File-by-File Explanation (Python ML stack)

### `data/generate_data.py`

Creates a synthetic supply chain dataset of 50 spare parts with 4 years of daily demand history (73,050 rows). Real supply chain data from companies is confidential, so generating synthetic data with the same statistical patterns is standard practice. Each part's demand has three layers: a slow upward trend (~20% over 4 years), yearly seasonality (peaks around October, factory maintenance season), and random spikes (~5/year, simulating emergency orders). Static attributes (category, supplier, region, lead time, price) never change — TFT has a dedicated channel to learn from them.

### `forecasting/model.py`

Defines how to prepare the data and configure the TFT model. `load_and_prepare()` casts columns to float32 and adds the integer time index and calendar features TFT needs. `build_dataset()` wraps everything in `TimeSeriesDataSet` — pytorch-forecasting's format that handles per-part normalization, sliding-window creation, and input-type separation. `build_model()` creates TFT with hidden size 64, 4 attention heads, 10% dropout, and QuantileLoss for the three quantiles.

### `forecasting/train.py`

Trains the model and logs everything to MLflow. Lightning handles the training loop — no `for epoch in range(...)` anywhere; you configure a `Trainer`, call `trainer.fit()`, and Lightning handles the forward pass, loss, backprop, optimizer steps, and validation. Three callbacks run: `EarlyStopping` (stops if validation loss plateaus for 5 epochs), `ModelCheckpoint` (saves the best weights), `LearningRateMonitor` (logs the LR schedule). MLflow records every hyperparameter, every epoch's metrics, and the final model artifact.

### `forecasting/export_forecasts.py`

Loads the best checkpoint in `forecasting/saved_model/`, runs the TFT for every part in the dataset, and writes `lib/data/forecasts.json` — the file the web app reads as real "TFT model" forecasts. This is the bridge between the Python training stack and the Next.js app: no live model server, just a static JSON handoff.

### `rag/ingest.py`

Builds the knowledge base — converts supply chain documents into embedding vectors and stores them in ChromaDB. An LLM knows nothing about a company's own reorder policies, supplier reliability history, or safety-stock formula; RAG (Retrieval-Augmented Generation) gives it that knowledge by searching a document store first. The embedding model (`all-MiniLM-L6-v2`) produces 384-dimensional vectors and runs entirely locally — no API key needed. ChromaDB indexes those vectors with HNSW (Hierarchical Navigable Small World) for fast nearest-neighbor search — a different job than a document database like MongoDB, which excels at exact-value filtering but has no built-in concept of semantic similarity.

### `rag/retriever.py`

The local search half of RAG: embeds the question with the same model used at ingest time, asks ChromaDB for the nearest stored vectors by cosine similarity, and returns the top 3 matches (filtering out anything below 0.3 similarity). The web app's equivalent, `lib/tools/knowledge.ts`, does keyword/TF-overlap scoring over `lib/data/docs.json` instead — no embeddings at runtime, which keeps the Vercel deploy dependency-free. For this project's small (10-document) knowledge base the quality difference is minor in practice.

### `agent/agent.py`

The original multi-provider Python agent — a full ReAct loop (Reason → Act → Observe → Reason → ... → Answer) over three tools (inventory, forecast, knowledge search), runnable via the Streamlit app (`app.py`) or the notebook. The Next.js `app/api/chat/route.ts` is the production reimplementation of this same loop on top of the Vercel AI SDK, with the same three tools and BYOK across four providers.

### `mlops/monitor.py`

The original MLflow-backed MLOps implementation: prediction logging (`log_prediction()`), drift detection (`compute_drift_metrics()` — MAE, calibration, degradation vs. baseline), and model registry helpers (`get_registered_model_info()`, `promote_to_production()`). `lib/db/drift.ts` is the TypeScript port of the drift logic used by the production MLOps tab.

### `notebooks/walkthrough.ipynb`

A guided tutorial notebook that runs through every Python component step by step: dataset exploration, TFT dataset construction, model architecture inspection, RAG ingestion and retrieval testing, individual tool testing, the full agent loop, and MLflow run inspection. Intended for anyone reading the code for the first time.

### `app.py`

The original Streamlit web application — an optional local alternative to the Next.js app, using the same Python agent and the full local stack (`requirements-local.txt`). Adds a Model Registry / promotion UI directly wired to a running MLflow server, which the production web app doesn't have (Neon logs predictions and computes drift, but doesn't manage MLflow model-version promotion).

---

## Skills Demonstrated

| Skill Area | Implementation |
|---|---|
| Full-stack TypeScript / Next.js | App Router, Vercel AI SDK, shadcn/ui, Recharts, Drizzle |
| Agentic AI, tool-calling | `app/api/chat/route.ts` (production) and `agent/agent.py` (original ReAct loop) |
| Multi-provider LLM integration, BYOK | Anthropic, OpenAI, Groq, Google via one AI SDK interface |
| RAG, embedding-based search | ChromaDB + sentence-transformers (training/local); TF-keyword search (production) |
| Demand forecasting, deep learning | Temporal Fusion Transformer (pytorch-forecasting) on a 50-part time series |
| Serverless data persistence | Neon Postgres (HTTP driver) + Drizzle, `DATABASE_URL`-optional design |
| MLOps — experiment tracking & versioning | MLflow logging, Model Registry with Staging → Production promotion |
| MLOps — production monitoring | Prediction logging + drift detection, ported to run on the deployed app |
| CI / testing | GitHub Actions (lint, typecheck, test, data build), Vitest unit tests |
| Data engineering | Pandas synthetic time-series generation, CSV → JSON build pipeline |
