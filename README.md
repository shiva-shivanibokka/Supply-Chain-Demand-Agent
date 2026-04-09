---
title: Supply Chain Demand Agent
emoji: 📦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
short_description: Agentic AI for supply chain demand forecasting with RAG
---

# Supply Chain Demand Agent

An end-to-end agentic AI system for supply chain demand forecasting. Built for capital equipment and semiconductor manufacturing companies managing large inventories of spare parts.

The system forecasts 30-day part demand using a Temporal Fusion Transformer, answers natural language questions using a RAG pipeline over internal supply chain documents, and acts autonomously through an AI agent that decides which tools to call to answer a question.

This project was originally built using **Anthropic Claude**. It has since been extended to support multiple LLM providers — users can plug in whichever API key they have, including a **free Groq key** (no credit card required).

> **Bring Your Own Key:** This app never uses a shared API key. You select your provider (Anthropic, OpenAI, or Groq) and paste your own key at the top of the page. It is held in session memory only and never stored or shared. See the [LLM Provider](#llm-provider--bring-your-own-key) section for how to get a key.

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
  Agent (Claude) — decides which tools to call
      ↓                  ↓                    ↓
Inventory tool     Forecast tool        RAG search tool
 reads CSV          TFT or statistical   ChromaDB (local)
                    baseline (cloud)     or keyword search (cloud)
      ↓                  ↓                    ↓
          Claude reads all results, writes final answer
                          ↓
          Gradio UI (Hugging Face Spaces) or Streamlit (local)
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Forecasting model | Temporal Fusion Transformer (pytorch-forecasting) |
| Deep learning | PyTorch + Lightning |
| Agent / LLM | Anthropic Claude, OpenAI GPT, or Groq (user's choice) |
| RAG embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector database | ChromaDB |
| MLOps | MLflow (experiment tracking, model registry, prediction logging, drift detection) |
| UI (cloud) | Gradio + Plotly — deployed on Hugging Face Spaces |
| UI (local) | Streamlit + Plotly — full stack with MLOps Monitor tab |
| RAG (cloud) | Keyword search over pre-built embeddings (numpy only) |
| RAG (local) | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Data | Pandas, synthetic supply chain dataset |

---

## Live Demo

Deployed on Hugging Face Spaces (Gradio):
[https://huggingface.co/spaces/shiva-1993/Supply-Chain-Demand-Agent](https://huggingface.co/spaces/shiva-1993/Supply-Chain-Demand-Agent)

The cloud version uses a lightweight statistical forecaster (no PyTorch required) and a keyword-based RAG retriever (no ChromaDB required). The full TFT model and MLOps monitor are available in local deployment only.

---

## Project Structure

```
Supply-Chain-Demand-Agent/
├── gradio_app.py                   ← Gradio UI — Hugging Face Spaces (cloud entry point)
├── app.py                          ← Streamlit UI — local only (full stack with MLOps)
├── requirements.txt                ← Cloud-minimal deps (gradio, pandas, numpy, anthropic, openai, plotly)
├── requirements-local.txt          ← Full local stack (torch, chromadb, mlflow, streamlit)
├── .env                            ← your API key goes here (never committed)
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example        ← template for Streamlit Cloud deployment
├── data/
│   ├── generate_data.py            ← synthetic dataset generator
│   └── supply_chain_data.csv       ← 73,050 rows, 50 parts × 4 years
├── forecasting/
│   ├── model.py                    ← TFT configuration + dataset builder
│   └── train.py                    ← training loop + MLflow tracking
├── rag/
│   ├── ingest.py                   ← embeds 10 documents into ChromaDB (local)
│   ├── retriever.py                ← semantic search via ChromaDB (local)
│   ├── retriever_cloud.py          ← keyword search, no chromadb (cloud)
│   └── embeddings.npz              ← pre-built embeddings committed to repo (cloud)
├── agent/
│   └── agent.py                    ← multi-provider agent (Anthropic / OpenAI / Groq)
├── mlops/
│   └── monitor.py                  ← prediction logging, drift detection, model registry
└── notebooks/
    └── walkthrough.ipynb           ← step-by-step tutorial notebook
```

---

## Setup

### Option A — Hugging Face Spaces (no local setup needed)

The app is deployed at: [https://huggingface.co/spaces/shiva-1993/Supply-Chain-Demand-Agent](https://huggingface.co/spaces/shiva-1993/Supply-Chain-Demand-Agent)

Just open the link, paste your API key (Anthropic, OpenAI, or Groq), and use it.

---

### Option B — Local Setup (full stack with TFT model + MLOps)

**1. Clone the repo**
```bash
git clone https://github.com/shivani-shivanibokka/Supply-Chain-Demand-Agent.git
cd Supply-Chain-Demand-Agent
```

**2. Create and activate the virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements-local.txt
```

**4. Add your API key**

Open `.env` and add whichever key you have:
```
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic Claude (original)
# or
OPENAI_API_KEY=sk-...          # OpenAI GPT models
# or
GROQ_API_KEY=gsk_...           # Groq — free tier, no credit card needed
```
You only need one. If you skip this step you can also paste the key directly in the app sidebar.

---

## Running the Project

**Step 1 — Generate the dataset**
```bash
python -m data.generate_data
```

**Step 2 — Build the RAG knowledge base**
```bash
python -m rag.ingest
```

**Step 3 — Train the TFT model** (5–20 minutes, uses GPU if available)
```bash
python -m forecasting.train
```

**Step 4 — View training results**
```bash
mlflow ui
# open http://localhost:5000
```

**Step 5 — Run the app**

Full local stack (Streamlit, with TFT model + MLOps Monitor):
```bash
streamlit run app.py
# open http://localhost:8501
```

Or run the Gradio version locally (same cloud version, no torch/mlflow needed):
```bash
python gradio_app.py
# open http://localhost:7860
```

---

## What the App Shows

The app has four tabs. None of the content is hardcoded — everything is derived live from the dataset.

The first three tabs (AI Assistant, Inventory Dashboard, Demand Forecast) are available in both the cloud Gradio app and the local Streamlit app. The MLOps Monitor tab is local-only — it requires a running MLflow server.

---

### Tab 1 — AI Assistant

A chat interface powered by the Claude agent.

- **API key input** — select your provider and paste your key at the top of the page. The key is held in session memory only and never stored.
- **Suggested questions** — three quick-start buttons that pre-fill common questions.
- **Agent reasoning steps** — as the agent works, you see each step live: which tool it called, and a preview of what it found, before the final answer appears.
- **Chat history** — conversation persists across questions within the same session. A "Clear conversation" button resets it.

The agent can answer questions like:
- Which parts are closest to a stockout?
- What does the reorder policy say for this category?
- How reliable is a given supplier?
- What is the safety stock formula and how do I apply it?

---

### Tab 2 — Inventory Dashboard

A visual snapshot of all parts and their current risk levels, computed from the latest date in the dataset.

**KPI row (top):**
- Total parts tracked
- Number of CRITICAL parts (inventory will run out before a reorder can arrive)
- Number of WARNING parts (inventory is below 2× the lead time demand)
- Number of OK parts

**Category drill-down:**
A horizontal radio selector lets you pick a part category (Controller, Filter, Pump, Sensor, Valve). Two charts update immediately:
- **Days of Supply** — bar chart for every part in that category, color-coded by risk, with a dashed reference line at the average lead time. Bars below that line are in stockout danger.
- **Inventory vs Daily Demand** — grouped bars showing current stock vs average daily usage per part. This shows *why* a part is at its risk level.

A summary caption shows total parts, risk counts, average lead time, and average days of supply for the selected category.

**Full inventory table:**
Every part with columns: Part ID, Category, Supplier, Region, Inventory, Average Daily Demand, Days of Supply, Lead Time, Unit Price, Risk. Risk column uses emoji indicators (🔴 🟡 🟢) that are readable on any background.

**What "Region" means:** Each part is tagged with the geographic region where it is primarily deployed — North America, Europe, Asia Pacific, or Latin America. The TFT model uses region as a static covariate, meaning it can learn that demand patterns differ by region and account for that in its forecasts.

---

### Tab 3 — Demand Forecast

Pick any part from the dropdown to see its demand history and 30-day forecast.

**Part info panel (left):**
- Category, Supplier, Region, Lead Time, Unit Price
- Data range (first date to last date in the dataset)
- Total number of daily records

**Demand history chart (right):**
Shows the last 90 days of actual recorded demand for the selected part. Helps you see the part's volatility and any recent spikes before looking at the forecast.

**Forecast metric cards:**
- Daily demand (median) — the average expected daily usage
- 30-day total (p50) — best single estimate of total demand over the next 30 days
- Lower bound (p10) — demand will probably stay above this
- Order qty for 90% SL — the p90 upper bound; order this much to avoid stockout 90% of the time

**Forecast chart:**
Plots the 30-day forecast window with three lines — p10 (lower), p50 (median), p90 (upper) — and a shaded band between p10 and p90 showing the uncertainty range. Wider bands mean more volatile demand.

**Info bar at the bottom:**
Shows the selected part's lead time and the exact p90 order quantity with a plain-English recommendation.

---

### Tab 4 — MLOps Monitor

> **Local deployment only.** This tab requires a running MLflow server and is not available in the Hugging Face Spaces deployment. Run `streamlit run app.py` locally with the full stack (`requirements-local.txt`) to access it.

Tracks model health in production across three sections.

**Model Registry:**
Shows all registered versions of the TFT model with their stage (Staging / Production / Archived), creation time, validation MAE, and training metadata. A dropdown lets you promote a Staging version to Production directly from the UI — this is how you control which model version serves live forecasts.

**Prediction Log:**
Every forecast generated by the agent or the Demand Forecast tab is logged here automatically. Columns show: timestamp, part queried, forecast source (TFT model vs statistical baseline), p50 daily demand, and p10/p50/p90 totals. A bar chart shows which parts have been queried most frequently.

**Drift Detection:**
Click "Run Drift Check" to compare logged predictions against actual demand from the dataset. Three metrics are computed and displayed:
- **MAE (30-day)** vs the training baseline MAE — with a percentage degradation delta
- **Calibration score** — what percentage of actuals fell inside the p10–p90 band (target ~80%)
- **Drift alert** — fires if current MAE is more than 20% worse than training baseline

All drift metrics are logged back to MLflow so you can trend them over time with `mlflow ui`.

---

## File-by-File Explanation

---

### `data/generate_data.py`

Creates a synthetic supply chain dataset of 50 spare parts with 4 years of daily demand history (73,050 rows).

Real supply chain data from companies is confidential, so generating synthetic data with the same statistical patterns is standard practice. Each part's demand has three layers:

- **Slow upward trend** — demand grows ~20% over 4 years
- **Yearly seasonality** — peaks around October, when factories do end-of-year maintenance
- **Random spikes** — ~5 times per year, simulating emergency orders from equipment failures

Each part also has static attributes (category, supplier, region, lead time, price) that never change. These are critical for TFT — it has a dedicated channel to learn from them.

<br>

---

### `forecasting/model.py`

Defines how to prepare the data and configure the TFT model.

**Why TFT and not LSTM?**

An LSTM reads a sequence one step at a time and passes a hidden state forward. It works, but its memory fades over long sequences and it treats all inputs the same way.

TFT improves on this in four specific ways:

**1. Separates inputs into types**

| Input type | Example | How TFT uses it |
|---|---|---|
| Static (never changes) | part category, supplier | Learns "Valves from SupplierC behave like X" |
| Past unknown | demand, inventory | Learns from historical patterns |
| Future known | month, quarter | Uses calendar to predict seasonality in advance |

An LSTM mixes all of these together. TFT processes each type through a different path.

**2. Variable Selection Network**

Before forecasting, TFT learns which input features actually matter. If `day_of_week` turns out to be useless for a particular part type, TFT learns to ignore it. This is automatic feature selection built into the architecture.

**3. Self-attention over the full history**

Like in GPT/BERT, TFT uses attention to look back across the entire 90-day window and decide which past time steps are most relevant right now. An LSTM's memory fades — TFT can jump back and find a relevant spike from 2 months ago.

**4. Prediction intervals, not just one number**

TFT outputs three quantiles:
- **p10** — lower bound (demand will probably be above this)
- **p50** — median, the best single guess
- **p90** — upper bound (use this for safety stock calculations)

This is far more useful for inventory decisions than a single point estimate.

`load_and_prepare()` casts columns to float32 and adds the integer time index and calendar features TFT needs.

`build_dataset()` wraps everything in `TimeSeriesDataSet` — pytorch-forecasting's format that handles normalization per part, sliding window creation, and input type separation automatically.

`build_model()` creates TFT with hidden size 64, 4 attention heads, 10% dropout, and QuantileLoss for the three quantiles.

<br>

---

### `forecasting/train.py`

Trains the model and logs everything to MLflow.

**Lightning** handles the entire training loop. There is no `for epoch in range(...)` written anywhere. You configure a `Trainer`, call `trainer.fit()`, and Lightning handles the forward pass, loss computation, backpropagation, optimizer steps, GPU placement, and validation automatically.

**Three callbacks** run during training:

- `EarlyStopping` — stops training if validation loss doesn't improve for 5 consecutive epochs. Prevents wasting hours of compute on a model that has already peaked.
- `ModelCheckpoint` — saves the model weights every time validation loss hits a new low. Even if the model gets worse in later epochs, you always keep the best version.
- `LearningRateMonitor` — logs the learning rate at each epoch so you can verify in MLflow that the scheduler is reducing it correctly.

**MLflow** wraps the entire training run. Every hyperparameter, every metric at every epoch, and the final model file are saved automatically to the `mlruns/` folder. Run `mlflow ui` after training to see a full dashboard. If you train multiple times with different settings, MLflow lets you compare all runs side by side.

<br>

---

### `rag/ingest.py`

Builds the knowledge base — converts 10 supply chain documents into embedding vectors and stores them in ChromaDB.

**Why RAG?**

Claude is trained on general internet data. It knows nothing about your company's reorder policies, your supplier reliability history, or your safety stock formula. RAG (Retrieval-Augmented Generation) gives Claude that knowledge by letting it search a database of your own documents before answering.

**How embeddings work**

An embedding model converts text into a list of numbers that captures meaning. Sentences with similar meanings get similar numbers, sentences with different meanings get different numbers.

```
"inventory is running low"  → [0.21, -0.45, 0.87, ...]
"parts are almost out"      → [0.19, -0.43, 0.85, ...]  ← similar meaning, similar vector
"the weather is sunny"      → [-0.62, 0.11, -0.34, ...] ← unrelated, very different vector
```

The model used (`all-MiniLM-L6-v2`) produces 384-dimensional vectors. It runs completely locally — no API key needed, downloads once and is cached.

**ChromaDB** stores those vectors on disk. It's a database built specifically for searching vectors. When you search for "reorder policy for valves", it converts that query to a vector and finds the stored vectors closest to it using cosine similarity.

**Why ChromaDB and not MongoDB?**

This comes up often so it's worth explaining clearly. MongoDB is a document database — it's excellent at exact lookups like "give me all parts where supplier equals SupplierC" or "find all orders from 2023". That's filtering by value, and MongoDB handles it reliably.

But MongoDB cannot answer "find me documents whose *meaning* is similar to this question." It has no concept of semantic similarity. You could store embedding vectors inside MongoDB documents, but you'd have to load all of them into memory and compute similarity yourself — there's no built-in search index for that.

ChromaDB is built specifically for this. It stores vectors with a built-in HNSW index (Hierarchical Navigable Small World — a nearest-neighbor search algorithm) that makes similarity search fast even across thousands of documents. It also handles the embedding step, storage, and retrieval in one place.

Short version: MongoDB is the right tool for structured queries. ChromaDB is the right tool for semantic search. This project needs semantic search, so ChromaDB is the correct choice.

The 10 documents in the knowledge base cover: reorder policies, supplier profiles (SupplierA through D), past stockout incident reports, safety stock calculation formulas, and demand forecasting guidelines.

<br>

---

### `rag/retriever.py` (local) and `rag/retriever_cloud.py` (cloud)

The search half of RAG. Given a user's question, finds the most relevant documents from the knowledge base. Two implementations with identical public APIs — `agent.py` selects between them automatically at import time.

**`retriever.py` — local (ChromaDB + sentence-transformers):**
1. Take the question as plain text
2. Convert it to a vector using the same embedding model used during ingest
3. Ask ChromaDB: find the stored vectors closest to this one using cosine similarity
4. Return the top 3 matches — results below 0.3 similarity are filtered out

**`retriever_cloud.py` — cloud (numpy only, no heavy dependencies):**
1. Take the question as plain text
2. Tokenize and filter stopwords
3. Score each document by keyword overlap, weighted by term frequency
4. Return the top 3 matches from `rag/embeddings.npz` (pre-built, committed to repo)

The cloud retriever is less semantically precise than the ChromaDB version, but requires only numpy and installs in seconds. For the 10-document knowledge base in this project the quality difference is minimal in practice.

The retrieved document texts become the "context" that gets passed to Claude alongside the user's question.

<br>

---

### `agent/agent.py`

The brain of the project. A multi-provider AI agent that decides which tools to call, calls them, reads the results, and synthesizes a final answer.

**Multi-provider support:**
This file was originally built for Anthropic Claude only. It was later extended to support three providers using a thin abstraction layer — no third-party library required:

| Provider | SDK used | Tool call format |
|---|---|---|
| Anthropic | `anthropic` | Native tool use blocks |
| OpenAI | `openai` | `tools` + `tool_choice` |
| Groq | `openai` (same SDK, different `base_url`) | Same as OpenAI |

The user selects their provider and model in the app sidebar. The agent loop works identically regardless of which provider is active.

**What makes it an agent and not just a chatbot**

A chatbot takes your question and answers immediately from memory.

An agent follows a **ReAct loop** (Reason → Act → Observe → Reason → ...):

1. **Reason** — what do I need to answer this?
2. **Act** — call a tool to get real data
3. **Observe** — read what the tool returned
4. **Reason again** — is this enough? do I need more?
5. Repeat until ready
6. **Answer** — synthesize everything into a response

Claude natively supports tool use. Tools are defined as JSON schemas describing what each tool does and what inputs it accepts. Claude reads those descriptions and decides on its own which tools to call and in what order. The logic is not hardcoded.

**Three tools:**

`get_inventory_status` — reads the CSV, computes days of supply remaining for each part (current inventory ÷ average daily demand over last 30 days), and flags parts as:
- `CRITICAL` — will run out before the reorder can arrive
- `WARNING` — getting close, reorder should be placed soon
- `OK` — sufficient stock

`get_demand_forecast` — runs the trained TFT model to generate a 30-day forecast with p10/p50/p90 bounds. Falls back to a statistical baseline (mean + trend + standard deviation bounds) if the model hasn't been trained yet, so the agent is always usable.

`search_knowledge_base` — calls the RAG retriever to find relevant policies, supplier profiles, or incident reports from ChromaDB.

<br>

---

### `mlops/monitor.py`

Handles everything MLOps-related that happens after training.

**Prediction logging** — `log_prediction()` is called automatically every time `get_demand_forecast()` runs. It logs the part ID, p10/p50/p90 totals, daily median, forecast source, and timestamp to a dedicated MLflow experiment called `prediction-log`. This creates a full audit trail of what the model served and when.

**Drift detection** — `compute_drift_metrics()` pulls the prediction log, matches each logged prediction to the actual demand from the dataset, and computes:
- MAE over the last 30 days of predictions
- Calibration score: what % of actuals fell inside the p10–p90 band
- Degradation percentage vs training baseline
- A drift alert flag if degradation exceeds 20%

All computed drift metrics are logged back to MLflow so they can be trended over time.

**Model registry** — `get_registered_model_info()` queries the MLflow Model Registry for all versions of `supply-chain-tft`. `promote_to_production()` transitions a version from Staging to Production, archiving the previous Production version.

<br>

---

### `gradio_app.py`

The Gradio web application — the cloud entry point, deployed on Hugging Face Spaces. Three functional tabs:

**AI Assistant** — a chat interface with streaming tool-step output. Provider, model, and API key are selected from a shared row at the top of the page. Three quick-start buttons pre-fill common questions.

**Inventory Dashboard** — KPI summary, a category radio selector, two Plotly charts (days of supply + inventory vs demand), and the full inventory table.

**Demand Forecast** — part dropdown, historical demand chart, 30-day forecast chart with p10/p50/p90 quantile bounds, and a metadata summary table.

Uses only the packages in `requirements.txt` (no torch, no chromadb, no mlflow). The statistical forecaster and keyword RAG retriever handle everything on the free CPU tier.

---

### `app.py`

The Streamlit web application — local only, requires the full stack from `requirements-local.txt`. Adds a fourth tab not available in the cloud:

**AI Assistant** — same as Gradio, plus dynamic quick-start buttons built from actual supplier and category names in the dataset.

**Inventory Dashboard** — same as Gradio.

**Demand Forecast** — same as Gradio, plus part metadata panel and lead time recommendation info box.

**MLOps Monitor** — model registry with version promotion, prediction audit log, and drift detection with MAE and calibration metrics. Requires a running MLflow server (`mlflow ui`).

Streamlit is pure Python — no HTML, CSS, or JavaScript needed. Every time a user interacts with something, Streamlit re-runs the script and re-renders the UI.

<br>

---

### `notebooks/walkthrough.ipynb`

A guided tutorial notebook that runs through every component step by step with explanations. Intended for students and anyone reading the code for the first time.

**To use:** Open in Jupyter, select the `Supply Chain Agent (Python 3.12)` kernel, run cells top to bottom.

Covers: dataset exploration with demand pattern plots, TFT dataset construction, model architecture inspection, RAG ingestion and retrieval testing, individual tool testing, the full agent loop, and MLflow run inspection.

---

## LLM Provider — Bring Your Own Key

This project was originally built with **Anthropic Claude**. It now supports three providers so users can choose whichever they have access to.

| Provider | Models | Key format | Cost |
|---|---|---|---|
| **Anthropic** (original) | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-3-5 | `sk-ant-...` | Paid |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo | `sk-...` | Paid |
| **Groq** | llama3-70b, llama3-8b, mixtral-8x7b | `gsk_...` | Free tier |

The provider row at the top of the app lets you select a provider, pick a model, and paste your key. The key is held in memory only for that browser session and is never stored, logged, or shared.

---

**Why Groq is a good free option:**

Groq offers a generous free API tier with no credit card required. It runs open-source models (LLaMA 3, Mixtral) on custom hardware at very fast speeds. The agent works well with `llama3-70b-8192` — a good balance of quality and speed for free use.

---

**How to get each key:**

- **Anthropic:** [console.anthropic.com](https://console.anthropic.com/) → API Keys
- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Groq (free):** [console.groq.com/keys](https://console.groq.com/keys)

---

**For local development:**

To avoid typing your key on every restart, add it to the `.env` file at the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
# or
OPENAI_API_KEY=sk-your-key-here
# or
GROQ_API_KEY=gsk_your-key-here
```

The `.env` file is in `.gitignore` and will never be committed to GitHub.

---

## MLOps — Beyond Basic Training

This project implements the full MLOps lifecycle, not just training.

---

### Model Registry

After training, the TFT model is automatically registered in MLflow's Model Registry under the name `supply-chain-tft`. Each training run creates a new version, which starts in **Staging**. You can promote it to **Production** from the MLOps Monitor tab in the app.

This mirrors how real ML teams manage model releases — a model doesn't serve production traffic until it has been explicitly promoted.

```
Training run → v1 (Staging) → review → v1 (Production)
New training  → v2 (Staging) → review → v2 (Production), v1 (Archived)
```

Each registered version is tagged with: validation MAE, epochs trained, and number of parts.

---

### Prediction Logging

Every forecast generated — whether from the AI Assistant agent or the Demand Forecast tab — is automatically logged to MLflow's `prediction-log` experiment. Each log entry records:
- Which part was queried
- The p10, p50, p90 forecast values
- The timestamp
- Whether the TFT model or the statistical baseline served the prediction

This creates a full audit trail. In production, this is how you answer "what did the model actually predict on March 15th for PART_007?"

---

### Drift Detection

The **MLOps Monitor** tab includes a drift detection check that compares logged predictions against actual demand values from the dataset.

Three metrics are computed:

**MAE (Mean Absolute Error)** — the average prediction error in units/day. Compared against the baseline MAE from training. If the current MAE is more than 20% worse, a drift alert is triggered.

**Calibration score** — what percentage of actual demand values fell inside the predicted p10–p90 band. Should be around 80%. If it drops significantly, the model's uncertainty estimates have become unreliable.

**Drift alert** — a binary flag that fires when degradation exceeds the threshold. Logged back to MLflow so you can trend it over time with `mlflow ui`.

To run a drift check: open the MLOps Monitor tab → click **Run Drift Check**.

---

## Skills Demonstrated

| Skill Area | Implementation |
|---|---|
| Agentic AI, reasoning workflows | `agent/agent.py` — full ReAct loop, live reasoning steps in UI |
| RAG, embedding-based search systems | ChromaDB + sentence-transformers (local); numpy keyword retriever (cloud) |
| LLM projects, GenAI applications | Multi-provider: Anthropic, OpenAI, Groq |
| Demand forecast, material forecast | TFT (pytorch-forecasting) on 50-part supply chain time series |
| Deep learning frameworks | PyTorch + Lightning — TFT training with GPU support |
| MLOps — experiment tracking | MLflow logs all hyperparameters, metrics, and model artifacts |
| MLOps — model versioning | MLflow Model Registry with Staging → Production promotion |
| MLOps — prediction monitoring | Every forecast logged; drift detection with MAE + calibration score |
| AI-driven prototypes for stakeholders | Gradio app deployed on Hugging Face Spaces; local Streamlit app with full MLOps |
| Data mining, data processing | Pandas pipeline, synthetic multivariate time-series generation |
