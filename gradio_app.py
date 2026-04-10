"""
gradio_app.py
-------------
Gradio version of the Supply Chain Demand Agent UI.
Deployed on Hugging Face Spaces — no Streamlit, no chromadb, no torch required.

Tabs:
  1. AI Assistant   — chat with the multi-provider agent
  2. Inventory Dashboard — risk overview and category drill-down
  3. Demand Forecast     — 30-day forecast chart per part

MLOps Monitor tab is hidden on the cloud (requires local MLflow).

Run locally:
  pip install gradio
  python gradio_app.py

How to deploy on Hugging Face Spaces:
  1. Create a new Space (Gradio SDK)
  2. Push this repo to the Space
  3. Add ANTHROPIC_API_KEY / OPENAI_API_KEY to Space Secrets
  4. HF Spaces reads requirements-cloud.txt automatically if named
     requirements.txt in the repo root — see notes in that file.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from agent.agent import (
    run_agent_with_steps,
    get_inventory_status,
    PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    get_demand_forecast,
)

from mlops.mlops_cloud import (
    log_prediction,
    get_prediction_log,
    compute_drift_metrics,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
DATA_PATH = "data/supply_chain_data.csv"
_df_cache = None
_summary_cache = None


def _df_to_html(df: pd.DataFrame, risk_col: str = None) -> str:
    """Convert a DataFrame to a styled HTML table."""
    risk_colors = {"CRITICAL": "#e74c3c", "WARNING": "#f39c12", "OK": "#27ae60"}

    header_cells = "".join(
        f'<th style="padding:10px 14px;text-align:left;background:#1e2533;'
        f"color:#a0aec0;font-weight:600;font-size:13px;border-bottom:2px solid #2d3748;"
        f'white-space:nowrap;">{col}</th>'
        for col in df.columns
    )

    rows_html = ""
    for i, row in df.iterrows():
        row_bg = "#1a1f2e" if int(i) % 2 == 0 else "#1e2533"
        cells = ""
        for col, val in zip(df.columns, row):
            if col == risk_col and val in risk_colors:
                color = risk_colors[val]
                cell = (
                    f'<td style="padding:9px 14px;font-size:13px;border-bottom:1px solid #2d3748;">'
                    f'<span style="background:{color}22;color:{color};padding:3px 10px;'
                    f'border-radius:12px;font-weight:600;font-size:12px;">{val}</span></td>'
                )
            else:
                cell = (
                    f'<td style="padding:9px 14px;color:#e2e8f0;font-size:13px;'
                    f"border-bottom:1px solid #2d3748;font-family:'Inter',sans-serif;\">"
                    f"{val}</td>"
                )
            cells += cell
        rows_html += f'<tr style="background:{row_bg};">{cells}</tr>'

    return (
        '<div style="overflow-x:auto;overflow-y:auto;max-height:520px;'
        'border-radius:8px;border:1px solid #2d3748;">'
        "<table style=\"width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;\">"
        f'<thead style="position:sticky;top:0;z-index:1;"><tr>{header_cells}</tr></thead>'
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return _df_cache


def _compute_summary() -> pd.DataFrame:
    global _summary_cache
    if _summary_cache is not None:
        return _summary_cache

    df = _load_data()
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()

    last_30 = df[df["date"] >= latest_date - pd.Timedelta(days=30)]
    avg_demand = (
        last_30.groupby("part_id")["demand"]
        .mean()
        .rename("avg_daily_demand")
        .reset_index()
    )
    latest = latest.merge(avg_demand, on="part_id", how="left")
    latest["days_of_supply"] = (
        latest["inventory"] / latest["avg_daily_demand"].clip(lower=0.1)
    ).round(1)

    def risk_level(row):
        if row["days_of_supply"] < row["lead_time_days"]:
            return "CRITICAL"
        elif row["days_of_supply"] < 2 * row["lead_time_days"]:
            return "WARNING"
        return "OK"

    latest["risk"] = latest.apply(risk_level, axis=1)
    _summary_cache = latest.sort_values("days_of_supply")
    return _summary_cache


# ---------------------------------------------------------------------------
# TAB 1 — AI Assistant
# ---------------------------------------------------------------------------
def chat(message, history, provider, model, api_key):
    """Gradio chat function — yields streamed response with tool steps."""
    if not api_key:
        yield history + [[message, "Please enter your API key in the sidebar."]]
        return

    os.environ[PROVIDERS[provider]["env_key"]] = api_key

    steps_text = ""
    answer = ""
    tool_icons = {
        "get_inventory_status": "📦",
        "get_demand_forecast": "📈",
        "search_knowledge_base": "🔍",
    }

    try:
        for step in run_agent_with_steps(
            message,
            provider=provider,
            model=model,
            api_key=api_key,
        ):
            if step["type"] == "tool_start":
                icon = tool_icons.get(step["tool"], "🔧")
                steps_text += f"\n{icon} *{step['label']}...*"
                yield history + [[message, steps_text.strip()]]

            elif step["type"] == "tool_result":
                if steps_text:
                    steps_text = steps_text.rstrip("...*") + "* ✓\n"
                    steps_text += f"> {step['preview']}\n"
                yield history + [[message, steps_text.strip()]]

            elif step["type"] == "answer":
                answer = step["text"]
                separator = "\n\n---\n\n" if steps_text else ""
                yield history + [[message, steps_text.strip() + separator + answer]]

            elif step["type"] == "error":
                yield history + [[message, f"Error: {step['text']}"]]

    except Exception as e:
        yield history + [[message, f"An error occurred: {str(e)}"]]


# ---------------------------------------------------------------------------
# TAB 2 — Inventory Dashboard
# ---------------------------------------------------------------------------
def build_dashboard(selected_category):
    summary = _compute_summary()
    color_map = {"CRITICAL": "#e74c3c", "WARNING": "#f39c12", "OK": "#27ae60"}

    cat_data = summary[summary["category"] == selected_category].sort_values(
        "days_of_supply"
    )
    cat_avg_lead = int(cat_data["lead_time_days"].mean())

    # Chart 1 — Days of supply
    fig1 = px.bar(
        cat_data,
        x="part_id",
        y="days_of_supply",
        color="risk",
        color_discrete_map=color_map,
        hover_data=["supplier", "region", "inventory", "lead_time_days"],
        labels={"days_of_supply": "Days of Supply", "part_id": ""},
        title=f"Days of Supply — {selected_category}",
        text="days_of_supply",
    )
    fig1.add_hline(
        y=cat_avg_lead,
        line_dash="dash",
        line_color="rgba(255,255,255,0.45)",
        annotation_text=f"Avg lead time ({cat_avg_lead}d)",
    )
    fig1.update_traces(
        texttemplate="%{text:.0f}d",
        textposition="outside",
        cliponaxis=False,
    )
    max_dos = float(cat_data["days_of_supply"].max())
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        height=420,
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=40),
        yaxis=dict(range=[0, max_dos * 1.25]),
    )

    # Chart 2 — Inventory vs daily demand
    cat_melted = cat_data[["part_id", "inventory", "avg_daily_demand"]].copy()
    cat_melted["inventory"] = cat_melted["inventory"].astype(int)
    cat_melted["avg_daily_demand"] = cat_melted["avg_daily_demand"].round(1)

    fig2 = px.bar(
        cat_melted,
        x="part_id",
        y=["inventory", "avg_daily_demand"],
        barmode="group",
        labels={"value": "Units", "part_id": "", "variable": ""},
        title=f"Inventory vs Daily Demand — {selected_category}",
        color_discrete_map={
            "inventory": "#3498db",
            "avg_daily_demand": "#e67e22",
        },
        text_auto=True,
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        height=380,
    )

    # Summary table
    n_critical = int((cat_data["risk"] == "CRITICAL").sum())
    n_warning = int((cat_data["risk"] == "WARNING").sum())
    n_ok = int((cat_data["risk"] == "OK").sum())

    table_df = summary[
        [
            "part_id",
            "category",
            "supplier",
            "region",
            "inventory",
            "avg_daily_demand",
            "days_of_supply",
            "lead_time_days",
            "price_usd",
            "risk",
        ]
    ].copy()
    table_df["inventory"] = table_df["inventory"].astype(int)
    table_df["lead_time_days"] = table_df["lead_time_days"].astype(int)
    table_df["avg_daily_demand"] = table_df["avg_daily_demand"].round(1)
    table_df["days_of_supply"] = table_df["days_of_supply"].round(1)
    table_df["price_usd"] = table_df["price_usd"].round(2)
    table_df.columns = [
        "Part ID",
        "Category",
        "Supplier",
        "Region",
        "Inventory",
        "Avg Daily Demand",
        "Days of Supply",
        "Lead Time (days)",
        "Price (USD)",
        "Risk",
    ]
    table_html = _df_to_html(table_df, risk_col="Risk")

    kpis = f"""
**Total Parts:** {len(summary)} &nbsp;&nbsp;|&nbsp;&nbsp;
🔴 **Critical:** {(summary["risk"] == "CRITICAL").sum()} &nbsp;&nbsp;|&nbsp;&nbsp;
🟡 **Warning:** {(summary["risk"] == "WARNING").sum()} &nbsp;&nbsp;|&nbsp;&nbsp;
🟢 **OK:** {(summary["risk"] == "OK").sum()}

**{selected_category}:** {len(cat_data)} parts &nbsp;·&nbsp;
🔴 {n_critical} critical &nbsp;🟡 {n_warning} warning &nbsp;🟢 {n_ok} OK &nbsp;·&nbsp;
Avg lead time: {cat_avg_lead} days
"""
    return kpis, fig1, fig2, table_html


# ---------------------------------------------------------------------------
# TAB 3 — Demand Forecast
# ---------------------------------------------------------------------------
def build_forecast(part_id):
    df = _load_data()
    part_data = df[df["part_id"] == part_id]
    part_meta = part_data.iloc[0]

    lookback = min(60, len(part_data))
    recent_demand = part_data["demand"].tail(lookback).values
    avg = float(recent_demand.mean())
    std = float(recent_demand.std())
    lead = int(part_meta["lead_time_days"])
    horizon = 30
    trend = np.linspace(0, avg * 0.05, horizon)
    p50 = np.maximum(avg + trend, 0)
    p10 = np.maximum(p50 - 1.65 * std, 0)
    p90 = p50 + 1.65 * std

    forecast_dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1), periods=horizon
    )

    # History chart
    history_days = min(90, len(part_data))
    recent = part_data.tail(history_days)
    fig_hist = px.line(
        recent,
        x="date",
        y="demand",
        title=f"Last {history_days} Days — {part_id}",
        labels={"demand": "Daily Demand (units)", "date": "Date"},
    )
    fig_hist.update_traces(line_color="#3498db")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        height=480,
        margin=dict(t=60, b=60, l=70, r=30),
    )

    # Forecast chart
    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(p90) + list(p10[::-1]),
            fill="toself",
            fillcolor="rgba(148,163,184,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Uncertainty (p10–p90)",
            hoverinfo="skip",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=p90,
            line=dict(color="#f87171", dash="dash", width=2),
            name="p90 — order qty (90% SL)",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=p50,
            line=dict(color="#e2e8f0", width=3),
            name="p50 — median forecast",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=p10,
            line=dict(color="#6ee7b7", dash="dash", width=2),
            name="p10 — lower bound",
        )
    )
    fig_fc.update_layout(
        title=f"30-Day Demand Forecast — {part_id}",
        xaxis_title="Date",
        yaxis_title="Demand (units/day)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        height=500,
        margin=dict(t=60, b=100, l=70, r=30),
    )

    meta_md = f"""
**Category:** {part_meta["category"]} &nbsp;|&nbsp;
**Supplier:** {part_meta["supplier"]} &nbsp;|&nbsp;
**Region:** {part_meta["region"]} &nbsp;|&nbsp;
**Lead Time:** {lead} days &nbsp;|&nbsp;
**Unit Price:** ${float(part_meta["price_usd"]):,.2f}

| Metric | Value |
|---|---|
| Daily demand (median) | {avg:.1f} units/day |
| {horizon}-day total (p50) | {int(p50.sum()):,} units |
| Lower bound (p10) | {int(p10.sum()):,} units |
| Order qty for 90% SL (p90) | {int(p90.sum()):,} units |
"""
    # Log to persistent storage for MLOps monitor
    log_prediction(
        part_id=part_id,
        p50_daily=avg,
        p50_total=float(p50.sum()),
        p10_total=float(p10.sum()),
        p90_total=float(p90.sum()),
        horizon_days=horizon,
        source="statistical",
    )
    return fig_hist, fig_fc, meta_md


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    summary = _compute_summary()
    df = _load_data()
    part_ids = sorted(df["part_id"].unique())
    categories = sorted(summary["category"].unique())

    provider_names = list(PROVIDERS.keys())
    default_models = {p: PROVIDERS[p]["models"][0] for p in provider_names}

    _theme = gr.themes.Soft() if hasattr(gr, "themes") else None
    with gr.Blocks(title="Supply Chain Demand Agent", theme=_theme) as demo:
        gr.Markdown(
            "# 📦 Supply Chain Demand Agent\n"
            "An agentic AI system for supply chain demand forecasting. "
            "Uses RAG to retrieve supplier policies and a statistical forecaster "
            "for 30-day demand predictions."
        )

        # ── Shared API key row ──
        with gr.Row():
            provider_dd = gr.Dropdown(
                choices=provider_names,
                value=DEFAULT_PROVIDER,
                label="LLM Provider",
                scale=1,
            )
            model_dd = gr.Dropdown(
                choices=PROVIDERS[DEFAULT_PROVIDER]["models"],
                value=DEFAULT_MODEL,
                label="Model",
                scale=1,
            )
            api_key_box = gr.Textbox(
                label="API Key (held in memory only — never stored)",
                placeholder="Paste your API key here",
                type="password",
                scale=3,
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
            )

        def update_models(provider):
            models = PROVIDERS[provider]["models"]
            return gr.Dropdown(choices=models, value=models[0])

        provider_dd.change(update_models, inputs=provider_dd, outputs=model_dd)

        # ── Tabs ──
        with gr.Tabs():
            # TAB 1 — AI Assistant
            with gr.Tab("🤖 AI Assistant"):
                gr.Markdown(
                    "Ask questions about inventory levels, demand forecasts, "
                    "supplier policies, or which parts need attention. "
                    "The agent decides which tools to call and shows its reasoning."
                )
                chatbot = gr.Chatbot(height=480, label="Supply Chain Agent")
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask the supply chain agent...",
                        label="Your question",
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    gr.Button("Which parts are at risk?").click(
                        fn=lambda: (
                            "Which parts are most at risk of running out soon? What immediate actions should we take?"
                        ),
                        outputs=msg_box,
                    )
                    gr.Button("SupplierA reliability?").click(
                        fn=lambda: (
                            "What are the known reliability issues with SupplierA?"
                        ),
                        outputs=msg_box,
                    )
                    gr.Button("Safety stock formula?").click(
                        fn=lambda: (
                            "How should I calculate safety stock? Walk me through the formula."
                        ),
                        outputs=msg_box,
                    )

                clear_btn = gr.Button("Clear conversation", size="sm")

                def submit(message, history, provider, model, api_key):
                    history = history or []
                    for updated in chat(message, history, provider, model, api_key):
                        yield updated, ""

                send_btn.click(
                    submit,
                    inputs=[msg_box, chatbot, provider_dd, model_dd, api_key_box],
                    outputs=[chatbot, msg_box],
                )
                msg_box.submit(
                    submit,
                    inputs=[msg_box, chatbot, provider_dd, model_dd, api_key_box],
                    outputs=[chatbot, msg_box],
                )
                clear_btn.click(lambda: [], outputs=chatbot)

            # TAB 2 — Inventory Dashboard
            with gr.Tab("📊 Inventory Dashboard"):
                gr.Markdown("### Inventory Risk Dashboard")
                kpi_md = gr.Markdown()
                category_radio = gr.Radio(
                    choices=categories,
                    value=categories[0],
                    label="Select Category",
                )
                supply_chart = gr.Plot(label="Days of Supply")
                demand_chart = gr.Plot(label="Inventory vs Daily Demand")
                gr.Markdown("### Full Inventory Table")
                inv_table = gr.HTML()

                def refresh_dashboard(cat):
                    kpis, f1, f2, tbl = build_dashboard(cat)
                    return kpis, f1, f2, tbl

                category_radio.change(
                    refresh_dashboard,
                    inputs=category_radio,
                    outputs=[kpi_md, supply_chart, demand_chart, inv_table],
                )
                # Load on startup
                demo.load(
                    refresh_dashboard,
                    inputs=category_radio,
                    outputs=[kpi_md, supply_chart, demand_chart, inv_table],
                )

            # TAB 3 — Demand Forecast
            with gr.Tab("📈 Demand Forecast"):
                gr.Markdown(
                    "### 30-Day Demand Forecast\n"
                    "Select a part to see its demand forecast. "
                    "The shaded area shows the uncertainty range (p10–p90). "
                    "Use the p90 line for safety stock calculations."
                )
                part_dd = gr.Dropdown(
                    choices=part_ids,
                    value=part_ids[0],
                    label="Select Part",
                )
                meta_md = gr.Markdown()
                hist_chart = gr.Plot(label="Historical Demand")
                fc_chart = gr.Plot(label="30-Day Forecast")

                def refresh_forecast(part_id):
                    h, f, m = build_forecast(part_id)
                    return h, f, m

                part_dd.change(
                    refresh_forecast,
                    inputs=part_dd,
                    outputs=[hist_chart, fc_chart, meta_md],
                )
                demo.load(
                    refresh_forecast,
                    inputs=part_dd,
                    outputs=[hist_chart, fc_chart, meta_md],
                )

            # TAB 4 — MLOps Monitor
            with gr.Tab("⚙️ MLOps Monitor"):
                gr.Markdown(
                    "### MLOps Monitor\n"
                    "Every forecast you run is automatically logged here. "
                    "Logs persist across sessions using HF Spaces persistent storage (`/data`). "
                    "Drift detection compares your forecast accuracy against a naive baseline."
                )

                with gr.Row():
                    refresh_btn = gr.Button("Refresh Log", variant="secondary")
                    drift_btn = gr.Button("Run Drift Check", variant="primary")

                # ── Drift metrics ──
                drift_md = gr.Markdown("*Click 'Run Drift Check' to compute metrics.*")

                # ── Prediction log ──
                gr.Markdown("### Prediction Log (last 100 forecasts)")
                log_table = gr.HTML()

                # ── Most-queried parts chart ──
                popular_chart = gr.Plot(label="Most Queried Parts")

                def load_mlops():
                    df = get_prediction_log(limit=100)
                    if df.empty:
                        chart = go.Figure().update_layout(
                            title="No predictions logged yet",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#e2e8f0",
                        )
                        return (
                            "<p style='color:#a0aec0;padding:12px;'>No predictions logged yet.</p>",
                            chart,
                        )
                    counts = df["part_id"].value_counts().head(15).reset_index()
                    counts.columns = ["part_id", "count"]
                    chart = px.bar(
                        counts,
                        x="part_id",
                        y="count",
                        title="Most Queried Parts",
                        labels={"count": "Forecasts Run", "part_id": ""},
                        color="count",
                        color_continuous_scale="Blues",
                    )
                    chart.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        height=320,
                        showlegend=False,
                        coloraxis_showscale=False,
                    )
                    display_cols = [
                        "timestamp",
                        "part_id",
                        "source",
                        "p50_daily",
                        "p50_total",
                        "p10_total",
                        "p90_total",
                    ]
                    log_df = df[display_cols].copy()
                    log_df.columns = [
                        "Timestamp",
                        "Part ID",
                        "Source",
                        "Daily Demand (p50)",
                        "30d Total (p50)",
                        "30d Lower (p10)",
                        "30d Upper (p90)",
                    ]
                    return _df_to_html(log_df), chart

                def run_drift():
                    m = compute_drift_metrics()
                    if m["status"] == "NO DATA":
                        return (
                            "**No predictions logged yet.** Run some forecasts first."
                        )
                    flag = "🔴 DRIFT DETECTED" if m["drift_flag"] else "🟢 OK"
                    return (
                        f"**Status:** {flag}\n\n"
                        f"| Metric | Value |\n"
                        f"|---|---|\n"
                        f"| Predictions evaluated | {m['n_predictions']} |\n"
                        f"| Forecast MAE (p50 vs actual avg demand) | {m['mae']} units/day |\n"
                        f"| Naive baseline MAE (predict global mean) | {m['baseline_mae']} units/day |\n"
                        f"| Calibration (actuals inside p10–p90 band) | {m['calibration']}% |\n"
                        f"\n*Drift flag triggers when forecast MAE > 1.5× baseline MAE.*"
                    )

                refresh_btn.click(load_mlops, outputs=[log_table, popular_chart])
                drift_btn.click(run_drift, outputs=drift_md)
                demo.load(load_mlops, outputs=[log_table, popular_chart])

    return demo


# Build the demo at module level so HF Spaces can find it
demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
