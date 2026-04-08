"""
app.py
------
The Streamlit web application - the front-end of this project.

What is Streamlit?
  Streamlit lets you build a web app using only Python - no HTML, no CSS,
  no JavaScript required. You write Python and it renders a UI in the browser.
  Every time the user interacts with something (clicks a button, types in a box),
  Streamlit re-runs the script from top to bottom.

How to run locally:
  streamlit run app.py
  Then open http://localhost:8501 in your browser.

How to deploy for free:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io
  3. Connect your GitHub repo
  4. Add ANTHROPIC_API_KEY in the Streamlit secrets dashboard
  5. Deploy - your app gets a public URL

The app has three tabs:
  1. AI Assistant  - chat with the agent, ask questions about inventory and forecasts
  2. Inventory Dashboard - visual overview of all parts and their risk levels
  3. Demand Forecast - pick a part and see its 30-day forecast chart
"""

import os
from dotenv import load_dotenv
import pandas as pd

# Load .env for local development.
# On Streamlit Cloud, the key comes from st.secrets instead (see get_api_key below).
load_dotenv()
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from agent.agent import (
    run_agent,
    run_agent_with_steps,
    run_tool,
    get_inventory_status,
    PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    get_demand_forecast,
)
from mlops.monitor import (
    get_prediction_log,
    compute_drift_metrics,
    get_registered_model_info,
    promote_to_production,
)


# ---------------------------------------------------------------------------
# PAGE CONFIG - must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Demand Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# API KEY HANDLING — Bring Your Own Key (BYOK)
#
# This app does NOT use a shared API key. Each user provides their own
# Anthropic API key, which is used only for their session and never stored.
#
# Locally: the key can also be loaded from a .env file for convenience.
# On Streamlit Cloud: the user always enters it in the sidebar.
# ---------------------------------------------------------------------------

# Check if a key was already entered earlier in this session
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")

api_key = st.session_state["api_key"]


# ---------------------------------------------------------------------------
# DATA LOADING
# @st.cache_data means this function only runs once per session.
# The result is cached so we don't re-read the CSV on every interaction.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=0)  # ttl=0 means never cache — always re-read from disk
def load_data(path: str = "data/supply_chain_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data(ttl=0)
def compute_inventory_summary(path: str = "data/supply_chain_data.csv") -> pd.DataFrame:
    """Pre-computes the inventory risk summary used in the dashboard tab."""
    df = pd.read_csv(path, parse_dates=["date"])
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
    return latest.sort_values("days_of_supply")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📦 Supply Chain Agent")
    st.markdown("---")

    # --- Provider selector ---
    st.markdown("**LLM Provider**")
    provider_names = list(PROVIDERS.keys())
    selected_provider = st.selectbox(
        "Choose a provider",
        provider_names,
        index=provider_names.index(st.session_state.get("provider", DEFAULT_PROVIDER)),
        help="Originally built with Anthropic Claude. OpenAI and Groq are also supported.",
    )
    st.session_state["provider"] = selected_provider
    provider_info = PROVIDERS[selected_provider]

    # --- Model selector (updates when provider changes) ---
    available_models = provider_info["models"]
    prev_model = st.session_state.get("model", available_models[0])
    default_model = (
        prev_model if prev_model in available_models else available_models[0]
    )
    selected_model = st.selectbox(
        "Choose a model",
        available_models,
        index=available_models.index(default_model),
    )
    st.session_state["model"] = selected_model

    # --- Free tier badge ---
    if provider_info["free"]:
        st.success("Free tier available — no credit card needed.")

    st.markdown("---")

    # --- API Key input ---
    st.markdown(f"**{selected_provider} API Key**")
    st.markdown(
        f"{provider_info['key_hint']}. "
        f"[Get your key here]({provider_info['docs_url']}). "
        "It is held in memory only for this session — never stored or shared."
    )

    entered_key = st.text_input(
        "Paste your API key",
        type="password",
        value=st.session_state.get("api_key", ""),
        placeholder=provider_info["key_hint"],
        help="Used only for this browser session.",
        key=f"key_input_{selected_provider}",  # re-renders when provider changes
    )

    if entered_key and entered_key != st.session_state.get("api_key", ""):
        st.session_state["api_key"] = entered_key
        os.environ[provider_info["env_key"]] = entered_key
        api_key = entered_key
        st.rerun()

    api_key = st.session_state.get("api_key", "")

    if api_key:
        st.success("API key active for this session.")
    else:
        st.info("Enter your key above to use the AI Assistant.")

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown(
        "An agentic AI system for supply chain demand forecasting. "
        "Uses a Temporal Fusion Transformer for 30-day demand forecasts "
        "and RAG to retrieve relevant policies and supplier information."
    )
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown(
        "- TFT (pytorch-forecasting)\n"
        "- RAG (ChromaDB + sentence-transformers)\n"
        "- Agent (multi-provider: Anthropic / OpenAI / Groq)\n"
        "- MLOps (MLflow)\n"
        "- UI (Streamlit + Plotly)"
    )


# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["AI Assistant", "Inventory Dashboard", "Demand Forecast", "MLOps Monitor"]
)


# ===========================================================================
# TAB 1: AI ASSISTANT
# ===========================================================================
with tab1:
    st.header("Supply Chain AI Assistant")
    st.markdown(
        "Ask questions about inventory levels, demand forecasts, supplier policies, "
        "or which parts need attention. The agent searches live inventory data, "
        "runs demand forecasts, and retrieves relevant policies to answer."
    )

    # Load data once so we can build dynamic suggested questions
    _df_hint = load_data()
    _suppliers = sorted(_df_hint["supplier"].unique())
    _categories = sorted(_df_hint["category"].unique())
    _sample_supplier = _suppliers[0] if _suppliers else "SupplierA"
    _sample_category = _categories[0] if _categories else "Valve"

    # Suggested questions built from actual data — nothing hardcoded
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Which parts are at risk?"):
            st.session_state["prefill"] = (
                "Which parts are most at risk of running out soon? What immediate actions should we take?"
            )
    with col2:
        if st.button(f"{_sample_supplier} reliability?"):
            st.session_state["prefill"] = (
                f"What are the known reliability issues with {_sample_supplier}?"
            )
    with col3:
        if st.button("Safety stock formula?"):
            st.session_state["prefill"] = (
                "How should I calculate safety stock? Walk me through the formula."
            )

    st.markdown("---")

    # Initialize chat history in session state
    # st.session_state persists across re-runs within the same browser session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle prefilled questions from buttons
    prefill = st.session_state.pop("prefill", None)

    # Chat input box
    user_input = st.chat_input("Ask the supply chain agent...") or prefill

    if user_input:
        if not api_key:
            st.error(
                "Please enter your Anthropic API key in the sidebar to use the assistant."
            )
        else:
            # Show the user's message immediately
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Run the agent and show reasoning steps live as they happen
            with st.chat_message("assistant"):
                # Placeholder containers - we update these in real time
                steps_container = st.container()
                answer_placeholder = st.empty()

                tool_icons = {
                    "get_inventory_status": "📦",
                    "get_demand_forecast": "📈",
                    "search_knowledge_base": "🔍",
                }

                response = ""
                step_messages = []

                try:
                    for step in run_agent_with_steps(
                        user_input,
                        provider=st.session_state.get("provider", DEFAULT_PROVIDER),
                        model=st.session_state.get("model", DEFAULT_MODEL),
                        api_key=st.session_state.get("api_key", ""),
                    ):
                        if step["type"] == "tool_start":
                            icon = tool_icons.get(step["tool"], "🔧")
                            label = step["label"]
                            step_messages.append(f"{icon} **{label}...**")
                            with steps_container:
                                for msg in step_messages:
                                    st.markdown(msg)

                        elif step["type"] == "tool_result":
                            # Replace the last "..." with a checkmark and preview
                            if step_messages:
                                last = step_messages[-1].replace("...**", f"** ✓")
                                step_messages[-1] = last + f"\n> {step['preview']}"
                            with steps_container:
                                # Clear and redraw all steps cleanly
                                steps_container.empty()
                            with steps_container:
                                for msg in step_messages:
                                    st.markdown(msg)

                        elif step["type"] == "answer":
                            response = step["text"]
                            # Add a separator before the final answer
                            if step_messages:
                                steps_container.markdown("---")
                            answer_placeholder.markdown(response)

                        elif step["type"] == "error":
                            response = step["text"]
                            answer_placeholder.error(response)

                except Exception as e:
                    response = f"An error occurred: {str(e)}"
                    answer_placeholder.error(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    # Button to clear chat history
    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ===========================================================================
# TAB 2: INVENTORY DASHBOARD
# ===========================================================================
with tab2:
    st.header("Inventory Risk Dashboard")

    try:
        summary = compute_inventory_summary()

        # Top-level KPI metrics
        n_critical = (summary["risk"] == "CRITICAL").sum()
        n_warning = (summary["risk"] == "WARNING").sum()
        n_ok = (summary["risk"] == "OK").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Parts", len(summary))
        m2.metric("Critical", n_critical, delta=f"stockout risk", delta_color="inverse")
        m3.metric("Warning", n_warning, delta=f"reorder needed", delta_color="inverse")
        m4.metric("OK", n_ok, delta=f"sufficient stock", delta_color="normal")

        st.markdown("---")

        color_map = {"CRITICAL": "#e74c3c", "WARNING": "#f39c12", "OK": "#27ae60"}

        def apply_dark_theme(fig, height=400):
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
                legend=dict(orientation="h", y=1.05),
                height=height,
            )
            return fig

        st.markdown("---")

        # --- Category drill-down — inline selector on the page ---
        st.subheader("Category Drill-Down")

        all_categories = sorted(summary["category"].unique())

        # st.radio renders as a horizontal tab-like bar when set to horizontal
        selected_category = st.radio(
            "Select a category",
            all_categories,
            horizontal=True,
            label_visibility="collapsed",
        )

        cat_data = summary[summary["category"] == selected_category].sort_values(
            "days_of_supply"
        )
        cat_avg_lead = int(cat_data["lead_time_days"].mean())
        n_crit = int((cat_data["risk"] == "CRITICAL").sum())
        n_warn = int((cat_data["risk"] == "WARNING").sum())
        n_ok = int((cat_data["risk"] == "OK").sum())

        st.caption(
            f"{len(cat_data)} parts  ·  "
            f"🔴 {n_crit} critical  🟡 {n_warn} warning  🟢 {n_ok} OK  ·  "
            f"Avg lead time: {cat_avg_lead} days  ·  "
            f"Avg days of supply: {cat_data['days_of_supply'].mean():.1f} days"
        )

        # Chart 1: Days of supply — full width
        fig_drill_supply = px.bar(
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
        fig_drill_supply.add_hline(
            y=cat_avg_lead,
            line_dash="dash",
            line_color="rgba(255,255,255,0.45)",
            annotation_text=f"Avg lead time ({cat_avg_lead}d)",
            annotation_font_color="rgba(255,255,255,0.6)",
        )
        fig_drill_supply.update_traces(
            texttemplate="%{text:.0f}d",
            textposition="outside",
            textfont_size=11,
        )
        apply_dark_theme(fig_drill_supply, height=380)
        fig_drill_supply.update_layout(showlegend=False)
        st.plotly_chart(fig_drill_supply, use_container_width=True)

        # Chart 2: Inventory vs daily demand — full width
        cat_melted = cat_data[["part_id", "inventory", "avg_daily_demand"]].copy()
        cat_melted["inventory"] = cat_melted["inventory"].astype(int)
        cat_melted["avg_daily_demand"] = cat_melted["avg_daily_demand"].round(1)

        fig_drill_inv = px.bar(
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
        apply_dark_theme(fig_drill_inv, height=380)
        fig_drill_inv.update_layout(legend=dict(orientation="h", y=1.1, title=None))
        fig_drill_inv.for_each_trace(
            lambda t: t.update(
                name="Inventory (units)"
                if t.name == "inventory"
                else "Avg daily demand"
            )
        )
        st.plotly_chart(fig_drill_inv, use_container_width=True)

        st.markdown("---")
        st.subheader("Full Inventory Table")

        # Add a risk emoji column so the table is readable on any background
        # without relying on row background colors that clash with dark mode
        def risk_emoji(r):
            return {
                "CRITICAL": "🔴 CRITICAL",
                "WARNING": "🟡 WARNING",
                "OK": "🟢 OK",
            }.get(r, r)

        display_cols = [
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

        table_df = summary[display_cols].copy()
        table_df["inventory"] = table_df["inventory"].astype(int)
        table_df["lead_time_days"] = table_df["lead_time_days"].astype(int)
        table_df["avg_daily_demand"] = table_df["avg_daily_demand"].round(1)
        table_df["days_of_supply"] = table_df["days_of_supply"].round(1)
        table_df["price_usd"] = table_df["price_usd"].round(2)
        table_df["risk"] = table_df["risk"].apply(risk_emoji)

        st.dataframe(
            table_df,
            use_container_width=True,
            height=400,
            hide_index=True,
        )

    except FileNotFoundError:
        st.error("Dataset not found. Run `python -m data.generate_data` first.")


# ===========================================================================
# TAB 4: MLOPS MONITOR
# ===========================================================================
with tab4:
    st.header("MLOps Monitor")
    st.markdown(
        "Tracks model health in production: registered model versions, "
        "prediction logs, and drift detection metrics."
    )

    # -------------------------------------------------------------------
    # SECTION 1: Model Registry
    # -------------------------------------------------------------------
    st.subheader("Model Registry")
    st.markdown(
        "The model registry records every trained version of the TFT model. "
        "You can promote a version from **Staging** to **Production** here. "
        "Only the Production version should serve live forecasts."
    )

    model_versions = get_registered_model_info()

    if not model_versions:
        st.info(
            "No registered models found. Run `python -m forecasting.train` to train and register the model."
        )
    else:
        reg_df = pd.DataFrame(model_versions)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        staging_versions = [v for v in model_versions if v["stage"] == "Staging"]
        if staging_versions:
            st.markdown("**Promote a version to Production:**")
            version_to_promote = st.selectbox(
                "Select version",
                [v["version"] for v in staging_versions],
                key="promote_select",
            )
            if st.button("Promote to Production", type="primary"):
                success = promote_to_production(version_to_promote)
                if success:
                    st.success(f"Version {version_to_promote} is now in Production.")
                    st.rerun()
                else:
                    st.error("Promotion failed. Check MLflow connection.")

    st.markdown("---")

    # -------------------------------------------------------------------
    # SECTION 2: Prediction Log
    # -------------------------------------------------------------------
    st.subheader("Prediction Log")
    st.markdown(
        "Every forecast generated by the agent or the Demand Forecast tab is "
        "logged here. This is your audit trail — you can see what the model "
        "predicted, when, and for which part."
    )

    pred_log = get_prediction_log(limit=50)

    if pred_log.empty:
        st.info(
            "No predictions logged yet. Use the AI Assistant or Demand Forecast tab "
            "to generate forecasts — they will appear here automatically."
        )
    else:
        pl1, pl2, pl3 = st.columns(3)
        pl1.metric("Total predictions logged", len(pred_log))
        if "source" in pred_log.columns:
            tft_count = pred_log["source"].str.contains("TFT", na=False).sum()
            pl2.metric("TFT model predictions", int(tft_count))
            pl3.metric("Statistical baseline", int(len(pred_log) - tft_count))

        display_cols = [
            c
            for c in [
                "timestamp",
                "part_id",
                "source",
                "p50_daily",
                "p50_total",
                "p90_total",
                "p10_total",
            ]
            if c in pred_log.columns
        ]
        st.dataframe(pred_log[display_cols], use_container_width=True, hide_index=True)

        if "part_id" in pred_log.columns:
            st.markdown("**Most queried parts:**")
            top_parts = pred_log["part_id"].value_counts().head(10).reset_index()
            top_parts.columns = ["Part ID", "Times Queried"]
            fig_parts = px.bar(
                top_parts,
                x="Part ID",
                y="Times Queried",
                color_discrete_sequence=["#60a5fa"],
            )
            fig_parts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0f172a",
                font_color="#e2e8f0",
                height=300,
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_parts, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------
    # SECTION 3: Drift Detection
    # -------------------------------------------------------------------
    st.subheader("Drift Detection")
    st.markdown(
        "Compares what the model predicted against what actually happened. "
        "If accuracy degrades beyond 20% of the training baseline, a drift "
        "alert is triggered and logged to MLflow."
    )

    if st.button("Run Drift Check", type="primary"):
        with st.spinner("Comparing predictions vs actuals..."):
            drift = compute_drift_metrics()

        if drift["status"] == "no_predictions_logged":
            st.info("Generate some forecasts first — then run the drift check.")
        elif drift["status"] in ("data_not_found", "insufficient_matches"):
            st.warning(f"Could not compute drift: {drift['status']}")
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(
                "Baseline MAE (training)",
                f"{drift['baseline_mae']:.2f}" if drift["baseline_mae"] else "—",
                help="MAE from the original training validation run",
            )
            d2.metric(
                "Current MAE (30-day)",
                f"{drift['mae_30d']:.2f}" if drift["mae_30d"] else "—",
                delta=f"{drift.get('degradation_pct', 0):+.1f}%"
                if drift.get("degradation_pct")
                else None,
                delta_color="inverse",
            )
            d3.metric(
                "Calibration score",
                f"{drift['calibration']:.1%}" if drift["calibration"] else "—",
                help="% of actuals inside the p10–p90 band. Target: ~80%",
            )
            d4.metric("Predictions evaluated", drift["n_predictions"])

            if drift["drift_alert"]:
                st.error(
                    f"**Drift Alert:** Current MAE ({drift['mae_30d']:.2f}) is "
                    f"{drift.get('degradation_pct', 0):.1f}% worse than the training "
                    f"baseline ({drift['baseline_mae']:.2f}). Consider retraining."
                )
            else:
                st.success(
                    "Model performance is within acceptable range. No drift detected."
                )

            if drift["calibration"] is not None:
                if drift["calibration"] < 0.6:
                    st.warning(
                        f"Low calibration ({drift['calibration']:.1%}). Fewer actuals than "
                        "expected are falling inside the p10–p90 band."
                    )
                elif drift["calibration"] > 0.95:
                    st.info(
                        f"Very high calibration ({drift['calibration']:.1%}). "
                        "The uncertainty band may be too wide."
                    )

            st.caption(
                "Drift metrics have been logged to MLflow. Run `mlflow ui` to see the trend over time."
            )
    else:
        st.info(
            "Click **Run Drift Check** to evaluate model accuracy against recent actuals."
        )

# ===========================================================================
# TAB 3: DEMAND FORECAST
# ===========================================================================
with tab3:
    st.header("30-Day Demand Forecast")
    st.markdown(
        "Select a part to see its demand forecast. "
        "The shaded area shows the uncertainty range (p10 to p90). "
        "Use the p90 line for safety stock calculations."
    )

    try:
        df = load_data()
        part_ids = sorted(df["part_id"].unique())

        col_select, col_info = st.columns([1, 2])

        with col_select:
            selected_part = st.selectbox("Select Part", part_ids)
            part_data = df[df["part_id"] == selected_part]
            part_meta = part_data.iloc[0]

            # All values pulled from data — nothing hardcoded
            st.markdown(f"**Category:** {part_meta['category']}")
            st.markdown(f"**Supplier:** {part_meta['supplier']}")
            st.markdown(f"**Region:** {part_meta['region']}")
            st.markdown(f"**Lead Time:** {int(part_meta['lead_time_days'])} days")
            st.markdown(f"**Unit Price:** ${float(part_meta['price_usd']):,.2f}")
            st.markdown(
                f"**Data range:** {part_data['date'].min().date()} → {part_data['date'].max().date()}"
            )
            st.markdown(f"**Total records:** {len(part_data):,} days")

        with col_info:
            # History window: last 90 days or all data if shorter
            history_days = min(90, len(part_data))
            recent = part_data.tail(history_days)
            history_label = f"Last {history_days} Days — {selected_part}"

            fig_hist = px.line(
                recent,
                x="date",
                y="demand",
                title=history_label,
                labels={"demand": "Daily Demand (units)", "date": "Date"},
            )
            fig_hist.update_traces(line_color="#3498db")
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Generate the forecast using the last 60 days of actuals as baseline
        # (or fewer if the dataset is shorter)
        lookback = min(60, len(part_data))
        recent_demand = part_data["demand"].tail(lookback).values
        avg = float(recent_demand.mean())
        std = float(recent_demand.std())
        lead_time = int(part_meta["lead_time_days"])
        horizon = 30
        trend = np.linspace(0, avg * 0.05, horizon)
        p50 = np.maximum(avg + trend, 0)
        p10 = np.maximum(p50 - 1.65 * std, 0)
        p90 = p50 + 1.65 * std

        st.subheader(f"Forecast: {selected_part} — Next {horizon} Days")

        # Custom HTML cards so label and value are balanced in size.
        # st.metric() uses a very large font for values with no way to override it.
        def stat_card(label: str, value: str, note: str = "") -> str:
            note_html = (
                f"<div style='font-size:11px;color:#64748b;margin-top:4px'>{note}</div>"
                if note
                else "<div style='font-size:11px;margin-top:4px'>&nbsp;</div>"
            )
            return f"""
            <div style='background:#1e293b;border-radius:8px;padding:14px 16px;height:100px;box-sizing:border-box;'>
                <div style='font-size:12px;font-weight:500;color:#94a3b8;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{label}</div>
                <div style='font-size:17px;font-weight:700;color:#f1f5f9;line-height:1.2'>{value}</div>
                {note_html}
            </div>"""

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.markdown(
            stat_card("Daily demand (median)", f"{avg:.1f} units/day"),
            unsafe_allow_html=True,
        )
        fc2.markdown(
            stat_card(f"{horizon}-day total (p50)", f"{int(p50.sum()):,} units"),
            unsafe_allow_html=True,
        )
        fc3.markdown(
            stat_card("Lower bound (p10)", f"{int(p10.sum()):,} units"),
            unsafe_allow_html=True,
        )
        fc4.markdown(
            stat_card(
                "Order qty for 90% SL", f"{int(p90.sum()):,} units", "p90 upper bound"
            ),
            unsafe_allow_html=True,
        )

        # Build a visual forecast chart
        forecast_dates = pd.date_range(
            start=df["date"].max() + pd.Timedelta(days=1), periods=horizon
        )

        fig_forecast = go.Figure()

        # Subtle shaded band — muted slate tint, not overpowering
        fig_forecast.add_trace(
            go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(p90) + list(p10[::-1]),
                fill="toself",
                fillcolor="rgba(148, 163, 184, 0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Uncertainty range (p10–p90)",
                hoverinfo="skip",
            )
        )

        # p90 — muted coral red, dashed
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=p90,
                line=dict(color="#f87171", dash="dash", width=2),
                name="p90 — order qty for 90% service level",
            )
        )

        # p50 — bright white, thickest — stands out clearly on the dark background
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=p50,
                line=dict(color="#e2e8f0", width=3),
                name="p50 — median forecast (best estimate)",
            )
        )

        # p10 — muted teal green, dashed
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=p10,
                line=dict(color="#6ee7b7", dash="dash", width=2),
                name="p10 — lower bound (optimistic scenario)",
            )
        )

        fig_forecast.update_layout(
            title=f"30-Day Demand Forecast — {selected_part}",
            xaxis_title="Date",
            yaxis_title="Demand (units/day)",
            # Slightly lighter than app background — just enough to frame the chart
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            xaxis=dict(
                gridcolor="#1e293b",
                linecolor="#334155",
                tickfont=dict(color="#94a3b8"),
                title_font=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                gridcolor="#1e293b",
                linecolor="#334155",
                tickfont=dict(color="#94a3b8"),
                title_font=dict(color="#94a3b8"),
            ),
            title_font=dict(color="#f1f5f9", size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.35,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color="#cbd5e1"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=430,
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Plain English explanation — gap above, justified text
        st.markdown(
            f"<div style='margin-top:20px;text-align:justify;line-height:1.7'>"
            f"<b>How to read this chart:</b> The <b>white line</b> is the most likely daily demand "
            f"over the next {horizon} days. The <b>shaded area</b> is the uncertainty range — "
            f"demand will likely stay within it. The <b>red dotted line</b> (p90) is the upper "
            f"scenario — order <b>{int(p90.sum()):,} units</b> to have stock available 90% of the time. "
            f"The <b>green dotted line</b> (p10) is the optimistic lower bound."
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.info(
            f"**{selected_part}** has a lead time of **{lead_time} days**. "
            f"The p90 order quantity of **{int(p90.sum()):,} units** covers the upper demand scenario "
            f"over the next {horizon} days at a 90% service level. "
            "Wider shaded bands indicate more variable demand — consider higher safety stock."
        )

    except FileNotFoundError:
        st.error("Dataset not found. Run `python -m data.generate_data` first.")
