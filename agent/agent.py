"""
agent/agent.py
--------------
This file defines the AI agent - the brain of the entire project.

What is an agent?
  A regular LLM takes a question and immediately produces an answer
  based only on what it was trained on.

  An AGENT is different. Before answering, it can:
    1. Decide WHICH tools it needs to answer the question
    2. CALL those tools to get real data
    3. Look at the tool results
    4. Decide if it needs more information (call another tool)
    5. THEN produce a final answer grounded in actual data

  This is called a ReAct loop: Reason → Act → Observe → Reason → ...

Multi-provider support:
  This project was originally built with Anthropic Claude.
  It now supports multiple LLM providers so users can plug in
  whichever API key they have:

  Provider   | Models                          | Key prefix
  -----------|---------------------------------|------------
  Anthropic  | claude-opus-4-5, claude-sonnet  | sk-ant-
  OpenAI     | gpt-4o, gpt-4o-mini, gpt-4-turbo| sk-
  Groq       | llama3-70b, mixtral-8x7b (free) | gsk_

  Each provider has a different SDK and a different way of handling
  tool calls. We handle all three with a unified interface so the
  rest of the code doesn't need to know which provider is in use.

Our agent has 3 tools:
  1. search_knowledge_base  - searches the RAG vector DB
  2. get_inventory_status   - reads the CSV, computes stockout risk
  3. get_demand_forecast    - runs the TFT model forecast
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Generator

from rag.retriever import retrieve, format_context


# ---------------------------------------------------------------------------
# PROVIDER CONFIGURATION
# Maps provider names to their default models and env variable names
# ---------------------------------------------------------------------------

PROVIDERS = {
    "Anthropic": {
        "models": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-3-5"],
        "env_key": "ANTHROPIC_API_KEY",
        "key_hint": "Starts with sk-ant-",
        "free": False,
        "docs_url": "https://console.anthropic.com/",
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "env_key": "OPENAI_API_KEY",
        "key_hint": "Starts with sk-",
        "free": False,
        "docs_url": "https://platform.openai.com/api-keys",
    },
    "Groq (Free)": {
        "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        "env_key": "GROQ_API_KEY",
        "key_hint": "Starts with gsk_",
        "free": True,
        "docs_url": "https://console.groq.com/keys",
    },
}

DEFAULT_PROVIDER = "Anthropic"
DEFAULT_MODEL = "claude-opus-4-5"


# ---------------------------------------------------------------------------
# TOOL DEFINITIONS (OpenAI-compatible format)
# Both OpenAI and Groq use the same JSON schema.
# Anthropic uses a slightly different format - we convert below.
# ---------------------------------------------------------------------------

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the internal supply chain knowledge base for relevant policies, "
                "supplier information, incident reports, and operational guidelines. "
                "Use this when you need to answer questions about reorder policies, "
                "supplier reliability, safety stock rules, or past incidents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_status",
            "description": (
                "Get current inventory levels, days of supply remaining, and stockout risk. "
                "Returns the parts most at risk of stockout. Use this when asked about "
                "current stock levels, which parts are running low, or which need attention."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "Optional. Specific part ID to check (e.g. 'PART_007'). "
                        "If not provided, returns the top at-risk parts.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many at-risk parts to return. Default is 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demand_forecast",
            "description": (
                "Get the 30-day demand forecast for a specific part. Returns predicted "
                "demand with low (p10), median (p50), and high (p90) estimates. "
                "Use this when asked about future demand or how much stock to order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "The part ID to forecast (e.g. 'PART_007').",
                    },
                },
                "required": ["part_id"],
            },
        },
    },
]

# Anthropic uses a slightly different schema - no "type": "function" wrapper
TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------


def search_knowledge_base(query: str) -> str:
    docs = retrieve(query, top_k=3)
    return format_context(docs)


def get_inventory_status(
    part_id: Optional[str] = None,
    top_n: int = 10,
    data_path: str = "data/supply_chain_data.csv",
) -> str:
    df = pd.read_csv(data_path, parse_dates=["date"])
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

    if part_id:
        row = latest[latest["part_id"] == part_id]
        if row.empty:
            return f"Part '{part_id}' not found in dataset."
        row = row.iloc[0]
        return (
            f"Part: {row['part_id']} | Category: {row['category']} | "
            f"Supplier: {row['supplier']} | Region: {row['region']}\n"
            f"Inventory: {int(row['inventory'])} units | "
            f"Avg daily demand: {row['avg_daily_demand']:.1f} units/day\n"
            f"Days of supply: {row['days_of_supply']} days | "
            f"Lead time: {int(row['lead_time_days'])} days | "
            f"Risk: {row['risk']}"
        )

    at_risk = latest[latest["risk"].isin(["CRITICAL", "WARNING"])]
    at_risk = at_risk.sort_values("days_of_supply").head(top_n)

    if at_risk.empty:
        return "All parts have sufficient inventory levels."

    lines = [f"Top {len(at_risk)} at-risk parts as of {latest_date.date()}:\n"]
    for _, row in at_risk.iterrows():
        lines.append(
            f"  [{row['risk']}] {row['part_id']} ({row['category']}, {row['supplier']}) "
            f"- {row['days_of_supply']} days supply remaining "
            f"(lead time: {int(row['lead_time_days'])} days)"
        )
    return "\n".join(lines)


def get_demand_forecast(
    part_id: str,
    data_path: str = "data/supply_chain_data.csv",
    model_dir: str = "forecasting/saved_model",
) -> str:
    df = pd.read_csv(data_path, parse_dates=["date"])
    part_data = df[df["part_id"] == part_id].sort_values("date")

    if part_data.empty:
        return f"No data found for part '{part_id}'."

    import glob

    checkpoint_files = glob.glob(f"{model_dir}/*.ckpt")
    source = "statistical baseline"
    if checkpoint_files:
        try:
            result = _forecast_with_tft(part_id, part_data, checkpoint_files[0])
            source = "TFT model"
            _log_forecast_to_mlflow(part_id, result, source)
            return result
        except Exception:
            pass

    result = _forecast_statistical(part_id, part_data)
    _log_forecast_to_mlflow(part_id, result, source)
    return result


def _log_forecast_to_mlflow(part_id: str, forecast_text: str, source: str) -> None:
    """
    Parses the forecast result string and logs it to the prediction log.
    Runs silently — a logging failure should never break the agent.
    """
    try:
        from mlops.monitor import log_prediction

        lines = {
            line.split(":")[0].strip(): line.split(":")[1].strip()
            for line in forecast_text.split("\n")
            if ":" in line
        }
        p50_daily = float(lines.get("Daily demand (median)", "0").split()[0])
        p50_total = float(
            lines.get("Total 30-day demand", "0").split()[0].replace(",", "")
        )
        p10_total = float(
            lines.get("Lower bound (p10)", "0").split()[0].replace(",", "")
        )
        p90_total = float(
            lines.get("Upper bound (p90)", "0").split()[0].replace(",", "")
        )
        log_prediction(
            part_id=part_id,
            p10_total=p10_total,
            p50_total=p50_total,
            p90_total=p90_total,
            p50_daily=p50_daily,
            horizon_days=30,
            source=source,
        )
    except Exception:
        pass  # never let logging crash the agent


def _forecast_with_tft(part_id: str, part_data: pd.DataFrame, ckpt_path: str) -> str:
    from pytorch_forecasting import TemporalFusionTransformer
    from forecasting.model import load_and_prepare, build_dataset, DECODER_LENGTH
    import torch
    from torch.utils.data import DataLoader

    full_df = load_and_prepare("data/supply_chain_data.csv")
    training_ds, _ = build_dataset(full_df)
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()

    part_df = full_df[full_df["part_id"] == part_id]
    from pytorch_forecasting import TimeSeriesDataSet

    pred_ds = TimeSeriesDataSet.from_dataset(training_ds, part_df, predict=True)
    loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        preds = model.predict(loader, mode="quantiles", return_y=False)

    p10 = preds[:, :, 0].cpu().numpy().flatten()[:DECODER_LENGTH]
    p50 = preds[:, :, 1].cpu().numpy().flatten()[:DECODER_LENGTH]
    p90 = preds[:, :, 2].cpu().numpy().flatten()[:DECODER_LENGTH]
    return _format_forecast(part_id, p10, p50, p90, source="TFT model")


def _forecast_statistical(part_id: str, part_data: pd.DataFrame) -> str:
    recent = part_data["demand"].tail(60).values
    avg = recent.mean()
    std = recent.std()
    horizon = 30
    trend = np.linspace(0, avg * 0.05, horizon)
    p50 = np.maximum(avg + trend, 0)
    p10 = np.maximum(p50 - 1.65 * std, 0)
    p90 = p50 + 1.65 * std
    return _format_forecast(part_id, p10, p50, p90, source="statistical baseline")


def _format_forecast(part_id, p10, p50, p90, source):
    return (
        f"30-day demand forecast for {part_id} ({source}):\n"
        f"  Daily demand (median): {p50.mean():.1f} units/day\n"
        f"  Total 30-day demand:   {int(p50.sum())} units (p50)\n"
        f"  Lower bound (p10):     {int(p10.sum())} units\n"
        f"  Upper bound (p90):     {int(p90.sum())} units\n"
        f"  Recommendation:        Order at least {int(p90.sum())} units for 90% service level"
    )


def run_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    if tool_name == "search_knowledge_base":
        return search_knowledge_base(tool_input["query"])
    elif tool_name == "get_inventory_status":
        return get_inventory_status(
            part_id=tool_input.get("part_id"),
            top_n=tool_input.get("top_n", 10),
        )
    elif tool_name == "get_demand_forecast":
        return get_demand_forecast(tool_input["part_id"])
    else:
        return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# PROVIDER BACKENDS
# Each backend handles one provider's specific API format for tool use
# ---------------------------------------------------------------------------


def _call_anthropic(messages: list, model: str, api_key: str) -> dict:
    """
    Calls the Anthropic API and returns a normalized response dict:
      {
        "stop_reason": "tool_use" | "end_turn",
        "text"       : str,            # final answer text if end_turn
        "tool_calls" : [               # list if tool_use
            {"id": str, "name": str, "input": dict}
        ],
        "raw"        : original response object
      }
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS_ANTHROPIC,
        messages=messages,
    )

    if response.stop_reason == "end_turn":
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        return {
            "stop_reason": "end_turn",
            "text": text,
            "tool_calls": [],
            "raw": response,
        }

    tool_calls = [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in response.content
        if b.type == "tool_use"
    ]
    return {
        "stop_reason": "tool_use",
        "text": "",
        "tool_calls": tool_calls,
        "raw": response,
    }


def _call_openai_compatible(
    messages: list, model: str, api_key: str, base_url: str = None
) -> dict:
    """
    Calls OpenAI-compatible APIs (OpenAI and Groq share the same SDK format).
    Returns the same normalized dict as _call_anthropic.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=messages,
        tools=TOOLS_OPENAI,
        tool_choice="auto",
    )

    choice = response.choices[0]

    if choice.finish_reason == "stop" or choice.finish_reason == "end_turn":
        text = choice.message.content or ""
        return {
            "stop_reason": "end_turn",
            "text": text,
            "tool_calls": [],
            "raw": response,
        }

    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
            )
    return {
        "stop_reason": "tool_use",
        "text": "",
        "tool_calls": tool_calls,
        "raw": response,
    }


def _build_client(provider: str, api_key: str):
    """Returns a callable that matches the signature (messages, model, api_key) -> normalized dict."""
    if provider == "Anthropic":
        return _call_anthropic
    elif provider == "OpenAI":
        return lambda msgs, model, key: _call_openai_compatible(msgs, model, key)
    elif provider == "Groq (Free)":
        return lambda msgs, model, key: _call_openai_compatible(
            msgs, model, key, base_url="https://api.groq.com/openai/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _append_assistant_turn(messages: list, result: dict, provider: str) -> None:
    """
    Appends the assistant's response to the message history in the correct
    format for each provider. Anthropic and OpenAI use different message shapes.
    """
    if provider == "Anthropic":
        messages.append({"role": "assistant", "content": result["raw"].content})
    else:
        # OpenAI / Groq
        msg = result["raw"].choices[0].message
        messages.append(
            {"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls}
        )


def _append_tool_results(
    messages: list, tool_calls: list, results: list, provider: str
) -> None:
    """
    Appends tool results back to the message history.
    Anthropic batches all tool results in one user message.
    OpenAI/Groq sends one tool message per result.
    """
    if provider == "Anthropic":
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tc["id"], "content": res}
                    for tc, res in zip(tool_calls, results)
                ],
            }
        )
    else:
        for tc, res in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": res,
                }
            )


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a supply chain intelligence assistant for a capital equipment
manufacturing company. You help supply chain managers make decisions about parts inventory,
demand forecasting, and supplier management.

You have access to three tools:
- search_knowledge_base: find relevant policies, supplier profiles, and incident reports
- get_inventory_status: check current stock levels and identify at-risk parts
- get_demand_forecast: get 30-day demand forecasts for specific parts

Always use the tools to get real data before answering. Do not guess or make up numbers.
Be specific and actionable. When a part is at risk, explain exactly why and what to do.
Keep answers concise and structured - supply chain managers are busy people."""

TOOL_LABELS = {
    "get_inventory_status": "Checking inventory levels",
    "get_demand_forecast": "Running demand forecast",
    "search_knowledge_base": "Searching knowledge base",
}


# ---------------------------------------------------------------------------
# AGENT LOOPS
# ---------------------------------------------------------------------------


def run_agent(
    user_message: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    max_iterations: int = 5,
) -> str:
    """
    Runs the full agentic ReAct loop and returns the final answer as a string.
    Works with any supported provider.
    """
    if not api_key:
        api_key = os.environ.get(PROVIDERS[provider]["env_key"], "")

    call_llm = _build_client(provider, api_key)
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        result = call_llm(messages, model, api_key)

        if result["stop_reason"] == "end_turn":
            return result["text"]

        if result["stop_reason"] == "tool_use":
            _append_assistant_turn(messages, result, provider)
            tool_results = [
                run_tool(tc["name"], tc["input"]) for tc in result["tool_calls"]
            ]
            _append_tool_results(messages, result["tool_calls"], tool_results, provider)

    return "Unable to complete the analysis. Please try a more specific question."


def run_agent_with_steps(
    user_message: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    max_iterations: int = 5,
) -> Generator:
    """
    Same as run_agent() but yields reasoning steps live so the UI
    can display them as they happen.

    Yields dicts:
      {"type": "tool_start",  "tool": name, "label": human_label}
      {"type": "tool_result", "tool": name, "preview": first_line}
      {"type": "answer",      "text": final_answer}
      {"type": "error",       "text": error_message}
    """
    if not api_key:
        api_key = os.environ.get(PROVIDERS[provider]["env_key"], "")

    call_llm = _build_client(provider, api_key)
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        result = call_llm(messages, model, api_key)

        if result["stop_reason"] == "end_turn":
            yield {"type": "answer", "text": result["text"]}
            return

        if result["stop_reason"] == "tool_use":
            _append_assistant_turn(messages, result, provider)

            tool_results = []
            for tc in result["tool_calls"]:
                label = TOOL_LABELS.get(tc["name"], tc["name"])
                yield {"type": "tool_start", "tool": tc["name"], "label": label}

                res = run_tool(tc["name"], tc["input"])
                preview = res.strip().split("\n")[0][:120]
                yield {"type": "tool_result", "tool": tc["name"], "preview": preview}
                tool_results.append(res)

            _append_tool_results(messages, result["tool_calls"], tool_results, provider)

    yield {
        "type": "error",
        "text": "Unable to complete analysis. Try a more specific question.",
    }


if __name__ == "__main__":
    import sys

    provider = "Anthropic"
    model = "claude-opus-4-5"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to test.")
        sys.exit(1)
    q = "Which parts are most at risk right now?"
    print(f"Question: {q}\n")
    print(run_agent(q, provider=provider, model=model, api_key=api_key))
