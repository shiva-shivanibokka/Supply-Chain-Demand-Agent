"""Scripted evaluation of the chat agent's tool loop with a LOCAL model (Ollama).

The deployed agent is app/api/chat/route.ts (Vercel AI SDK). This script is a
faithful Python PORT of that loop: same SYSTEM prompt, same 3 tool names,
descriptions and parameters, same step cap (6), and Python ports of
lib/tools/{inventory,forecast,knowledge}.ts reading the same lib/data/*.json.
It is NOT the TypeScript runtime itself (see RESULTS.md, threats to validity).

Ground truth: computed from lib/data/*.json by the same (ported) tool code, so
"answer accuracy" measures whether the model picks the right tool and relays
the right number - not whether the underlying forecast is accurate.
No LLM-generated labels are used.

Usage:
  python -m eval_sop.agent_eval --build-questions
  python -m eval_sop.agent_eval --model qwen2.5:7b-instruct-q8_0
"""
import argparse
import json
import math
import os
import random
import re
import time

from eval_sop.common import ROOT, RESULTS

LIB = os.path.join(ROOT, "lib", "data")
PARTS = json.load(open(os.path.join(LIB, "parts.json")))
FORECASTS = json.load(open(os.path.join(LIB, "forecasts.json")))
DOCS = json.load(open(os.path.join(LIB, "docs.json")))
BY_ID = {p["part_id"]: p for p in PARTS}
QPATH = os.path.join(ROOT, "eval_sop", "agent_questions.json")

SYSTEM = ("You are a supply chain intelligence assistant for a capital equipment manufacturing "
          "company. You help managers with parts inventory, demand forecasting, and supplier "
          "management. Always use the tools to get real data before answering — never guess "
          "numbers. Be specific, actionable, and concise.")


def js_round(x, nd=0):  # JS Math.round semantics (half up), not Python banker's rounding
    f = 10 ** nd
    return math.floor(x * f + 0.5) / f


def to_fixed1(x):
    return f"{js_round(x, 1):.1f}"


# ---- ports of lib/tools/*.ts -------------------------------------------------
def days_of_supply(p):
    return js_round(p["inventory"] / max(p["avg_daily_demand"], 0.1), 1)


def risk(p):
    d = days_of_supply(p)
    if d < p["lead_time_days"]:
        return "CRITICAL"
    if d < 2 * p["lead_time_days"]:
        return "WARNING"
    return "OK"


def fmt_num(x):  # JS template literal prints 12.0 as "12"
    return str(int(x)) if float(x).is_integer() else str(x)


def get_inventory_status(part_id=None, top_n=None):
    top_n = top_n if top_n and top_n > 0 else 10
    if part_id:
        p = BY_ID.get(part_id)
        if not p:
            return f"Part '{part_id}' not found in dataset."
        return (f"Part: {p['part_id']} | Category: {p['category']} | Supplier: {p['supplier']} | Region: {p['region']}\n"
                f"Inventory: {int(js_round(p['inventory']))} units | Avg daily demand: {to_fixed1(p['avg_daily_demand'])} units/day\n"
                f"Days of supply: {fmt_num(days_of_supply(p))} days | Lead time: {p['lead_time_days']} days | Risk: {risk(p)}")
    at = sorted([p for p in PARTS if risk(p) != "OK"], key=days_of_supply)[: int(top_n)]
    if not at:
        return "All parts have sufficient inventory levels."
    return f"Top {len(at)} at-risk parts:\n" + "\n".join(
        f"  [{risk(p)}] {p['part_id']} ({p['category']}, {p['supplier']}) - {fmt_num(days_of_supply(p))} days supply (lead time: {p['lead_time_days']} days)"
        for p in at)


def forecast_numbers(part_id):
    pre = FORECASTS.get(part_id)
    if pre and pre.get("p50"):
        d, src = pre, "TFT model"
    else:
        hist = [h["demand"] for h in BY_ID[part_id]["history"]][-60:]
        avg = sum(hist) / len(hist)
        sd = math.sqrt(sum((x - avg) ** 2 for x in hist) / len(hist)) if len(hist) > 1 else 0
        p50 = [max(avg + avg * 0.05 * i / 29, 0) for i in range(30)]
        d = {"p50": p50, "p10": [max(v - 1.65 * sd, 0) for v in p50], "p90": [v + 1.65 * sd for v in p50]}
        src = "statistical baseline"
    return {"p10": int(js_round(sum(d["p10"]))), "p50": int(js_round(sum(d["p50"]))),
            "p90": int(js_round(sum(d["p90"]))), "p50_daily": sum(d["p50"]) / len(d["p50"]), "source": src}


def get_demand_forecast(part_id):
    if part_id not in BY_ID:
        return f"No data found for part '{part_id}'."
    f = forecast_numbers(part_id)
    return (f"30-day demand forecast for {part_id} ({f['source']}):\n"
            f"  Daily demand (median): {f['p50_daily']:.1f} units/day\n"
            f"  Total 30-day demand: {f['p50']} units (p50)\n"
            f"  Lower bound (p10): {f['p10']} units\n"
            f"  Upper bound (p90): {f['p90']} units\n"
            f"  Recommendation: Order at least {f['p90']} units for 90% service level")


STOP = set(("a an the is are was were be been being have has had do does did will would could should may might to of in for on with at by from as into and but or not that this it its we i you he she they all any each some if about up so out what").split())


def tokenize(t):
    return [w for w in re.findall(r"[a-z0-9_]+", t.lower()) if w not in STOP and len(w) > 2]


def search_knowledge_base(query, top_k=3):
    q = set(tokenize(query))
    if not q:
        return "No relevant documents found in the knowledge base."
    scored = []
    for d in DOCS:
        words = tokenize(d["text"])
        ws = set(words)
        ov = [t for t in q if t in ws]
        if not ov:
            scored.append((d, 0.0))
            continue
        base = len(ov) / len(q)
        tf = sum(words.count(t) for t in ov)
        scored.append((d, base + min(tf / (len(words) + 1), 0.3)))
    mx = max(s for _, s in scored)
    top = sorted([(d, s / mx if mx > 0 else 0) for d, s in scored], key=lambda x: -x[1])
    top = [x for x in top if x[1] >= 0.05][:top_k]
    if not top:
        return "No relevant documents found in the knowledge base."
    return "\n\n".join(f"[Source: {d['id']} | Category: {d['category']} | Relevance: {s:.2f}]\n{d['text'].strip()}" for d, s in top)


TOOLS = [
    {"type": "function", "function": {
        "name": "get_inventory_status",
        "description": "Get current inventory levels, days of supply, and stockout risk. Use for questions about stock levels or which parts are running low.",
        "parameters": {"type": "object", "properties": {
            "part_id": {"type": "string", "description": "Specific part e.g. PART_007; omit for top at-risk parts"},
            "top_n": {"type": "number", "description": "How many at-risk parts (default 10)"}}}}},
    {"type": "function", "function": {
        "name": "get_demand_forecast",
        "description": "Get the 30-day demand forecast (p10/p50/p90) for a part. Use for future demand or order quantity.",
        "parameters": {"type": "object", "properties": {
            "part_id": {"type": "string", "description": "Part ID e.g. PART_007"}}, "required": ["part_id"]}}},
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search internal supply-chain docs for reorder policies, supplier reliability, safety-stock rules, and past incidents.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query, be specific"}}, "required": ["query"]}}},
]


def run_tool(name, args):
    if name == "get_inventory_status":
        tn = args.get("top_n")
        return get_inventory_status(args.get("part_id") or None, int(tn) if tn else None)
    if name == "get_demand_forecast":
        return get_demand_forecast(args.get("part_id", ""))
    if name == "search_knowledge_base":
        return search_knowledge_base(args.get("query", ""))
    return f"Unknown tool {name}"


# ---- question set ------------------------------------------------------------
def build_questions():
    rng = random.Random(2024)
    ids = sorted(BY_ID)
    by_risk = {r: [p for p in ids if risk(BY_ID[p]) == r] for r in ("CRITICAL", "WARNING", "OK")}
    qs = []
    for p in rng.sample(ids, 8):
        qs.append(dict(q=f"How many units of {p} do we currently have in stock?", tool="get_inventory_status",
                       part_id=p, kind="number", answer=int(js_round(BY_ID[p]["inventory"]))))
    for p in rng.sample(ids, 5):
        qs.append(dict(q=f"How many days of supply are left for {p}?", tool="get_inventory_status",
                       part_id=p, kind="number", answer=days_of_supply(BY_ID[p])))
    risk_parts = []
    for r, n in (("CRITICAL", 2), ("WARNING", 2), ("OK", 1)):
        risk_parts += rng.sample(by_risk[r], min(n, len(by_risk[r])))
    for p in risk_parts:
        qs.append(dict(q=f"What is the stockout risk level for {p}?", tool="get_inventory_status",
                       part_id=p, kind="label", answer=risk(BY_ID[p])))
    at = sorted([p for p in ids if risk(BY_ID[p]) != "OK"], key=lambda p: days_of_supply(BY_ID[p]))
    qs.append(dict(q="Which part currently has the lowest days of supply?", tool="get_inventory_status",
                   part_id=None, kind="ids", answer=[at[0]]))
    qs.append(dict(q="List the 3 most at-risk parts right now.", tool="get_inventory_status",
                   part_id=None, kind="ids", answer=at[:3]))
    for p in rng.sample(ids, 3):
        qs.append(dict(q=f"What is the total 30-day median (p50) demand forecast for {p}?", tool="get_demand_forecast",
                       part_id=p, kind="number", answer=forecast_numbers(p)["p50"]))
    for p in rng.sample(ids, 3):
        qs.append(dict(q=f"How many units of {p} should we order to cover a 90% service level over the next 30 days?",
                       tool="get_demand_forecast", part_id=p, kind="number", answer=forecast_numbers(p)["p90"]))
    # knowledge-base facts copied verbatim from lib/data/docs.json (human-written docs, not LLM labels)
    qs += [
        dict(q="What is SupplierA's on-time delivery rate?", tool="search_knowledge_base", part_id=None, kind="number", answer=96.2),
        dict(q="Per our safety stock policy, what Z value should be used for parts with a lead time over 20 days?",
             tool="search_knowledge_base", part_id=None, kind="number", answer=1.96),
        dict(q="How many days of production downtime did the Q3 2022 PART_007 stockout cause?", tool="search_knowledge_base",
             part_id=None, kind="number", answer=4),
        dict(q="What is SupplierD's quality rating from the 2023 audit?", tool="search_knowledge_base",
             part_id=None, kind="number", answer=3.9),
    ]
    for i, q in enumerate(qs):
        q["id"] = i + 1
    with open(QPATH, "w") as f:
        json.dump(qs, f, indent=1)
    print(f"wrote {len(qs)} questions to {QPATH}")
    return qs


def nums_in(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def grade(q, final):
    if q["kind"] == "number":
        tol = 0.05 if not float(q["answer"]).is_integer() else 0.5
        return any(abs(v - q["answer"]) <= tol for v in nums_in(final))
    if q["kind"] == "label":
        found = {l for l in ("CRITICAL", "WARNING", "OK") if re.search(rf"\b{l}\b", final, re.I)}
        return q["answer"] in found and len(found) == 1
    if q["kind"] == "ids":
        return all(i in final for i in q["answer"])
    raise ValueError(q["kind"])


def run_agent(client, model, question, max_steps=6):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls = []
    for _ in range(max_steps):
        r = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS,
                                           temperature=0, seed=0)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or "", calls
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = run_tool(tc.function.name, args)
            calls.append({"name": tc.function.name, "args": args})
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "", calls  # step cap hit without a final answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-questions", action="store_true")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q8_0")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()
    if args.build_questions:
        build_questions()
        return
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="ollama")  # local Ollama, no key
    qs = json.load(open(QPATH))
    recs = []
    for q in qs:
        t0 = time.time()
        final, calls = run_agent(client, args.model, q["q"])
        names = [c["name"] for c in calls]
        tool_ok = q["tool"] in names
        if tool_ok and q["part_id"]:
            tool_ok = any(c["name"] == q["tool"] and c["args"].get("part_id") == q["part_id"] for c in calls)
        recs.append(dict(id=q["id"], question=q["q"], expected_tool=q["tool"], expected_answer=q["answer"],
                         tool_calls=calls, first_tool=names[0] if names else None,
                         tool_correct=bool(tool_ok), answer_correct=bool(grade(q, final)),
                         final_answer=final, seconds=round(time.time() - t0, 1)))
        print(q["id"], recs[-1]["tool_correct"], recs[-1]["answer_correct"], flush=True)
    n = len(recs)
    summ = dict(model=args.model, runtime="ollama (local)", n_questions=n,
                tool_selection_accuracy=sum(r["tool_correct"] for r in recs) / n,
                first_tool_correct=sum(r["first_tool"] == r["expected_tool"] for r in recs) / n,
                answer_accuracy=sum(r["answer_correct"] for r in recs) / n,
                no_tool_call=sum(1 for r in recs if not r["tool_calls"]),
                by_tool={t: dict(n=sum(r["expected_tool"] == t for r in recs),
                                 tool_acc=sum(r["tool_correct"] for r in recs if r["expected_tool"] == t),
                                 ans_acc=sum(r["answer_correct"] for r in recs if r["expected_tool"] == t))
                         for t in ("get_inventory_status", "get_demand_forecast", "search_knowledge_base")})
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    with open(os.path.join(RESULTS, f"agent_eval_{tag}.json"), "w") as f:
        json.dump(dict(summary=summ, records=recs), f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
