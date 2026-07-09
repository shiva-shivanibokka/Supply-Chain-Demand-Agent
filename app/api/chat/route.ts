import { streamText, tool, stepCountIs, convertToModelMessages, type UIMessage } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { z } from "zod";
import { getInventoryStatus } from "@/lib/tools/inventory";
import { getForecast } from "@/lib/tools/forecast";
import { searchKnowledge } from "@/lib/tools/knowledge";
import { logPrediction } from "@/lib/db/log";
import { PROVIDERS, type ProviderName } from "@/lib/providers";

export const maxDuration = 30;

const SYSTEM = `You are a supply chain intelligence assistant for a capital equipment manufacturing company. You help managers with parts inventory, demand forecasting, and supplier management. Always use the tools to get real data before answering — never guess numbers. Be specific, actionable, and concise.`;

function buildModel(provider: ProviderName, model: string, apiKey: string) {
  switch (provider) {
    case "Anthropic":
      return createAnthropic({ apiKey })(model);
    case "OpenAI":
      return createOpenAI({ apiKey })(model);
    case "Groq":
      return createOpenAI({ apiKey, baseURL: "https://api.groq.com/openai/v1" })(model);
    case "Google":
      return createGoogleGenerativeAI({ apiKey })(model);
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
}

export async function POST(req: Request) {
  let body: { provider: ProviderName; model: string; apiKey: string; messages: UIMessage[] };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const { provider, model, apiKey, messages } = body;

  if (!provider || !(provider in PROVIDERS)) {
    return Response.json({ error: "Unknown or missing provider" }, { status: 400 });
  }
  if (!model || typeof model !== "string") {
    return Response.json({ error: "Missing model" }, { status: 400 });
  }
  if (!Array.isArray(messages) || messages.length === 0) {
    return Response.json({ error: "Missing or empty messages" }, { status: 400 });
  }
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "Missing API key" }), { status: 400 });
  }

  const result = streamText({
    model: buildModel(provider, model, apiKey),
    system: SYSTEM,
    messages: await convertToModelMessages(messages),
    // Keep looping so the agent reads tool results and writes a final answer,
    // instead of stopping after the first tool call.
    stopWhen: stepCountIs(6),
    tools: {
      get_inventory_status: tool({
        description:
          "Get current inventory levels, days of supply, and stockout risk. Use for questions about stock levels or which parts are running low.",
        inputSchema: z.object({
          part_id: z.string().optional().describe("Specific part e.g. PART_007; omit for top at-risk parts"),
          top_n: z.number().optional().describe("How many at-risk parts (default 10)"),
        }),
        execute: async ({ part_id, top_n }) => getInventoryStatus({ partId: part_id, topN: top_n }),
      }),
      get_demand_forecast: tool({
        description: "Get the 30-day demand forecast (p10/p50/p90) for a part. Use for future demand or order quantity.",
        inputSchema: z.object({ part_id: z.string().describe("Part ID e.g. PART_007") }),
        execute: async ({ part_id }) => {
          const f = getForecast(part_id);
          if (f.source !== "none") {
            await logPrediction({
              partId: part_id, source: f.source, p50Daily: f.p50Daily,
              p50Total: f.p50, p10Total: f.p10, p90Total: f.p90, horizonDays: 30,
            });
          }
          return f.text;
        },
      }),
      search_knowledge_base: tool({
        description:
          "Search internal supply-chain docs for reorder policies, supplier reliability, safety-stock rules, and past incidents.",
        inputSchema: z.object({ query: z.string().describe("Search query, be specific") }),
        execute: async ({ query }) => searchKnowledge(query),
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
