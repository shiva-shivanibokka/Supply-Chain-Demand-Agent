"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  isToolUIPart,
  getToolName,
  type UIMessage,
} from "ai";
import { DEFAULT_PROVIDER, PROVIDERS, type ProviderName } from "@/lib/providers";
import { ProviderBar } from "@/components/provider-bar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const TOOL_LABELS: Record<string, string> = {
  get_inventory_status: "Checking inventory",
  get_demand_forecast: "Forecasting demand",
  search_knowledge_base: "Searching knowledge base",
};

const toolLabel = (name: string) => TOOL_LABELS[name] ?? name;

type Part = UIMessage["parts"][number];

function ToolStep({ part }: { part: Part }) {
  if (!isToolUIPart(part)) return null;
  const label = toolLabel(getToolName(part));
  if (part.state === "output-error") {
    return <div className="text-xs text-destructive">⚠️ {label} failed: {part.errorText}</div>;
  }
  if (part.state === "output-available") {
    return <div className="text-xs text-muted-foreground">✅ {label}</div>;
  }
  return <div className="text-xs text-muted-foreground animate-pulse">🔧 {label}…</div>;
}

function ReasoningTrace({ parts }: { parts: Part[] }) {
  const steps = parts.filter(
    (p) => isToolUIPart(p) || p.type === "reasoning",
  );
  if (steps.length === 0) return null;
  return (
    <details open className="rounded-lg border border-border bg-background/40 px-3 py-2">
      <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Agent reasoning
      </summary>
      <div className="mt-2 space-y-1.5">
        {steps.map((part, i) =>
          part.type === "reasoning" ? (
            <p key={i} className="text-xs italic text-muted-foreground">
              {part.text}
            </p>
          ) : (
            <ToolStep key={i} part={part} />
          ),
        )}
      </div>
    </details>
  );
}

function MessageBubble({ message }: { message: UIMessage }) {
  const text = message.parts
    .filter((p): p is Extract<Part, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");

  if (message.role === "user") {
    return (
      <div className="ml-auto max-w-[80%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
        {text}
      </div>
    );
  }

  return (
    <div className="mr-auto max-w-[85%] space-y-2">
      <ReasoningTrace parts={message.parts} />
      {text && (
        <div className="rounded-lg bg-muted px-3 py-2 text-sm whitespace-pre-wrap">{text}</div>
      )}
    </div>
  );
}

export function Assistant() {
  const [provider, setProvider] = useState<ProviderName>(DEFAULT_PROVIDER);
  const [model, setModel] = useState(PROVIDERS[DEFAULT_PROVIDER].models[0]);
  const [apiKey, setApiKey] = useState("");
  const [input, setInput] = useState("");
  const { messages, sendMessage, status, error, setMessages } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
      body: { provider, model, apiKey },
    }),
  });

  const busy = status === "submitted" || status === "streaming";

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    sendMessage({ text: input });
    setInput("");
  }

  return (
    <div className="flex flex-col gap-3">
      <ProviderBar
        provider={provider}
        onProviderChange={setProvider}
        model={model}
        onModelChange={setModel}
        apiKey={apiKey}
        onApiKeyChange={setApiKey}
      />
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Ask about at-risk parts, demand forecasts, or supplier policy. The agent shows its
          reasoning as it calls tools.
        </p>
        <Button variant="ghost" size="sm" onClick={() => setMessages([])} disabled={messages.length === 0}>
          Clear conversation
        </Button>
      </div>

      <div className="panel-edge h-[55vh] min-h-[320px] space-y-3 overflow-y-auto rounded-xl border border-border bg-card/30 p-3">
        {messages.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No messages yet — try &ldquo;Which parts are most at risk, and what should I reorder?&rdquo;
          </p>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {status === "submitted" && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="inline-block size-2 animate-pulse rounded-full bg-[#6366f1]" />
            Agent is reasoning…
          </div>
        )}
        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error.message || "Something went wrong. Check your API key and model, then try again."}
          </div>
        )}
      </div>

      {apiKey ? (
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask the supply chain assistant…"
            disabled={busy}
          />
          <Button type="submit" disabled={busy || !input.trim()}>
            {busy ? "Working…" : "Send"}
          </Button>
        </form>
      ) : (
        <p className="rounded-lg border border-dashed border-border p-3 text-center text-sm text-muted-foreground">
          Paste your API key above to chat — Groq offers a free key with no card required.
        </p>
      )}
    </div>
  );
}
