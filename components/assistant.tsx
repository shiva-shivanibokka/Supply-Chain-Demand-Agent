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

function toolLabel(name: string) {
  return TOOL_LABELS[name] ?? name;
}

function ToolStep({ part }: { part: UIMessage["parts"][number] }) {
  if (!isToolUIPart(part)) return null;
  const name = getToolName(part);
  const label = toolLabel(name);

  if (part.state === "output-error") {
    return <div className="text-xs text-destructive">⚠️ {label} failed: {part.errorText}</div>;
  }
  if (part.state === "output-available") {
    return <div className="text-xs text-muted-foreground">✅ {label}</div>;
  }
  return <div className="text-xs text-muted-foreground animate-pulse">🔧 {label}…</div>;
}

function MessageBubble({ message }: { message: UIMessage }) {
  if (message.role === "user") {
    const text = message.parts
      .filter((p): p is Extract<UIMessage["parts"][number], { type: "text" }> => p.type === "text")
      .map((p) => p.text)
      .join("");
    return (
      <div className="ml-auto max-w-[80%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
        {text}
      </div>
    );
  }

  return (
    <div className="mr-auto max-w-[80%] space-y-1.5 rounded-lg bg-muted px-3 py-2 text-sm">
      {message.parts.map((part, i) => {
        if (isToolUIPart(part)) return <ToolStep key={i} part={part} />;
        if (part.type === "text" && part.text) return <p key={i}>{part.text}</p>;
        return null;
      })}
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
    <div className="flex h-full flex-col gap-3">
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
          Ask about at-risk parts, demand forecasts, or supplier policy.
        </p>
        <Button variant="ghost" size="sm" onClick={() => setMessages([])} disabled={messages.length === 0}>
          Clear conversation
        </Button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-border p-3">
        {messages.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No messages yet — try &ldquo;Which parts are most at risk?&rdquo;
          </p>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {error && <p className="text-sm text-destructive">{error.message}</p>}
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
            Send
          </Button>
        </form>
      ) : (
        <p className="rounded-lg border border-dashed border-border p-3 text-center text-sm text-muted-foreground">
          Paste your API key above to chat.
        </p>
      )}
    </div>
  );
}
