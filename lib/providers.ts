export type ProviderName = "Anthropic" | "OpenAI" | "Groq" | "Google";

export const PROVIDERS: Record<
  ProviderName,
  { label: string; models: string[]; keyPlaceholder: string; free: boolean }
> = {
  Anthropic: {
    label: "Anthropic",
    models: ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"],
    keyPlaceholder: "sk-ant-...",
    free: false,
  },
  OpenAI: {
    label: "OpenAI",
    models: ["gpt-4o", "gpt-4o-mini"], // verify current IDs at impl time
    keyPlaceholder: "sk-...",
    free: false,
  },
  Groq: {
    label: "Groq (free)",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], // verify current IDs
    keyPlaceholder: "gsk_...",
    free: true,
  },
  Google: {
    label: "Google Gemini",
    models: ["gemini-2.0-flash", "gemini-1.5-pro"], // verify current IDs
    keyPlaceholder: "AIza...",
    free: false,
  },
};

export const DEFAULT_PROVIDER: ProviderName = "Groq";
