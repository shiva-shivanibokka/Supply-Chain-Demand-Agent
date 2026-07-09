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
    // Groq deprecated the llama-3.x IDs on 2026-06-17; GPT-OSS are the replacements.
    models: ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    keyPlaceholder: "gsk_...",
    free: true,
  },
  Google: {
    label: "Google Gemini",
    models: ["gemini-2.5-flash", "gemini-2.0-flash"],
    keyPlaceholder: "AIza...",
    free: false,
  },
};

export const DEFAULT_PROVIDER: ProviderName = "Groq";
