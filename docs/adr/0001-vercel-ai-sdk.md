# 0001 — Use the Vercel AI SDK instead of a hand-rolled multi-provider loop

## Context

The chat assistant has to support four LLM providers (Anthropic, OpenAI, Groq, Google) selected per-request via BYOK, stream tokens to the browser, and let the model call tools (inventory, forecast, knowledge search) mid-conversation. Each provider's native SDK has its own streaming format, tool-call schema, and message shape. A hand-rolled loop would mean four parsers, four tool-call adapters, and four sets of edge cases to keep in sync.

## Decision

Use the Vercel AI SDK (`ai` + `@ai-sdk/*` provider packages) as the single interface between the chat route and all four providers. Tools are defined once with Zod schemas and work identically regardless of which provider is active; the SDK handles streaming, tool-call parsing, and message normalization.

## Consequences

One code path serves all four providers, so adding a fifth provider is a package install plus a config entry, not a new adapter. The SDK's version and provider APIs move fast, so provider/model IDs need periodic verification against each vendor's current lineup. In exchange, the app gets native streaming and tool-calling for free instead of maintained in-house.
