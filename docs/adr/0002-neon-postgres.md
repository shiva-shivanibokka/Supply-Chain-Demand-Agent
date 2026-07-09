# 0002 — Use Neon Postgres for prediction logging instead of disk or live-only drift

## Context

The MLOps tab needs a prediction log (what did the model forecast, and when) to compute drift metrics later. The app is deployed on Vercel, where serverless functions have no persistent local disk between invocations, so writing to a file doesn't survive across requests. A traditional Postgres connection pool is also risky here: each serverless invocation can open a new connection, and a modest-traffic app can exhaust a small Postgres instance's connection limit in minutes.

## Decision

Use Neon Postgres via its HTTP driver (`@neondatabase/serverless` + Drizzle) for the prediction log. Neon is provisioned through the Vercel Marketplace integration, which sets `DATABASE_URL` automatically, and is free at this project's scale.

## Consequences

Predictions persist across requests and deployments, giving the drift-detection feature something real to compute against instead of only in-memory state. The HTTP driver avoids pool exhaustion since each query is a stateless HTTP call rather than a held TCP connection. `DATABASE_URL` is optional for local dev — `db` is `null` when unset, and logging becomes a no-op — so the app still runs without a database configured.
