import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";
import * as schema from "./schema";

// Neon's Vercel integration standardizes on POSTGRES_URL; accept either so the
// database "just works" once provisioned, without hand-editing env vars.
const url = process.env.DATABASE_URL || process.env.POSTGRES_URL;

// `db` is null when no connection string is set so local dev / build / tests
// without a database still run — logging becomes a no-op (surfaced, not swallowed).
export const db = url ? drizzle(neon(url), { schema }) : null;
