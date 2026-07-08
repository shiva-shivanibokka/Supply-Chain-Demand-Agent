import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";
import * as schema from "./schema";

const url = process.env.DATABASE_URL;

// `db` is null when DATABASE_URL is unset so local dev / build / tests without
// a database still run — logging becomes a no-op (surfaced, not swallowed).
export const db = url ? drizzle(neon(url), { schema }) : null;
