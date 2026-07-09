import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Non-JS project dirs (Python stack, virtualenv, generated artifacts) —
    // not part of the Next.js app and not gitignored consistently enough to
    // rely on ESLint's defaults alone.
    "venv/**",
    "__pycache__/**",
    "mlruns/**",
    "lightning_logs/**",
    "forecasting/saved_model/**",
    "rag/chroma_db/**",
    "notebooks/**",
  ]),
]);

export default eslintConfig;
