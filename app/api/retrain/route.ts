// Dispatches the "Retrain TFT" GitHub Actions workflow. The heavy lifting
// (PyTorch training) runs on GitHub's runners, not here — Vercel functions have
// no GPU and can't run PyTorch. Requires a GITHUB_DISPATCH_TOKEN (a fine-grained
// PAT with Actions: read/write on this repo) set in the Vercel project.
export const maxDuration = 15;

const REPO = "shiva-shivanibokka/Supply-Chain-Demand-Agent";
const WORKFLOW = "retrain.yml";
const COOLDOWN_MS = 10 * 60 * 1000;

// ponytail: best-effort, per serverless instance only — resets on cold start
// and isn't shared across instances. A real rate limit needs shared storage.
let lastDispatch = 0;

export async function POST() {
  const token = process.env.GITHUB_DISPATCH_TOKEN;
  if (!token) {
    return Response.json(
      {
        ok: false,
        error:
          "Retraining button isn't wired up yet — add GITHUB_DISPATCH_TOKEN in the Vercel project to enable it. The pipeline still runs from the repo's GitHub Actions tab.",
      },
      { status: 501 },
    );
  }

  if (Date.now() - lastDispatch < COOLDOWN_MS) {
    return Response.json(
      { ok: false, error: "Retraining was triggered recently — wait a few minutes before retrying." },
      { status: 429 },
    );
  }

  let res: Response;
  try {
    res = await fetch(
      `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
        },
        body: JSON.stringify({ ref: "main" }),
      },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return Response.json({ ok: false, error: `Failed to reach GitHub: ${msg}` }, { status: 502 });
  }

  if (res.status === 204) {
    lastDispatch = Date.now();
    return Response.json({
      ok: true,
      message:
        "Retraining dispatched. It runs on GitHub Actions (CPU) and auto-commits refreshed forecasts to main, which redeploys, when done (~20–40 min).",
    });
  }

  const text = await res.text();
  return Response.json(
    { ok: false, error: `GitHub dispatch failed (${res.status}): ${text.slice(0, 200)}` },
    { status: 502 },
  );
}
