// Dispatches the "Retrain TFT" GitHub Actions workflow. The heavy lifting
// (PyTorch training) runs on GitHub's runners, not here — Vercel functions have
// no GPU and can't run PyTorch. Requires a GITHUB_DISPATCH_TOKEN (a fine-grained
// PAT with Actions: read/write on this repo) set in the Vercel project.
export const maxDuration = 15;

const REPO = "shiva-shivanibokka/Supply-Chain-Demand-Agent";
const WORKFLOW = "retrain.yml";

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

  const res = await fetch(
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

  if (res.status === 204) {
    return Response.json({
      ok: true,
      message:
        "Retraining dispatched. It runs on GitHub Actions (CPU) and opens a PR with refreshed forecasts when done.",
    });
  }

  const text = await res.text();
  return Response.json(
    { ok: false, error: `GitHub dispatch failed (${res.status}): ${text.slice(0, 200)}` },
    { status: 502 },
  );
}
