import { getForecast } from "@/lib/tools/forecast";
import { logPrediction } from "@/lib/db/log";

export async function POST(req: Request) {
  let body: { partId?: string };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const { partId } = body;

  if (!partId) {
    return new Response(JSON.stringify({ error: "Missing partId" }), { status: 400 });
  }

  const f = getForecast(partId);
  if (f.source === "none") {
    return new Response(JSON.stringify({ error: `Unknown part '${partId}'` }), { status: 400 });
  }

  await logPrediction({
    partId, source: f.source, p50Daily: f.p50Daily,
    p50Total: f.p50, p10Total: f.p10, p90Total: f.p90, horizonDays: 30,
  });

  return Response.json({
    p10: f.p10,
    p50: f.p50,
    p90: f.p90,
    p50Daily: f.p50Daily,
    source: f.source,
    daily: f.daily,
  });
}
