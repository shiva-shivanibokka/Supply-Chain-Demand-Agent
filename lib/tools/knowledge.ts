import docs from "@/lib/data/docs.json";

type Doc = { id: string; category: string; text: string };
const DOCS = docs as Doc[];

const STOP = new Set(
  "a an the is are was were be been being have has had do does did will would could should may might to of in for on with at by from as into and but or not that this it its we i you he she they all any each some if about up so out what".split(
    " ",
  ),
);

function tokenize(t: string): string[] {
  return (t.toLowerCase().match(/[a-z0-9_]+/g) ?? []).filter((w) => !STOP.has(w) && w.length > 2);
}

export function searchKnowledge(query: string, topK = 3): string {
  const q = new Set(tokenize(query));
  if (q.size === 0) return "No relevant documents found in the knowledge base.";
  const scored = DOCS.map((d) => {
    const words = tokenize(d.text);
    const set = new Set(words);
    const overlap = [...q].filter((t) => set.has(t));
    if (overlap.length === 0) return { d, score: 0 };
    const base = overlap.length / q.size;
    const tf = overlap.reduce((s, t) => s + words.filter((w) => w === t).length, 0);
    return { d, score: base + Math.min(tf / (words.length + 1), 0.3) };
  });
  const max = Math.max(...scored.map((s) => s.score));
  const top = scored
    .map((s) => ({ ...s, score: max > 0 ? s.score / max : 0 }))
    .sort((a, b) => b.score - a.score)
    .filter((s) => s.score >= 0.05)
    .slice(0, topK);
  if (top.length === 0) return "No relevant documents found in the knowledge base.";
  return top
    .map((s) => `[Source: ${s.d.id} | Category: ${s.d.category} | Relevance: ${s.score.toFixed(2)}]\n${s.d.text.trim()}`)
    .join("\n\n");
}
