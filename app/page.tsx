import parts from "@/lib/data/parts.json";

export default function Home() {
  return <main className="p-8">Loaded {parts.length} parts. UI coming next.</main>;
}
