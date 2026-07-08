// One-time extractor: converts rag/embeddings.npz doc arrays to lib/data/docs.json.
// The cloud retriever (lib/tools/knowledge.ts) only needs document texts, not vectors,
// so this avoids shipping numpy/torch to the Next.js app.
//
// .npz is a NumPy zip archive with no first-class Node reader, so run the actual
// extraction with Python (numpy only, no torch needed):
//
//   python -c "import numpy as np, json; d=np.load('rag/embeddings.npz', allow_pickle=True); json.dump([{'id':str(i),'category':str(c),'text':str(t)} for i,c,t in zip(d['ids'],d['categories'],d['documents'])], open('lib/data/docs.json','w'))"
//
// Re-run that command any time rag/embeddings.npz changes.
console.log(
  "Run the Python one-liner in this file's header comment to (re)generate lib/data/docs.json from rag/embeddings.npz.",
);
