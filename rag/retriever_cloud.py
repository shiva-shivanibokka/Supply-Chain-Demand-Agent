"""
Lightweight RAG retriever for cloud deployment.

Uses keyword overlap scoring instead of chromadb so no heavy dependencies
are needed on Hugging Face Spaces. Embeddings were pre-built locally and
saved to rag/embeddings.npz which is committed to the repo.

Same public API as rag/retriever.py:
    retrieve(query, top_k) -> list of dicts
    format_context(docs)   -> str
"""

import os
import re
import numpy as np
from typing import List, Dict

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npz")
TOP_K = 3
MIN_SCORE = 0.05

# Loaded once and kept in memory
_STORE = None


def _get_store():
    global _STORE
    if _STORE is None:
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        _STORE = {
            "ids": list(data["ids"]),
            "documents": list(data["documents"]),
            "categories": list(data["categories"]),
            "embeddings": data["embeddings"].astype(np.float32),
        }
    return _STORE


STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "and",
    "but",
    "or",
    "not",
    "that",
    "this",
    "it",
    "its",
    "we",
    "i",
    "you",
    "he",
    "she",
    "they",
    "all",
    "any",
    "each",
    "some",
    "if",
    "about",
    "up",
    "so",
    "out",
    "what",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def _keyword_scores(query: str, documents: List[str]) -> np.ndarray:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return np.zeros(len(documents))

    scores = np.zeros(len(documents))
    for i, doc in enumerate(documents):
        doc_words = _tokenize(doc)
        doc_set = set(doc_words)
        overlap = query_terms & doc_set
        if not overlap:
            continue
        base = len(overlap) / len(query_terms)
        tf = sum(doc_words.count(t) for t in overlap)
        bonus = min(tf / (len(doc_words) + 1), 0.3)
        scores[i] = base + bonus

    mx = scores.max()
    if mx > 0:
        scores /= mx
    return scores


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Returns top_k documents most relevant to the query."""
    store = _get_store()
    scores = _keyword_scores(query, store["documents"])
    idxs = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in idxs:
        score = float(scores[idx])
        if score < MIN_SCORE:
            continue
        results.append(
            {
                "text": store["documents"][idx],
                "id": store["ids"][idx],
                "category": store["categories"][idx],
                "similarity": round(score, 4),
            }
        )
    return results


def format_context(docs: List[Dict]) -> str:
    """Formats retrieved docs into a string for the LLM prompt."""
    if not docs:
        return "No relevant documents found in the knowledge base."
    parts = []
    for d in docs:
        parts.append(
            f"[Source: {d['id']} | Category: {d['category']} | "
            f"Relevance: {d['similarity']:.2f}]\n{d['text'].strip()}"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    tests = [
        "what should I do if valve inventory is running low?",
        "tell me about SupplierC reliability issues",
        "how do I calculate safety stock?",
    ]
    for q in tests:
        print(f"Q: {q}")
        for d in retrieve(q):
            print(f"  [{d['similarity']:.2f}] {d['id']} ({d['category']})")
        print()
