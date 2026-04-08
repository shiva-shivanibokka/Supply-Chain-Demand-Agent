"""
rag/retriever.py
----------------
This file handles the RETRIEVAL step of RAG.

ingest.py built the knowledge base (converted documents to vectors and stored them).
retriever.py uses that knowledge base to answer the question:
  "Given a user's query, which stored documents are most relevant?"

How the search works:
  1. Take the user's question as a plain string
  2. Convert it to a vector using the same embedding model used during ingest
  3. Ask ChromaDB: "find the N stored vectors closest to this query vector"
  4. ChromaDB returns the matching document texts with similarity scores
  5. We return those texts - they become the "context" for the LLM

The similarity metric is cosine similarity.
Cosine similarity measures the angle between two vectors.
  - Score = 1.0  → identical meaning (perfect match)
  - Score = 0.7  → very similar
  - Score = 0.3  → loosely related
  - Score = 0.0  → completely unrelated

We filter out results below a minimum similarity threshold so the LLM
doesn't get fed irrelevant context that would confuse its answer.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict

from rag.ingest import CHROMA_PATH, EMBEDDING_MODEL, COLLECTION_NAME


# How many documents to retrieve per query.
# 3 is usually enough - more context helps but also adds noise.
TOP_K = 3

# Minimum similarity score to include a result.
# Below this, the document is probably not relevant to the query.
MIN_SIMILARITY = 0.3


def get_collection() -> chromadb.Collection:
    """
    Loads and returns the existing ChromaDB collection.
    Call ingest.py first if this raises an error about missing collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Searches the knowledge base for documents relevant to the query.

    Args:
        query : the user's question or search text
        top_k : how many results to return (default 3)

    Returns:
        A list of dicts, each containing:
          - "text"       : the full document text
          - "id"         : the document ID (e.g. "policy_001")
          - "category"   : the document category (policy, incident, supplier, guidance)
          - "similarity" : cosine similarity score (0 to 1, higher = more relevant)

    Example:
        results = retrieve("what is the reorder policy for valves?")
        # Returns the reorder policy doc, maybe a supplier doc, etc.
    """

    collection = get_collection()

    # query_texts triggers the embedding function to convert the query string
    # to a vector, then searches for the nearest stored vectors.
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns distance (lower = more similar when using cosine space).
    # Convert to similarity score: similarity = 1 - distance
    # This makes it intuitive: higher score = more relevant.
    retrieved = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)

        # Skip results below our similarity threshold
        if similarity < MIN_SIMILARITY:
            continue

        retrieved.append(
            {
                "text": results["documents"][0][i],
                "id": results["ids"][0][i],
                "category": results["metadatas"][0][i].get("category", "unknown"),
                "similarity": similarity,
            }
        )

    return retrieved


def format_context(retrieved_docs: List[Dict]) -> str:
    """
    Formats the retrieved documents into a clean string block
    that we can paste into the LLM's prompt as context.

    The formatting makes it clear to Claude where each document starts
    and ends, and what its source is.

    Args:
        retrieved_docs: list of dicts from retrieve()

    Returns:
        A formatted string ready to be inserted into the agent prompt
    """

    if not retrieved_docs:
        return "No relevant documents found in the knowledge base."

    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"[Source: {doc['id']} | Category: {doc['category']} | "
            f"Relevance: {doc['similarity']:.2f}]\n{doc['text'].strip()}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick test - run this to verify RAG retrieval is working
    test_queries = [
        "what should I do if valve inventory is running low?",
        "tell me about SupplierC reliability issues",
        "how do I calculate safety stock?",
    ]

    print("Testing RAG retrieval...\n")
    for q in test_queries:
        print(f"Query: {q}")
        docs = retrieve(q)
        print(f"Retrieved {len(docs)} documents:")
        for d in docs:
            print(f"  [{d['similarity']:.2f}] {d['id']} ({d['category']})")
        print()
