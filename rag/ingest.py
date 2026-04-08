"""
rag/ingest.py
-------------
This file builds the RAG knowledge base.

What is RAG?
  RAG stands for Retrieval-Augmented Generation.

  The problem it solves: LLMs like Claude are trained on general internet data.
  They don't know anything specific about YOUR supply chain - your parts,
  your suppliers, your inventory policies, your historical incidents.

  RAG fixes this by giving the LLM a searchable knowledge base of your
  own documents. When a user asks a question, we first RETRIEVE the most
  relevant documents from the knowledge base, then GENERATE an answer
  using both the question and the retrieved context.

  It's like an open-book exam vs a closed-book exam.
  Without RAG: Claude answers from memory (general knowledge only).
  With RAG: Claude answers with your actual documents in front of it.

How RAG works (step by step):
  1. INGEST (this file):
     - Take your documents (text files, notes, policies, incident reports)
     - Split them into smaller chunks (~200 words each)
     - Convert each chunk into a vector (an array of ~384 numbers) using
       a sentence embedding model
     - Store all vectors in a vector database (ChromaDB)

  2. RETRIEVE (retriever.py):
     - When a question comes in, embed the question the same way
     - Find the chunks whose vectors are closest to the question vector
     - Return the top-k most relevant chunks

  3. GENERATE (agent.py):
     - Pass the question + retrieved chunks to Claude
     - Claude uses both to generate a grounded, accurate answer

What are embeddings (vectors)?
  A sentence embedding converts text into a list of numbers that captures
  its MEANING. Sentences that mean similar things get similar vectors.

  Example:
    "part is running low"     → [0.21, -0.45, 0.87, ...]
    "inventory almost empty"  → [0.19, -0.43, 0.85, ...]  ← very similar!
    "the weather is sunny"    → [-0.62, 0.11, -0.34, ...] ← very different

  The model we use (all-MiniLM-L6-v2) produces 384-dimensional vectors.
  It's small, fast, runs locally, and works well for this use case.

What is ChromaDB?
  ChromaDB is a vector database - a database specifically designed to
  store and search embeddings. When you ask "find me chunks similar to
  this question", it does a fast nearest-neighbor search across all
  stored vectors and returns the closest matches.
  It runs locally as a folder on disk - no server needed.

How to run this file:
  python -m rag.ingest
  This creates the vector DB in rag/chroma_db/
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Where ChromaDB will store its data on disk
CHROMA_PATH = "rag/chroma_db"

# The embedding model we use to convert text to vectors.
# all-MiniLM-L6-v2 is fast, lightweight (~80MB), and accurate enough for this.
# It runs completely locally - no API call needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The ChromaDB collection name - like a table name in a regular database
COLLECTION_NAME = "supply_chain_docs"


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE DOCUMENTS
# ---------------------------------------------------------------------------
# In a real company, these would be loaded from PDFs, Confluence pages,
# SharePoint docs, or an internal wiki. Here we write them as strings
# to keep the project self-contained.
#
# These are realistic supply chain documents modeled after the kind of
# content a semiconductor equipment company (like Applied Materials) would have.
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "policy_001",
        "text": """
        Reorder Policy for Critical Parts (Category: Valve, Sensor)
        When inventory for any Valve or Sensor part falls below 2x the average
        weekly demand, a reorder request must be submitted immediately.
        Lead times for these parts range from 7 to 21 days depending on supplier.
        SupplierA and SupplierB are preferred vendors for Valves due to their
        consistent 7-day lead times. SupplierC has a 14-day lead time but offers
        15% lower unit cost for bulk orders above 500 units.
        Safety stock should be maintained at 1.5x the lead time demand.
        """,
        "category": "policy",
    },
    {
        "id": "policy_002",
        "text": """
        Inventory Management Guidelines for Pumps and Controllers
        Pumps and Controllers are classified as high-criticality parts.
        A stockout of any Pump or Controller halts production on the affected line.
        Minimum stock level: 30 days of average demand at all times.
        Any pump part with price above $2000 must be approved by the Supply Chain
        Director before a reorder is placed. Controllers sourced from SupplierD
        have shown quality issues in 2023 - escalate to procurement if a SupplierD
        controller order exceeds 100 units.
        """,
        "category": "policy",
    },
    {
        "id": "incident_001",
        "text": """
        Incident Report: Q3 2022 Stockout - PART_007 (Filter, SupplierB)
        Date: September 14, 2022
        Summary: PART_007 (Filter) went out of stock across North America and
        Europe regions simultaneously. Root cause: a demand spike of 340% above
        baseline due to an unplanned maintenance campaign on 12 customer sites.
        Production downtime: 4 days across 3 facilities.
        Financial impact: $1.2M in delayed shipments.
        Corrective action: Safety stock for all Filter parts increased from
        14 days to 21 days. Demand sensing alerts added for >50% weekly deviation.
        """,
        "category": "incident",
    },
    {
        "id": "incident_002",
        "text": """
        Incident Report: Q1 2023 Supplier Delay - SupplierC Valve Orders
        Date: February 3, 2023
        Summary: SupplierC experienced a 3-week production delay due to raw
        material shortages (semiconductor-grade aluminum). All pending Valve
        orders from SupplierC were delayed by 18-25 days beyond quoted lead time.
        Affected parts: PART_012, PART_019, PART_031.
        Mitigation: Emergency orders placed with SupplierA at 22% premium cost.
        Recommendation: Maintain dual-source capability for all Valve parts.
        Do not allow SupplierC to hold more than 60% of Valve demand.
        """,
        "category": "incident",
    },
    {
        "id": "supplier_001",
        "text": """
        Supplier Profile: SupplierA
        Headquarters: San Jose, CA
        Specialization: Valves, Sensors, Filters
        Average lead time: 7 days (domestic), 14 days (international)
        Quality rating: 4.8/5.0 (2023 audit)
        On-time delivery rate: 96.2%
        Preferred for: high-criticality orders, emergency reorders
        Payment terms: Net 30
        Minimum order quantity: 10 units
        Notes: SupplierA has a dedicated inventory buffer for our top 20 parts.
        They can fulfill same-day emergency orders up to 200 units for Valves.
        """,
        "category": "supplier",
    },
    {
        "id": "supplier_002",
        "text": """
        Supplier Profile: SupplierB
        Headquarters: Austin, TX
        Specialization: Filters, Pumps
        Average lead time: 10 days
        Quality rating: 4.5/5.0 (2023 audit)
        On-time delivery rate: 91.7%
        Preferred for: Filters (best pricing), bulk orders
        Payment terms: Net 45
        Minimum order quantity: 50 units for Filters, 5 units for Pumps
        Notes: SupplierB had a quality incident in Q3 2022 (see incident_001).
        All Filter shipments now require incoming quality inspection.
        Price is 12% lower than SupplierA for equivalent Filter parts.
        """,
        "category": "supplier",
    },
    {
        "id": "supplier_003",
        "text": """
        Supplier Profile: SupplierC
        Headquarters: Taipei, Taiwan
        Specialization: Valves (bulk), Controllers
        Average lead time: 14 days (air freight), 35 days (sea freight)
        Quality rating: 4.2/5.0 (2023 audit)
        On-time delivery rate: 84.1%
        Preferred for: large bulk orders where cost is priority over speed
        Payment terms: Net 60
        Notes: SupplierC experienced significant delays in early 2023 due to
        raw material shortages. Recommended to not exceed 60% of Valve sourcing
        from SupplierC. They offer 15% discount for orders above 500 units.
        """,
        "category": "supplier",
    },
    {
        "id": "supplier_004",
        "text": """
        Supplier Profile: SupplierD
        Headquarters: Munich, Germany
        Specialization: Controllers, Sensors (precision)
        Average lead time: 21 days
        Quality rating: 3.9/5.0 (2023 audit) - flagged for improvement
        On-time delivery rate: 88.3%
        Notes: SupplierD had a batch quality issue in 2023 (mis-calibrated
        Controllers, batch IDs starting with CD-2023-Q2). All parts from this
        batch were recalled. Escalate any SupplierD Controller orders above
        100 units to procurement for review. Quality improvement plan is
        currently in progress with a re-audit scheduled for Q2 2024.
        """,
        "category": "supplier",
    },
    {
        "id": "forecast_guidance_001",
        "text": """
        Demand Forecasting Guidelines - Service Parts Division
        Forecast horizon: 30 days (operational planning) and 90 days (strategic)
        Key demand drivers to monitor:
          - Customer maintenance schedules (Q1 and Q3 are highest maintenance seasons)
          - New product installations (NPI ramp increases service part demand by 20-40%)
          - Reliability campaigns (field campaigns can spike specific part demand 2-5x)
          - Regional seasonality: Asia Pacific peaks in Q2, North America in Q3
        Forecast accuracy target: MAPE < 15% for top 20 parts by spend
        Escalation threshold: any forecast deviation > 30% week-over-week
        must be reviewed by the demand planning team.
        """,
        "category": "guidance",
    },
    {
        "id": "forecast_guidance_002",
        "text": """
        Safety Stock Calculation Policy
        Formula: Safety Stock = Z * sigma_demand * sqrt(lead_time)
        Where:
          Z = service level factor (1.65 for 95% service level)
          sigma_demand = standard deviation of daily demand over last 90 days
          lead_time = supplier lead time in days
        For parts with lead_time > 20 days, use Z = 1.96 (99% service level)
        because longer lead times mean more exposure to demand variability.
        Review safety stock levels quarterly or whenever a supplier lead time
        changes by more than 5 days. Parts with MAPE > 20% in the last quarter
        should have their safety stock multiplied by 1.3 as a buffer.
        """,
        "category": "guidance",
    },
]


def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Creates the ChromaDB vector database from the documents above.

    If the database already exists and force_rebuild is False,
    we just load and return the existing one. No need to re-embed
    every time you restart the app.

    Args:
        force_rebuild: if True, delete and rebuild the database from scratch

    Returns:
        The ChromaDB collection object (used by retriever.py to search)
    """

    # Delete existing DB if force rebuild requested
    if force_rebuild and os.path.exists(CHROMA_PATH):
        import shutil

        shutil.rmtree(CHROMA_PATH)
        print("Deleted existing ChromaDB. Rebuilding...")

    # PersistentClient stores the database as files on disk.
    # This means you only need to embed the documents once -
    # the vectors are saved and reloaded automatically next time.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # The embedding function tells ChromaDB which model to use
    # when converting text to vectors. We use the same model for
    # both ingestion (here) and retrieval (retriever.py) - they
    # MUST match or the similarity search won't work correctly.
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # get_or_create_collection: if it already exists, return it.
    # If not, create a new empty one.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
    )

    # If the collection already has documents, skip re-ingestion
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        print(
            f"Knowledge base already exists with {existing_count} documents. Skipping ingest."
        )
        print("Run with force_rebuild=True to rebuild from scratch.")
        return collection

    # Embed and insert all documents
    print(f"Embedding {len(DOCUMENTS)} documents using '{EMBEDDING_MODEL}'...")
    print("(First run downloads ~80MB model - subsequent runs are instant)\n")

    ids = [doc["id"] for doc in DOCUMENTS]
    texts = [doc["text"].strip() for doc in DOCUMENTS]
    metas = [{"category": doc["category"]} for doc in DOCUMENTS]

    # add() converts each text to a vector using the embedding function
    # and stores both the vector and the original text in ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
    )

    print(f"Done. {collection.count()} documents stored in {CHROMA_PATH}")
    return collection


if __name__ == "__main__":
    build_knowledge_base(force_rebuild=True)
    print("\nKnowledge base ready.")
