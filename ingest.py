"""
Chroma ingestion for the RAG knowledge base — Day 14-15.

Chunking decision (made deliberately, not defaulted): SECTION-SPLIT, not whole-document.
Each of the 4 policy docs is split on its markdown `##` headers. Reasoning:
  - Each doc already has 4-5 distinct, narrow sub-topics (e.g. refund_policy.md has
    "Return window", "Item condition", "Refund processing", "Return shipping costs",
    "Non-returnable items") that a single user question usually only needs ONE of.
  - Embedding the whole document as one vector would average across all of those
    sub-topics — exactly the "long-text-averages-out-the-specific-fact" problem
    flagged in chroma_langgraph_prep.md Part 3 — bad for retrieval precision on a
    short, specific question like "how long is the return window."
  - The docs are short enough (4-5 sections each, a few sentences per section) that
    section-level chunks are still coherent, self-contained units, not fragments.
  - Each chunk is prefixed with "{doc title} — {section title}" before embedding, so
    a chunk about "Return shipping costs" still carries the context that it's part of
    the Refund Policy, not a floating paragraph with no topic anchor.

This is the same "don't over-engineer past the actual problem size" judgment already
applied elsewhere in this project (SQLite over Postgres, Chroma over FAISS, EC2 over
Kubernetes) — 4 docs don't need a production chunking pipeline (recursive splitters,
overlap windows, token-based sizing). A plain header split is the right-sized tool here.
"""
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "policy_docs"

# doc_id -> filename. doc_id is what gets logged as retrieved_doc_id in logs.db,
# and what the live POST /documents endpoint (item 6) will take as its re-ingest key.
DOC_FILES = {
    "refund_policy": "refund_policy.md",
    "shipping_delivery_policy": "shipping_delivery_policy.md",
    "payment_methods": "payment_methods.md",
    "cancellation_fee_policy": "cancellation_fee_policy.md",
}


def chunk_markdown_by_section(text: str, doc_id: str) -> list[dict]:
    """
    Splits a markdown doc on '## ' headers. The H1 title (first '# ' line) is
    prefixed onto every chunk for context. Returns a list of
    {"id": ..., "text": ..., "metadata": {...}} dicts, ready for collection.upsert().
    """
    lines = text.strip().splitlines()

    has_h1 = bool(lines) and lines[0].startswith("# ")
    doc_title = lines[0].lstrip("#").strip() if has_h1 else doc_id
    body = "\n".join(lines[1:]) if has_h1 else text

    # Split on H2 headers, keeping the header line attached to its own section.
    parts = re.split(r"\n(?=## )", body.strip())

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        section_title_match = re.match(r"## (.+)", part)
        section_title = section_title_match.group(1).strip() if section_title_match else f"section_{i}"
        section_body = part[section_title_match.end():].strip() if section_title_match else part

        chunk_text = f"{doc_title} — {section_title}\n{section_body}"
        chunks.append({
            "id": f"{doc_id}-{i}",
            "text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "section_title": section_title,
            },
        })
    return chunks


def build_vector_store() -> None:
    """
    Uses get_or_create_collection + collection.upsert() (not .add()) — re-running this
    script after editing a doc updates the existing chunk IDs in place instead of
    erroring on duplicates. This upsert mechanism is what the live POST /documents
    endpoint (item 6) will reuse for re-ingesting a single doc without a full rebuild
    or a redeploy.
    """
    import chromadb  # imported here so chunk_markdown_by_section stays testable without chromadb installed

    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    # No embedding_function specified -> Chroma's default, all-MiniLM-L6-v2, local,
    # free, no API key. Matches the recommendation in chroma_langgraph_prep.md Part 4 —
    # matches this project's existing cost-conscious pattern (GGUF over API calls, etc).
    # hnsw:space="cosine" explicit, not the l2 default — makes retrieved "distance"
    # convert to a clean 0-1 similarity score (1 - distance) for the retry-loop
    # threshold in rag.py, instead of an unbounded, less-intuitive l2 number.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids, all_texts, all_metadatas = [], [], []
    for doc_id, filename in DOC_FILES.items():
        path = DOCS_DIR / filename
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown_by_section(text, doc_id)
        for c in chunks:
            all_ids.append(c["id"])
            all_texts.append(c["text"])
            all_metadatas.append(c["metadata"])
        print(f"{filename}: {len(chunks)} chunks")

    collection.upsert(ids=all_ids, documents=all_texts, metadatas=all_metadatas)
    print(f"\nTotal chunks upserted into '{COLLECTION_NAME}': {len(all_ids)}")


if __name__ == "__main__":
    build_vector_store()