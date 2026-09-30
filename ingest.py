
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
    erroring on duplicates. 
    """
    import chromadb  # imported here so chunk_markdown_by_section stays testable without chromadb installed

    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    # No embedding_function specified -> Chroma's default, all-MiniLM-L6-v2, local,
    # free, no API key.
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