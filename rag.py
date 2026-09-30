from pathlib import Path

CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "policy_docs"

# Router: the only 5 intents that dispatch to RAG. Every other intent stays on the
# existing direct-passthrough path (unchanged from Day 5-7's main.py).
RAG_ELIGIBLE_INTENTS = {
    "check_refund_policy",
    "check_payment_methods",
    "check_cancellation_fee",
    "delivery_options",
    "delivery_period",
}

# Cosine similarity (1 - distance). Not yet validated against real queries — this is a
# starting value, not a tuned one; flagged honestly, same as MAX_NEW_TOKENS's early
# guesses in main.py's history. Needs a real pass once retrieval is running live:
# check what scores actual matching vs. non-matching queries produce, adjust from there.
SIMILARITY_THRESHOLD = 0.35
MAX_RETRIEVAL_ATTEMPTS = 2

REFORMULATE_SYSTEM_PROMPT = (
    "Rewrite the following customer support question as a single, clear, formal "
    "sentence describing the general policy topic it's asking about, so it can be "
    "matched against a policy document. Output ONLY the rewritten sentence, nothing else."
)

RAG_SYSTEM_PROMPT = (
    "You are a customer support assistant. You are given a user question and one or more "
    "excerpts from the company's policy documents, each labeled with which document it's "
    "from. Some excerpts may not actually be relevant to the question — use only the "
    "excerpt(s) that genuinely answer it, and ignore the rest. Using ONLY the information "
    "in the relevant excerpt(s), respond with ONLY a JSON object with keys: category, "
    "intent, reply. The 'reply' must be grounded strictly in the excerpt(s) you use — do "
    "not add facts, numbers, or policy details that are not stated in them. Do NOT ask the "
    "user for an order number, account details, or any other identifying information — "
    "these are general policy questions, not account lookups, and you have everything you "
    "need in the excerpts to answer directly. If none of the excerpts actually answer the "
    "question, say so honestly in the reply rather than guessing or deflecting. "
    "Write the reply concisely but completely: include every fact from the relevant excerpt(s) "
    "that actually answers the question, in as few words as that takes, but do not add "
    "greetings, pleasantries, restating the question back, offers of further help, or "
    "any other filler beyond what's needed to answer it. "
    "The 'intent' MUST be exactly one of these values: check_refund_policy, "
    "check_payment_methods, check_cancellation_fee, delivery_options, delivery_period"
)


def get_collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),  # silences a harmless but
        # noisy telemetry-library version-mismatch warning (capture() argument
        # count bug inside Chroma's own PostHog wrapper) — no effect on queries.
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


from ingest import DOC_FILES

# Feed one candidate PER SOURCE DOCUMENT, not a global top-K across all chunks.
# Real reason (this session): global top-K lets one dominant topic crowd out others
# entirely on a compound question. Observed case: "if my stuff never showed up do i
# still gotta pay for it" — the combined embedding leaned so heavily toward
# delivery/refund content that payment_methods.md's actually-decisive fact ("charged
# at order placement, not shipment") never appeared in the top-2 OR top-3 global
# candidates. Retrieving the best match within EACH doc separately guarantees every
# policy area gets one fair shot, regardless of how the combined question embeds.
def retrieve_top_chunk_per_doc(collection, query_text: str) -> list[dict]:
    """One retrieval per known source document. Returns up to len(DOC_FILES)
    candidates, sorted best (highest similarity) first."""
    candidates = []
    for doc_id in DOC_FILES:
        results = collection.query(
            query_texts=[query_text],
            n_results=1,
            where={"doc_id": doc_id},
        )
        if not results["documents"][0]:
            continue
        candidates.append({
            "chunk_text": results["documents"][0][0],
            "doc_id": doc_id,
            "similarity": 1 - results["distances"][0][0],
        })
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


# the model was handed 4 excerpts unconditionally and — being a small
# model without strong selective-synthesis ability — tried to address all 4
# regardless of relevance (a query with zero cancellation content still got a
# cancellation-fee paragraph, because the excerpt was simply there to use).
# 0.25, not lower — real data (this session) showed shipping_delivery_policy
# (genuinely relevant, 0.2500) and cancellation_fee_policy (genuinely irrelevant,
# 0.2368) score only 0.0132 apart for a real query. No single threshold separates
# them; lowering the floor to rescue one rescues both, reintroducing the exact
# irrelevant-content bloat this filter exists to prevent. Between the two failure
# modes — dropping a borderline-relevant doc (mildly incomplete answer) vs. keeping
# a genuinely irrelevant one (observed: bloated, borderline-fabricated answers) —
# the first is strictly less harmful. Staying at 0.25 accepts occasional incomplete
# answers over occasional actively-wrong-shaped ones.
MIN_CANDIDATE_SIMILARITY = 0.25


def filter_relevant_candidates(candidates: list[dict], min_similarity: float = MIN_CANDIDATE_SIMILARITY) -> list[dict]:
    """Drops candidates below the relevance floor. Always keeps at least the single
    best candidate, even if it's below the floor too — generation needs something to
    ground on, and the retry loop's own SIMILARITY_THRESHOLD check (on the unfiltered
    best) is what actually decides whether that's trustworthy, not this filter."""
    filtered = [c for c in candidates if c["similarity"] >= min_similarity]
    return filtered if filtered else candidates[:1]


def reformulate_query(llm, original_query: str, seed: int | None = None) -> str:
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": REFORMULATE_SYSTEM_PROMPT},
            {"role": "user", "content": original_query},
        ],
        max_tokens=64,
        temperature=0.3,
        seed=seed,
    )
    return output["choices"][0]["message"]["content"].strip()


def retrieve_with_bounded_retry(llm, collection, user_query: str, seed: int | None = None) -> dict:
    """
    The bounded 2-iteration retry loop. Returns a dict with everything the caller
    (main.py's /predict handler) needs to log and act on:
    {chunk_text, doc_id, similarity_score, attempts, reformulated_query, grounding_ok}
    """
    query = user_query
    reformulated_query = None

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        candidates = retrieve_top_chunk_per_doc(collection, query)
        best = candidates[0]

        if best["similarity"] >= SIMILARITY_THRESHOLD or attempt == MAX_RETRIEVAL_ATTEMPTS:
            return {
                # Filtered — only plausibly-relevant candidates reach generation.
                "candidates": filter_relevant_candidates(candidates),
                # Raw, unfiltered, ALL docs' scores — kept purely so the caller can
                # log/inspect real numbers instead of guessing where to set
                # MIN_CANDIDATE_SIMILARITY. Not used for any decision itself.
                "all_candidates": candidates,
                "chunk_text": best["chunk_text"],   # best single chunk, kept for logging clarity
                "doc_id": best["doc_id"],
                "similarity_score": best["similarity"],
                "attempts": attempt,
                "reformulated_query": reformulated_query,
                "grounding_ok": best["similarity"] >= SIMILARITY_THRESHOLD,
            }

        # Below threshold and attempts remain: reformulate once, loop back and retry.
        query = reformulate_query(llm, user_query, seed=seed)
        reformulated_query = query

    # Unreachable given the loop bounds above, but keeps the type checker honest.
    raise RuntimeError("retrieve_with_bounded_retry exited without returning")


def generate_grounded_reply(llm, user_query: str, candidates: list[dict], max_tokens: int = 400, seed: int | None = None) -> tuple[dict | None, str]:
    """
    Same shape as main.py's generate_structured_reply(), but the model gets one
    candidate excerpt PER SOURCE DOCUMENT alongside the question, labeled
    by source doc, and is instructed to use only the one(s) that actually answer it.
    Reuses the same model for generation, not a separate one 
    """
    import json

    excerpts_text = "\n\n".join(
        f"Excerpt {i + 1} (from {c['doc_id']}):\n{c['chunk_text']}"
        for i, c in enumerate(candidates)
    )
    user_content = f"Question: {user_query}\n\nRelevant policy excerpt(s):\n{excerpts_text}"
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        # Lowered from 0.3 — real, observed failure: with 0.3, the model sometimes
        # deflected to a learned-from-original-fine-tuning reflex ("please provide
        # your order number") instead of answering directly from the excerpts,
        # specifically on action-flavored phrasing ("...id rather cancel"). Less
        # sampling randomness should mean tighter adherence to this prompt's
        # explicit instructions over the older trained behavioral patterns.
        temperature=0.15,
        seed=seed,
        response_format={"type": "json_object"},
    )
    raw_output = output["choices"][0]["message"]["content"]

    try:
        json_start = raw_output.index("{")
        json_end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        parsed = None

    return parsed, raw_output