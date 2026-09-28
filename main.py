import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama
from langgraph.graph import StateGraph, END

from db import init_db, log_request
from ingest import chunk_markdown_by_section, DOC_FILES, DOCS_DIR
from rag import (
    RAG_ELIGIBLE_INTENTS,
    get_collection,
    retrieve_with_bounded_retry,
    generate_grounded_reply,
)

from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

GGUF_REPO_ID = "Etasha/support_bot_gguf"
GGUF_FILENAME = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
MAX_NEW_TOKENS = 400
N_CTX = 2048
# Fixed seed for reproducible generation.
SEED = 42
# Default (2) matches EC2 t3.large's 2 vCPUs — the actual deploy target. Override
# locally via `export N_THREADS=4` Confirmed real, roughly-linear effect on this hardware: 2->4
# threads took one short generation call from 9.67s to 4.58s (debug_timing.py)..
N_THREADS = int(os.environ.get("N_THREADS", "2"))
SYSTEM_PROMPT = (
    "You are a customer support assistant. For every user message, respond with ONLY a JSON object "
    "with keys: category, intent, reply. Do not include any text outside the JSON object. "
    "The 'intent' MUST be exactly one of these values: cancel_order, change_order, "
    "change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, "
    "check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, "
    "delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, "
    "newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, "
    "review, set_up_shipping_address, switch_account, track_order, track_refund"
)
CLASSIFY_SYSTEM_PROMPT = (
    "You are a customer support intent classifier. For every user message, respond with ONLY "
    "a JSON object with keys: category, intent. Do not include a 'reply' key. Do not include "
    "any text outside the JSON object. "
    "\n\nEvery valid 'intent' value belongs to EXACTLY ONE of these three types. Each intent "
    "below has a real example message next to it — match the user's message to the closest "
    "example, not just to keywords:"
    "\n\nACTION — the user wants something done on, or is checking the status of, THEIR OWN "
    "specific order/account, even if no order number was given:"
    "\n  cancel_order (\"please cancel this order for me\")"
    "\n  change_order (\"can I swap an item in my order before it goes out\")"
    "\n  change_shipping_address (\"need to change where this is being delivered\")"
    "\n  check_invoice (\"i need to look at my invoice from last week\")"
    "\n  create_account (\"how do i make a new account here\")"
    "\n  delete_account (\"shut down my account please\")"
    "\n  edit_account (\"let me update my profile details\")"
    "\n  get_invoice (\"can you email me my invoice\")"
    "\n  get_refund (\"i need my money returned to me\")"
    "\n  newsletter_subscription (\"add me to your email list\")"
    "\n  payment_issue (\"i got billed twice for one order\")"
    "\n  place_order (\"how do i buy this item\")"
    "\n  recover_password (\"i forgot my login password\")"
    "\n  set_up_shipping_address (\"i want to add another delivery address\")"
    "\n  switch_account (\"move me from the free plan to premium\")"
    "\n  track_order (\"where is my package right now\")"
    "\n  track_refund (\"any update on my refund yet\")"
    "\n\nPOLICY — the user is asking a general question about rules, cost, or timing — nothing "
    "specific to their own order/account is being looked up or acted on:"
    "\n  check_cancellation_fee (\"do you charge anything if i cancel\")"
    "\n  check_payment_methods (\"what can i pay with\")"
    "\n  check_refund_policy (\"how does your refund process work\")"
    "\n  delivery_options (\"what shipping choices are available\")"
    "\n  delivery_period (\"how many days does shipping usually take\")"
    "\n\nOTHER — general communication, not an account action or a policy lookup:"
    "\n  complaint (\"my order keeps arriving late and i am fed up\")"
    "\n  contact_customer_service (\"how can i get in touch with support\")"
    "\n  contact_human_agent (\"i want to speak to a real person\")"
    "\n  registration_problems (\"the sign up page keeps throwing an error\")"
    "\n  review (\"id like to leave feedback on something i bought\")"
    "\n\nThe presence of words like 'cancel' or 'refund' does NOT by itself decide ACTION vs "
    "POLICY — only whether a specific existing order/account is actually being looked up or "
    "acted on decides it. A status/tracking question about the user's own order or refund is "
    "ACTION even with no order number given; a general rules/cost/timing question with nothing "
    "specific to look up is POLICY."
)
CLASSIFY_MAX_TOKENS = 150  
VALID_INTENTS = {
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
}

ID_PATTERN = re.compile(r"\b(ORD|INV|TRK|REF)-\d+\b")

DOMAIN_KEYWORDS = {
    "order", "orders", "my order", "purchase", "purchased", "buy", "bought",
    "shop", "shopping", "checkout", "cart", "item", "items", "product", "products",
    "cancel", "cancellation", "cancel my", "refund", "refunds", "return", "returns",
    "exchange", "money back", "my money", "reimburse", "reimbursement",
    "account", "sign up", "signup", "sign me up", "register", "registration",
    "create account", "delete account", "close my account", "log in", "login",
    "password", "recover password", "forgot password", "reset password",
    "payment", "payments", "pay", "paid", "card", "charge", "charged",
    "invoice", "bill", "billing", "price", "cost", "fee", "fees", "receipt",
    "transaction",
    "shipping", "ship", "delivery", "deliver", "delivered", "track", "tracking",
    "package", "packages", "shipment", "address", "arrive", "arrival",
    "eta", "how long", "when will", "late", "delayed", "delay",
    "lost", "missing", "damaged", "broken", "wrong item",
    "subscribe", "subscription", "unsubscribe", "newsletter", "emails",
    "mailing list", "opt out", "opt-in", "promo emails", "notifications",
    "plan", "membership", "premium", "basic plan", "upgrade", "downgrade",
    "switch account", "switch plan",
    "complaint", "complain", "complaining", "review", "rating", "feedback",
    "rude", "poor service", "bad experience", "issue", "problem", "trouble",
    "support", "customer service", "agent", "human", "person", "someone",
    "representative", "contact", "not happy", "unhappy", "dissatisfied",
    "what happened", "what are my options",
}


def looks_out_of_domain(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in DOMAIN_KEYWORDS)


REQUESTS_BY_BRANCH = Counter(
    "requests_by_branch_total", "Total requests by routing branch", ["branch"]
)
MALFORMED_JSON = Counter(
    "malformed_json_total", "Requests where model output failed to parse as JSON"
)
OFF_TAXONOMY = Counter(
    "off_taxonomy_total", "Requests with valid JSON but an intent outside VALID_INTENTS"
)
GROUNDING_FAILED = Counter(
    "grounding_failed_total", "RAG requests that failed the grounding/faithfulness check"
)  # first real use as of this build — was zero since Day 11-12, RAG branch now exists
GUARDRAIL_OUTCOME = Counter(
    "guardrail_outcome_total", "Total requests by final guardrail outcome", ["outcome"]
)


class ModelState:
    llm = None
    collection = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"Loading GGUF model {GGUF_REPO_ID}/{GGUF_FILENAME}...")
    state.llm = Llama.from_pretrained(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    print("Model loaded.")
    # Loaded once at startup, fresh PersistentClient connection per
    # request would be wasteful, not incorrect,but there's no reason
    #  to pay that cost on every /predict call.
    state.collection = get_collection()
    print("Chroma collection loaded.")
    yield


app = FastAPI(title="Support Bot", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # adds GET /metrics


class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    request_id: str
    category: str | None = None
    intent: str | None = None
    reply: str | None = None
    valid_json: bool
    guardrail_outcome: str
    fallback_reason: str | None = None
    latency_ms: float


class DocumentUpdateRequest(BaseModel):
    doc_id: str = Field(..., description=f"must be one of: {sorted(DOC_FILES)}")
    content: str = Field(..., min_length=1)


def extract_id_tokens(text: str) -> set[str]:
    return set(m.group(0) for m in ID_PATTERN.finditer(text))


def check_fabricated_ids(user_query: str, reply_text: str) -> tuple[bool, str | None]:
    provided_ids = extract_id_tokens(user_query)
    reply_ids = extract_id_tokens(reply_text)
    fabricated = reply_ids - provided_ids
    if fabricated:
        return False, f"fabricated_id: {sorted(fabricated)}"
    return True, None


def run_judge(parsed: dict | None) -> tuple[bool, str | None]:
    """Schema + taxonomy only. Grounding (RAG-specific) is checked separately in
    /predict, not folded in here, since it doesn't apply to the direct-passthrough
    branch at all — keeping it out of run_judge keeps this function meaningful for
    both branches instead of needing a grounding=None special case inside it."""
    if parsed is None:
        return False, "invalid_json"
    if not all(k in parsed for k in ("category", "intent", "reply")):
        return False, "missing_schema_keys"
    if not isinstance(parsed.get("category"), str) or not parsed["category"].strip():
        return False, "empty_category"
    if parsed.get("intent") not in VALID_INTENTS:
        return False, f"off_taxonomy_intent: {parsed.get('intent')!r}"
    return True, None


def classify_query(user_query: str) -> tuple[dict | None, str]:
    #Cheap, routing-only call — category+intent, no reply.
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    output = state.llm.create_chat_completion(
        messages=messages,
        max_tokens=CLASSIFY_MAX_TOKENS,
        temperature=0.3,
        seed=SEED,
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


def generate_structured_reply(user_query: str) -> tuple[dict | None, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    output = state.llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.3,
        seed=SEED,
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



class GraphState(TypedDict):
    user_query: str
    classify_intent: Optional[str]
    classify_category: Optional[str]
    branch: str  # "out_of_domain" | "direct" | "rag"
    retrieval: Optional[dict]
    parsed: Optional[dict]
    raw_output: str
    valid_json: bool
    in_taxonomy: bool
    grounding_passed: Optional[bool]
    judge_ok: bool
    judge_reason: Optional[str]
    ids_ok: bool
    ids_reason: Optional[str]
    outcome: str
    fallback_reason: Optional[str]


def ood_gate_node(gstate: GraphState) -> dict:
    if looks_out_of_domain(gstate["user_query"]):
        return {
            "branch": "out_of_domain", "parsed": None, "raw_output": "",
            "valid_json": False, "in_taxonomy": False, "grounding_passed": None,
            "outcome": "fallback", "fallback_reason": "out_of_domain_keyword_gate",
        }
    return {}


def route_after_ood(gstate: GraphState) -> str:
    return END if gstate.get("branch") == "out_of_domain" else "classify"


def classify_node(gstate: GraphState) -> dict:
    classify_parsed, _ = classify_query(gstate["user_query"])
    classify_valid = classify_parsed is not None
    classify_in_taxonomy = classify_valid and classify_parsed.get("intent") in VALID_INTENTS
    initial_intent = classify_parsed.get("intent") if classify_valid else None
    branch = "rag" if (classify_in_taxonomy and initial_intent in RAG_ELIGIBLE_INTENTS) else "direct"
    return {
        "classify_intent": initial_intent,
        "classify_category": classify_parsed.get("category") if classify_valid else None,
        "branch": branch,
    }


def route_after_classify(gstate: GraphState) -> str:
    return gstate["branch"]  # "rag" or "direct"


def rag_node(gstate: GraphState) -> dict:
    retrieval = retrieve_with_bounded_retry(state.llm, state.collection, gstate["user_query"], seed=SEED)
    parsed, raw_output = generate_grounded_reply(state.llm, gstate["user_query"], retrieval["candidates"], seed=SEED)
    valid_json = parsed is not None

    # don't trust the model's own restatement of intent.
    if valid_json:
        parsed["intent"] = gstate["classify_intent"]
    in_taxonomy = valid_json and parsed.get("intent") in VALID_INTENTS

    #  always answer, append a split-suggestion rather than
    # withholding when multiple docs genuinely cleared the relevance floor.
    if valid_json and len(retrieval["candidates"]) >= 2:
        parsed["reply"] = (
            parsed.get("reply", "") +
            " (This touches a few different policy areas — for full detail on "
            "each, feel free to ask about them one at a time.)"
        )

    return {
        "retrieval": retrieval, "parsed": parsed, "raw_output": raw_output,
        "valid_json": valid_json, "in_taxonomy": in_taxonomy,
        "grounding_passed": retrieval["grounding_ok"],
    }


def direct_node(gstate: GraphState) -> dict:
    parsed, raw_output = generate_structured_reply(gstate["user_query"])
    valid_json = parsed is not None
    in_taxonomy = valid_json and parsed.get("intent") in VALID_INTENTS
    return {
        "retrieval": None, "parsed": parsed, "raw_output": raw_output,
        "valid_json": valid_json, "in_taxonomy": in_taxonomy, "grounding_passed": None,
    }


def judge_node(gstate: GraphState) -> dict:
    parsed = gstate["parsed"]
    reply_text = parsed.get("reply", "") if parsed else ""
    judge_ok, judge_reason = run_judge(parsed)

    ids_ok, ids_reason = (True, None)
    if judge_ok:
        ids_ok, ids_reason = check_fabricated_ids(gstate["user_query"], reply_text)

    if gstate["branch"] == "rag":
        grounding_passed = gstate["grounding_passed"]
        outcome = "pass" if (judge_ok and ids_ok and grounding_passed) else "fallback"
        if not grounding_passed:
            retrieval = gstate["retrieval"]
            fallback_reason = (
                f"knowledge_gap: low_similarity_after_retry "
                f"(score={retrieval['similarity_score']:.3f}, doc={retrieval['doc_id']}, "
                f"attempts={retrieval['attempts']})"
            )
        else:
            fallback_reason = None if outcome == "pass" else (judge_reason or ids_reason)
    else:
        outcome = "pass" if (judge_ok and ids_ok) else "fallback"
        fallback_reason = None if outcome == "pass" else (judge_reason or ids_reason)

    return {
        "judge_ok": judge_ok, "judge_reason": judge_reason,
        "ids_ok": ids_ok, "ids_reason": ids_reason,
        "outcome": outcome, "fallback_reason": fallback_reason,
    }


_graph_builder = StateGraph(GraphState)
_graph_builder.add_node("ood_gate", ood_gate_node)
_graph_builder.add_node("classify", classify_node)
_graph_builder.add_node("rag", rag_node)
_graph_builder.add_node("direct", direct_node)
_graph_builder.add_node("judge", judge_node)
_graph_builder.set_entry_point("ood_gate")
_graph_builder.add_conditional_edges("ood_gate", route_after_ood, {"classify": "classify", END: END})
_graph_builder.add_conditional_edges("classify", route_after_classify, {"rag": "rag", "direct": "direct"})
_graph_builder.add_edge("rag", "judge")
_graph_builder.add_edge("direct", "judge")
_graph_builder.add_edge("judge", END)
compiled_graph = _graph_builder.compile()
# Compiled once at module load — building the graph structure doesn't touch the
# model/collection, so this is safe before lifespan runs. Only .invoke() (per
# request, after lifespan has loaded everything) actually calls into state.llm/
# state.collection via the node functions above.


@app.post("/predict", response_model=QueryResponse)
async def predict(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    initial_state: GraphState = {
        "user_query": req.user_query,
        "classify_intent": None, "classify_category": None, "branch": "",
        "retrieval": None, "parsed": None, "raw_output": "",
        "valid_json": False, "in_taxonomy": False, "grounding_passed": None,
        "judge_ok": False, "judge_reason": None, "ids_ok": True, "ids_reason": None,
        "outcome": "", "fallback_reason": None,
    }
    result = compiled_graph.invoke(initial_state)
    latency_ms = (time.perf_counter() - start) * 1000

    branch = result["branch"]
    parsed = result["parsed"]
    valid_json = result["valid_json"]
    in_taxonomy = result["in_taxonomy"]
    grounding_passed = result["grounding_passed"]
    outcome = result["outcome"]
    fallback_reason = result["fallback_reason"]
    raw_output = result["raw_output"]
    retrieval = result["retrieval"]
    retrieved_doc_id = retrieval["doc_id"] if retrieval else None
    retrieval_score = retrieval["similarity_score"] if retrieval else None
    reply_text = parsed.get("reply", "") if parsed else ""

    # branch is logged/metriced as "direct" for the out_of_domain case too — same
    # as the original inline version, which never had a separate DB/metrics bucket
    # for it (the out-of-domain gate short-circuits before classification exists).
    metrics_branch = "direct" if branch == "out_of_domain" else branch

    REQUESTS_BY_BRANCH.labels(branch=metrics_branch).inc()
    if not valid_json:
        MALFORMED_JSON.inc()
    elif not in_taxonomy:
        OFF_TAXONOMY.inc()
    GUARDRAIL_OUTCOME.labels(outcome=outcome).inc()
    if branch == "rag" and grounding_passed is False:
        GROUNDING_FAILED.inc()

    log_request(
        request_id=request_id, latency_ms=latency_ms, user_query=req.user_query,
        category=parsed.get("category") if valid_json else None,
        intent=parsed.get("intent") if valid_json else None, branch=metrics_branch,
        retrieved_doc_id=retrieved_doc_id, retrieval_score=retrieval_score,
        valid_json=valid_json, in_taxonomy=in_taxonomy, grounding_passed=grounding_passed,
        guardrail_outcome=outcome, fallback_reason=fallback_reason, raw_output=raw_output,
    )

    if outcome == "pass":
        return QueryResponse(
            request_id=request_id, category=parsed.get("category"), intent=parsed.get("intent"),
            reply=reply_text, valid_json=True, guardrail_outcome="pass",
            fallback_reason=None, latency_ms=latency_ms,
        )

    if branch == "out_of_domain":
        fallback_reply = ("I'm a customer support assistant and can only help with account, order, "
                           "payment, shipping, and similar questions — could you rephrase in that context?")
    else:
        fallback_reply = "Sorry, I couldn't process that reliably — could you rephrase or provide more detail?"

    return QueryResponse(
        request_id=request_id, category=None, intent=None,
        reply=fallback_reply,
        valid_json=valid_json, guardrail_outcome="fallback",
        fallback_reason=fallback_reason, latency_ms=latency_ms,
    )


@app.post("/documents")
async def update_document(req: DocumentUpdateRequest):
    """
    Item 6: live re-ingestion for one doc, no redeploy. Writes the new content to
    disk (so it survives a container restart, same as the original 4 docs) then
    re-chunks and upserts just that doc's chunks into the existing collection.
    collection.delete(where=...) first, not just upsert, because a doc's section
    count can change on edit — stale chunk IDs from a shrunk doc would otherwise
    linger in the collection forever.
    """
    if req.doc_id not in DOC_FILES:
        return {
            "status": "error",
            "reason": f"unknown doc_id {req.doc_id!r}, must be one of {sorted(DOC_FILES)}",
        }

    path = DOCS_DIR / DOC_FILES[req.doc_id]
    path.write_text(req.content, encoding="utf-8")

    chunks = chunk_markdown_by_section(req.content, req.doc_id)
    state.collection.delete(where={"doc_id": req.doc_id})
    state.collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return {"status": "ok", "doc_id": req.doc_id, "chunks_upserted": len(chunks)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.llm is not None,
        "collection_loaded": state.collection is not None,
    }