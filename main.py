import json
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama

from db import init_db, log_request

GGUF_REPO_ID = "Etasha/support_bot_gguf"
GGUF_FILENAME = "Llama-3.2-3B-Instruct.Q4_K_M.gguf" 
MAX_NEW_TOKENS = 400 
N_CTX = 2048
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
    "order", "orders", "cancel", "cancellation", "refund", "return", "returns",
    "account", "password", "login", "log in", "sign up", "signup", "register", "registration",
    "payment", "pay", "paid", "card", "charge", "charged", "invoice", "bill", "billing",
    "shipping", "ship", "delivery", "deliver", "track", "tracking", "package", "shipment",
    "address", "subscribe", "subscription", "newsletter", "complaint", "review",
    "feedback", "support", "agent", "human", "customer service", "cancel my",
    # added after CI caught these as real false-positive OOD rejections:
    "person", "someone", "representative", "premium", "basic plan", "upgrade",
    "downgrade", "plan", "membership",
}


def looks_out_of_domain(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in DOMAIN_KEYWORDS)


class ModelState:
    llm = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"Loading GGUF model {GGUF_REPO_ID}/{GGUF_FILENAME}...")
    state.llm = Llama.from_pretrained(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        n_ctx=N_CTX,
        n_threads=2, 
        verbose=False,
    )
    print("Model loaded.")
    yield


app = FastAPI(title="Support Bot", lifespan=lifespan)


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
    if parsed is None:
        return False, "invalid_json"
    if not all(k in parsed for k in ("category", "intent", "reply")):
        return False, "missing_schema_keys"
    if not isinstance(parsed.get("category"), str) or not parsed["category"].strip():
        return False, "empty_category"
    if parsed.get("intent") not in VALID_INTENTS:
        return False, f"off_taxonomy_intent: {parsed.get('intent')!r}"
    return True, None


def generate_structured_reply(user_query: str) -> tuple[dict | None, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    output = state.llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.3,
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


@app.post("/predict", response_model=QueryResponse)
async def predict(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    if looks_out_of_domain(req.user_query):
        latency_ms = (time.perf_counter() - start) * 1000
        log_request(
            request_id=request_id, latency_ms=latency_ms, user_query=req.user_query,
            category=None, intent=None, branch="direct", retrieved_doc_id=None,
            retrieval_score=None, valid_json=False, in_taxonomy=False, grounding_passed=None,
            guardrail_outcome="fallback", fallback_reason="out_of_domain_keyword_gate", raw_output="",
        )
        return QueryResponse(
            request_id=request_id, category=None, intent=None,
            reply="I'm a customer support assistant and can only help with account, order, "
                  "payment, shipping, and similar questions — could you rephrase in that context?",
            valid_json=False, guardrail_outcome="fallback",
            fallback_reason="out_of_domain_keyword_gate", latency_ms=latency_ms,
        )

    parsed, raw_output = generate_structured_reply(req.user_query)
    reply_text = parsed.get("reply", "") if parsed else ""

    valid_json = parsed is not None
    in_taxonomy = valid_json and parsed.get("intent") in VALID_INTENTS

    judge_ok, judge_reason = run_judge(parsed)

    ids_ok, ids_reason = (True, None)
    if judge_ok:
        ids_ok, ids_reason = check_fabricated_ids(req.user_query, reply_text)

    latency_ms = (time.perf_counter() - start) * 1000
    outcome = "pass" if (judge_ok and ids_ok) else "fallback"
    fallback_reason = None if outcome == "pass" else (judge_reason or ids_reason)

    log_request(
        request_id=request_id, latency_ms=latency_ms, user_query=req.user_query,
        category=parsed.get("category") if valid_json else None,
        intent=parsed.get("intent") if valid_json else None, branch="direct",
        retrieved_doc_id=None, retrieval_score=None, valid_json=valid_json,
        in_taxonomy=in_taxonomy, grounding_passed=None, guardrail_outcome=outcome,
        fallback_reason=fallback_reason, raw_output=raw_output,
    )

    if outcome == "pass":
        return QueryResponse(
            request_id=request_id, category=parsed.get("category"), intent=parsed.get("intent"),
            reply=reply_text, valid_json=True, guardrail_outcome="pass",
            fallback_reason=None, latency_ms=latency_ms,
        )

    return QueryResponse(
        request_id=request_id, category=None, intent=None,
        reply="Sorry, I couldn't process that reliably — could you rephrase or provide more detail?",
        valid_json=valid_json, guardrail_outcome="fallback",
        fallback_reason=fallback_reason, latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.llm is not None}