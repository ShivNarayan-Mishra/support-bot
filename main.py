import re
import time
import uuid
from contextlib import asynccontextmanager

import torch

# container was reporting 1 thread despite 2 vCPUs being available, forcing it explicitly
torch.set_num_threads(2)

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from db import init_db, log_request

BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"  # ungated mirror, same weights as meta-llama's
ADAPTER_ID = "Etasha/support_bot"
MAX_NEW_TOKENS = 256  # 128 was tried, cut replies off mid-JSON, causing invalid_json failures

# same prompt the model was trained on, kept verbatim so it doesn't drift from training
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

# category whitelist isn't enforced - only 20/27 intents are in the docs, intent is what
# actually drives routing anyway, so this stays a soft non-empty-string check
TORCH_DTYPE = torch.float16

ID_PATTERN = re.compile(r"\b(ORD|INV|TRK|REF)-\d+\b")

# keyword gate runs before the model so obviously off-topic queries skip the ~2min generation
# cost entirely. coarse heuristic - a real query with none of these words could get misgated
DOMAIN_KEYWORDS = {
    "order", "orders", "cancel", "cancellation", "refund", "return", "returns",
    "account", "password", "login", "log in", "sign up", "signup", "register", "registration",
    "payment", "pay", "paid", "card", "charge", "charged", "invoice", "bill", "billing",
    "shipping", "ship", "delivery", "deliver", "track", "tracking", "package", "shipment",
    "address", "subscribe", "subscription", "newsletter", "complaint", "review",
    "feedback", "support", "agent", "human", "customer service", "cancel my",
}


def looks_out_of_domain(query: str) -> bool:
    q = query.lower()
    return not any(kw in q for kw in DOMAIN_KEYWORDS)


class ModelState:
    tokenizer = None
    model = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # loads once at startup, not per-request - reloading a multi-GB model per call
    # would be unusable
    init_db()
    print(f"Loading base model {BASE_MODEL_ID} + adapter {ADAPTER_ID} (dtype={TORCH_DTYPE}, CPU)...")
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    state.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,  # avoids briefly holding a full fp32 copy during load
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER_ID)
    peft_model.eval()
    # NOT merging adapter into base - merge_and_unload() caused severe memory thrashing
    # on this instance's RAM (see running log). keeping base+adapter separate instead.
    state.model = peft_model
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
    guardrail_outcome: str  # "pass" | "fallback"
    fallback_reason: str | None = None
    latency_ms: float


def extract_id_tokens(text: str) -> set[str]:
    return set(m.group(0) for m in ID_PATTERN.finditer(text))


def check_fabricated_ids(user_query: str, reply_text: str) -> tuple[bool, str | None]:
    # catches the model inventing an order/tracking ID that the user never actually gave it
    provided_ids = extract_id_tokens(user_query)
    reply_ids = extract_id_tokens(reply_text)
    fabricated = reply_ids - provided_ids
    if fabricated:
        return False, f"fabricated_id: {sorted(fabricated)}"
    return True, None


def run_judge(parsed: dict | None) -> tuple[bool, str | None]:
    # schema + taxonomy check only. exact-match accuracy only exists in the offline eval
    # suite, since live traffic has no ground-truth label to check against
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
    import json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    # apply_chat_template matters - hand-building this string doesn't match what the
    # model actually trained on
    prompt_text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(prompt_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = state.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # slice off the echoed prompt, only decode what the model actually generated
    new_tokens = output_ids[0][input_len:]
    raw_output = state.tokenizer.decode(new_tokens, skip_special_tokens=True)

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

    # log every request, pass or fallback - a logging system that only records
    # successes can't tell you how often things are actually failing
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
    return {"status": "ok", "model_loaded": state.model is not None}