import re
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from db import init_db, log_request

# ---------------------------------------------------------------------------
# Config 
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"  # ungated mirror of meta-llama's model, same weights
ADAPTER_ID = "Etasha/support_bot"
MAX_NEW_TOKENS = 256 #128 tokens were tried but they cut off the queries halfway leading it to being
                     #qualified as invalid

#This same prompt was used to train the LoRA weights. As the user will be querying to the entire model
#now the prompt is reproduced here to prevent any drifts
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

# Same 27-intent whitelist, as a set, for the judge's taxonomy check.
VALID_INTENTS = {
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
}

# TODO: category whitelist (the ~10 Bitext categories) not provided yet — judge only checks
# that 'category' is a non-empty string for now, not real whitelist membership.

# CPU-loading constraint (per plan): load_in_4bit / bitsandbytes is GPU-only and will not run
# on the free-tier CPU deploy target. bf16, then dynamically INT8-quantized post-load (Entry 20)
# instead — x86 CPUs lack native fp16 matmul support (Entry 19's correction), and dynamic
# quantization further roughly halves linear-layer weight memory on top of that.
TORCH_DTYPE = torch.bfloat16

# Fabricated-ID guardrail patterns
ID_PATTERN = re.compile(r"\b(ORD|INV|TRK|REF)-\d+\b")

# Out-of-domain pre-check — deterministic, doesn't depend on model behavior.The model was unreliable
# in classifying OOD queries and also spent atleast 120s on them.This keyword check runs before the 
# is called to handle such issus and prevent unnecessary computation .
# This is not a semantic based gate and a simple heuristic to manage the OOD queries.Thus an obvious
# limitation wherein a query made without these keywords can be misclassified as out of domain.
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

# ---------------------------------------------------------------------------
# Model holder (loaded once at startup, not per-request)
# ---------------------------------------------------------------------------
class ModelState:
    tokenizer = None
    model = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print(f"Loading base model {BASE_MODEL_ID} + adapter {ADAPTER_ID} (dtype={TORCH_DTYPE}, CPU)...")
    state.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=TORCH_DTYPE,
        #device_map="cpu",  # removed (Entry 19) — forces accelerate backend, throws a
                             # misleading "accelerate is missing" error on CPU-only hosts.
                             # PyTorch defaults to CPU without this arg anyway.
        low_cpu_mem_usage=True,  # avoids holding a full fp32 copy in memory during load
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER_ID)

    # --- Merge the LoRA adapter into the base weights (Entry 22) ------------
    # Previously the base model and adapter stayed as two separate pieces — every forward
    # pass computed the base weights AND the LoRA delta on top, as two separate paths
    #while this does not reduce the model size as the LoRA layer is only 200MB, the 
    #additional per layer computation during runtime is removed. This also removes
    # the peft dependancy later.
    model = peft_model.merge_and_unload()
    model.eval()

    # --- Dynamic INT8 quantization: TRIED, REVERTED (Entry 21) --------------
    # torch.quantization.quantize_dynamic requires float32 activations at runtime — it is
    # NOT compatible with bfloat16 inputs, and crashes on the first real /predict call with
    # "RuntimeError: expected scalar type Float but found BFloat16". The "correct" fix
    # (cast the whole model to float32 before quantizing) doubles memory vs bf16, which is
    # too risky on this host's tight RAM ceiling (was already at ~7.5GB/9.7GB before
    # quantization even ran) — casting to fp32 first could easily trigger the same
    # OOM/swap crash this was meant to help avoid, just moved to a different step.
    # Reverted for now. If memory is still a real problem after a clean bf16-only latency
    # test, the next options to consider are merging the LoRA adapter into the base
    # weights (removes double-bookkeeping) or a GGUF/llama.cpp switch (bigger rewrite,
    # but the standard low-RAM CPU inference path) — not re-attempting fp32+quantize on
    # this hardware.
    state.model = model
    print("Model loaded (bf16, no quantization).")
    yield
    # no teardown needed for a read-only in-memory model


app = FastAPI(title="Support Bot", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fabricated-ID guardrail
# ---------------------------------------------------------------------------
def extract_id_tokens(text: str) -> set[str]:
    """Return the full matched ID-shaped tokens (e.g. 'ORD-48291'), not just the prefix groups."""
    return set(m.group(0) for m in ID_PATTERN.finditer(text))


def check_fabricated_ids(user_query: str, reply_text: str) -> tuple[bool, str | None]:
    """
    Returns (ok, reason). ok=False means the reply contains an ID-shaped token that was never
    present in the user's own message — the model fabricated it.
    """
    provided_ids = extract_id_tokens(user_query)
    reply_ids = extract_id_tokens(reply_text)
    fabricated = reply_ids - provided_ids
    if fabricated:
        return False, f"fabricated_id: {sorted(fabricated)}"
    return True, None


# ---------------------------------------------------------------------------
# Judge — schema + taxonomy check. Exact-match (the 3rd tier from the Project 1 eval
# harness) only exists offline against known-correct labels; live traffic has no ground
# truth to exact-match against (see project_journal.md Phase 8) so it's schema+taxonomy only here.
# ---------------------------------------------------------------------------
def run_judge(parsed: dict | None) -> tuple[bool, str | None]:
    if parsed is None:
        return False, "invalid_json"
    if not all(k in parsed for k in ("category", "intent", "reply")):
        return False, "missing_schema_keys"
    if not isinstance(parsed.get("category"), str) or not parsed["category"].strip():
        return False, "empty_category"  # TODO: real category whitelist once provided
    if parsed.get("intent") not in VALID_INTENTS:
        return False, f"off_taxonomy_intent: {parsed.get('intent')!r}"
    return True, None


# ---------------------------------------------------------------------------
# Model call — uses the tokenizer's chat template with system/user roles, matching
# the exact conversation format used at training time (system prompt + user instruction
# -> assistant emits the target_json string). See build_conversation() in the training
# notebook — this mirrors that structure, not a flat hand-built prompt string.
# ---------------------------------------------------------------------------
def generate_structured_reply(user_query: str) -> tuple[dict | None, str]:
    """Returns (parsed_dict_or_None, raw_output_text)."""
    import json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    prompt_text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(prompt_text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = state.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Only decode newly generated tokens, not the echoed prompt — avoids the prompt's
    # own JSON-shaped system instructions confusing the {..} extraction below.
    new_tokens = output_ids[0][input_len:]
    raw_output = state.tokenizer.decode(new_tokens, skip_special_tokens=True)

    try:
        json_start = raw_output.index("{")
        json_end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        parsed = None

    return parsed, raw_output


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=QueryResponse)
async def predict(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Out-of-domain gate — runs BEFORE the model, deterministic, saves latency on obvious
    # non-support queries. See DOMAIN_KEYWORDS comment for the honest trade-off.
    if looks_out_of_domain(req.user_query):
        latency_ms = (time.perf_counter() - start) * 1000
        log_request(
            request_id=request_id,
            latency_ms=latency_ms,
            user_query=req.user_query,
            category=None,
            intent=None,
            branch="direct",
            retrieved_doc_id=None,
            retrieval_score=None,
            valid_json=False,
            in_taxonomy=False,
            grounding_passed=None,
            guardrail_outcome="fallback",
            fallback_reason="out_of_domain_keyword_gate",
            raw_output="",
        )
        return QueryResponse(
            request_id=request_id,
            category=None,
            intent=None,
            reply="I'm a customer support assistant and can only help with account, order, "
                  "payment, shipping, and similar questions — could you rephrase in that context?",
            valid_json=False,
            guardrail_outcome="fallback",
            fallback_reason="out_of_domain_keyword_gate",
            latency_ms=latency_ms,
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

    # Every request is logged, pass or fallback, this is what makes drift/quality checkable later,
    # not just assumed. branch is always "direct" until RAG exists .
    log_request(
        request_id=request_id,
        latency_ms=latency_ms,
        user_query=req.user_query,
        category=parsed.get("category") if valid_json else None,
        intent=parsed.get("intent") if valid_json else None,
        branch="direct",
        retrieved_doc_id=None,
        retrieval_score=None,
        valid_json=valid_json,
        in_taxonomy=in_taxonomy,
        grounding_passed=None,
        guardrail_outcome=outcome,
        fallback_reason=fallback_reason,
        raw_output=raw_output,
    )

    if outcome == "pass":
        return QueryResponse(
            request_id=request_id,
            category=parsed.get("category"),
            intent=parsed.get("intent"),
            reply=reply_text,
            valid_json=True,
            guardrail_outcome="pass",
            fallback_reason=None,
            latency_ms=latency_ms,
        )

    return QueryResponse(
        request_id=request_id,
        category=None,
        intent=None,
        reply="Sorry, I couldn't process that reliably — could you rephrase or provide more detail?",
        valid_json=valid_json,
        guardrail_outcome="fallback",
        fallback_reason=fallback_reason,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.model is not None}