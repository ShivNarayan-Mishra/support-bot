import time

print("Loading Chroma collection...")
t0 = time.perf_counter()
from rag import get_collection, retrieve_top_chunk
collection = get_collection()
t_load = time.perf_counter() - t0
print(f"  collection load: {t_load:.2f}s")

print("Running retrieval only (embedding + Chroma search, no LLM)...")
t0 = time.perf_counter()
chunk_text, doc_id, similarity = retrieve_top_chunk(collection, "what is your policy on refunds")
t_retrieve = time.perf_counter() - t0
print(f"  retrieval only: {t_retrieve:.2f}s  (doc={doc_id}, score={similarity:.3f})")

print("Running the SAME retrieval a second time (checking for one-time warm-up)...")
t0 = time.perf_counter()
retrieve_top_chunk(collection, "what is your policy on refunds")
t_retrieve2 = time.perf_counter() - t0
print(f"  retrieval only (2nd call): {t_retrieve2:.2f}s")

print("\nLoading the LLM (this part is slow no matter what — expected)...")
from llama_cpp import Llama
t0 = time.perf_counter()
llm = Llama.from_pretrained(
    repo_id="Etasha/support_bot_gguf",
    filename="Llama-3.2-3B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=2,
    verbose=False,
)
t_model_load = time.perf_counter() - t0
print(f"  model load: {t_model_load:.2f}s")

print("Running ONE plain generation call, short prompt, for comparison...")
t0 = time.perf_counter()
llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Reply with only the word: ok"},
        {"role": "user", "content": "hello"},
    ],
    max_tokens=10,
    temperature=0.3,
)
t_gen = time.perf_counter() - t0
print(f"  one short generation call: {t_gen:.2f}s")

print("Isolating PREFILL cost of the REAL, full SYSTEM_PROMPT (max_tokens=1)...")
from main import SYSTEM_PROMPT
t0 = time.perf_counter()
llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what is your policy on refunds"},
    ],
    max_tokens=1,   # forces near-zero decode, isolates prefill of the real prompt
    temperature=0.3,
)
t_prefill_real = time.perf_counter() - t0
print(f"  real SYSTEM_PROMPT, max_tokens=1 (prefill only): {t_prefill_real:.2f}s")

print("SAME real SYSTEM_PROMPT, full 400-token budget (prefill + decode)...")
t0 = time.perf_counter()
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what is your policy on refunds"},
    ],
    max_tokens=400,
    temperature=0.3,
    response_format={"type": "json_object"},
)
t_full_real = time.perf_counter() - t0
n_completion_tokens = out.get("usage", {}).get("completion_tokens", "unknown")
print(f"  real SYSTEM_PROMPT, max_tokens=400: {t_full_real:.2f}s (completion_tokens={n_completion_tokens})")

print("\n--- SUMMARY ---")
print(f"collection load:        {t_load:.2f}s")
print(f"retrieval (1st call):   {t_retrieve:.2f}s")
print(f"retrieval (2nd call):   {t_retrieve2:.2f}s")
print(f"model load:             {t_model_load:.2f}s (one-time, not per-request)")
print(f"one short gen call:     {t_gen:.2f}s")
print(f"real prompt, prefill only (max_tokens=1): {t_prefill_real:.2f}s")
print(f"real prompt, full 400-token budget:       {t_full_real:.2f}s (completion_tokens={n_completion_tokens})")
print(f"  => decode-only cost for the actual reply: ~{t_full_real - t_prefill_real:.2f}s "
      f"for {n_completion_tokens} tokens")