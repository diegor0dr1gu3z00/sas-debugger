"""Pluggable inference backend for the demo app.

The demo can run in two modes:

* ``deterministic`` (default): the agent replays a curated expert trace. No LM.
* ``model``: the agent calls a real small LM to choose the root cause and the
  diagnostic SQL, which is then graded against the real trap/clean oracle.

Backends for the model mode:

* ``transformers`` — in-process (base + optional Peft LoRA adapter), mirrors the
  stack the tokens were actually fine-tuned with.
* ``llama.cpp`` — HTTP against a running ``llama-server`` OpenAI-compatible
  endpoint (``/v1/chat/completions``).

Both expose the same callable: ``generate(system, user, temperature=0.0) -> str``.
"""

from __future__ import annotations

import re

import requests

SYSTEM_PROMPT = (
    "You are a senior data-quality debugger. A reporting pipeline materialized a "
    "final table over a production database, and ONE row of that table is "
    "suspect: its value disagrees with an independent recomputation of the same "
    "business metric. You have read-only SQL access to the whole working "
    "database (source tables + staging tables + the final reported table). "
    "Investigate, form hypotheses about which table/field/transformation is "
    "wrong, then finish with a localisation and a diagnostic SQL query that "
    "flags every row with the same incoherence (reference the suspect columns; "
    "do NOT filter by the suspect's primary key; the query must return 0 rows "
    "on a healthy database).\n"
    "Emit actions as a single JSON object:\n"
    '  {"action": "localize", "root_cause": {"table": "...", "columns": '
    '["..."], "transformation": "..."}, "diagnostic_sql": "..."}'
)

SYSTEM_WALKTHROUGH = (
    "Produce a walkthrough of how the suspect value was produced: start at the "
    "final reported table and walk backwards through the transformation layers "
    "to the raw sources, naming every table and join the value flows through, "
    "with one evidence SQL query per layer. Respond ONLY with JSON:\n"
    '{"steps": [{"layer": "L1", "table": "...", "transformation": "...", '
    '"join": "...", "evidence_sql": "SELECT ..."}]}'
)


def make_generator(*, backend: str, base_model: str, adapter: str | None,
                   llama_url: str | None, device: str = "auto",
                   dtype: str = "auto", max_new_tokens: int = 512):
    """Return ``generate(system, user, temperature=0.0) -> str`` for a backend."""
    if backend == "llama.cpp":
        def gen(system: str, user: str, temperature: float = 0.0) -> str:
            return _llama(system, user, max_new_tokens, temperature, llama_url)
        return gen
    if backend != "transformers":
        raise SystemExit(f"unknown backend: {backend!r}")
    return _TransformersGenerator(base_model, adapter, device, dtype,
                                  max_new_tokens).generate


class _TransformersGenerator:
    def __init__(self, base_model: str, adapter: str | None, device: str,
                 dtype: str, max_new_tokens: int):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "bfloat16" if device == "cuda" else "float32"
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        tok = AutoTokenizer.from_pretrained(adapter or base_model,
                                            trust_remote_code=True, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        if device == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=torch_dtype, attn_implementation="sdpa",
                device_map={"": "cpu"}, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=torch_dtype, attn_implementation="sdpa",
                device_map="cuda:0", trust_remote_code=True)
        model.eval()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            model = model.merge_and_unload()
            model.eval()

        self._tok, self._model = tok, model
        self._max_new = max_new_tokens

    def generate(self, system: str, user: str, temperature: float = 0.0) -> str:
        import torch
        tok, model = self._tok, self._model
        do_sample = temperature > 0
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=8192
                  ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=self._max_new,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def _llama(system: str, user: str, max_new_tokens: int, temperature: float,
           llama_url: str | None) -> str:
    if not llama_url:
        raise SystemExit("backend llama.cpp requires --llama_url "
                         "(e.g. http://127.0.0.1:8080)")
    base = llama_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    payload = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 1.0,
        "stream": False,
    }
    r = requests.post(f"{base}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"])


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return __import__("json").loads(m.group(0))
    except Exception:
        return None
