"""Evaluation + baseline comparison on the stratified held-out split.

Three rows are compared:
  * ``oracle_baseline``  — the deterministic oracle alone (ground truth).  It
                           always catches the planted defect and runs clean, so
                           it is the ceiling the model is measured against.
  * ``base_model``       — the untrained small instruct model (same load stack).
  * ``sft_model``        — the QLoRA-SFT model (base + adapter), greedy decode.

Metrics (per test episode, graded by the execution-verified oracle):
  valid_query_rate    — the emitted diagnostic SQL is executable on the DB.
  catch_rate          — the SQL catches >=1 of the planted rows.
  false_positive_rate — the SQL fires on the clean reference DB.
  localization_acc    — table + suspect columns match the planted defect.
  extraction_useful   — the extraction request is present and executable.
  overall_success     — all of: valid, catches, precise, localization correct.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vendor.defect_catalog import DEFECTS_BY_ID
from vendor.generate_db import build_clean_conn, build_trap

# Lazy imports: loading the model stack is heavy and only needed for the two
# model rows, not for the oracle baseline.
from slm import oracle


def parse_json(text: str) -> dict | None:
    import re
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def grade_generated(defect, generated: str, trap, clean, planted_pks) -> dict:
    """Grade a generated response using already-open trap/clean connections."""
    parsed = parse_json(generated)
    diag_sql = (parsed.get("diagnostic_sql") or "").strip() if parsed else ""
    root_cause = (parsed.get("root_cause") or {}) if parsed else {}
    extraction = (parsed.get("extraction_request") or "").strip() if parsed else ""

    res = oracle.grade_response(root_cause, diag_sql, defect, clean, trap, planted_pks)
    sql = res["sql"]
    if extraction:
        st, _ = oracle.exec_sql(trap, extraction)
        extraction_useful = st == "rows"
    else:
        extraction_useful = False

    return {
        "valid": bool(sql["valid"]),
        "catch": bool(sql["catches"]),
        "false_positive": bool(sql["false_positives"]),
        "precise": bool(sql["precise"]),
        "localization": bool(res["localization"]["correct"]),
        "extraction_useful": bool(extraction_useful),
        "success": bool(res["success"]),
        "generated": generated,
    }


def grade_episode(episode: dict, generated: str) -> dict:
    """Grade one held-out episode against its reconstructed trap/clean DBs."""
    defect = DEFECTS_BY_ID[episode["defect_id"]]
    seed = episode["seed"]
    n_rows = episode.get("n_clean_rows", 1000)
    trap, planted = build_trap(defect, n_rows=n_rows, seed=seed, k=1)
    clean = build_clean_conn(n_rows, seed)
    try:
        return grade_generated(defect, generated, trap, clean, planted)
    finally:
        trap.close()
        clean.close()


def run_best_of_n(episodes: list[dict], tok, model, n: int, temperature: float,
                  sample_limit: int | None = None) -> dict:
    """Sample ``n`` responses per episode and keep the highest-reward one.

    The reward is the execution-verifiable ``slm.reward.compute_reward`` so the
    selection is fully deterministic (no LLM judge).  This is exactly the
    best-of-N step the stage-2 GRPO gate is conditioned on.
    """
    from slm.reward import RolloutEnv, compute_reward
    eps = episodes[:sample_limit] if sample_limit else episodes
    results = []
    for ep in eps:
        defect = DEFECTS_BY_ID[ep["defect_id"]]
        env = RolloutEnv(ep["defect_id"], seed=ep["seed"],
                         n_rows=ep.get("n_clean_rows", 1000))
        try:
            best, best_r = None, -1.0
            for _ in range(n):
                g = generate(tok, model, ep["system"], ep["user"],
                             do_sample=True, temperature=temperature)
                r = compute_reward(env, g)["reward"]
                if r > best_r:
                    best_r, best = r, g
            results.append(grade_generated(defect, best, env.trap, env.clean,
                                           env.planted_pks))
        finally:
            env.close()
    return {"results": results, "n": len(eps), "best_of_n": n}


def grade_baseline(episode: dict) -> dict:
    """Deterministic oracle baseline: the reference SQL + gold localization."""
    defect = DEFECTS_BY_ID[episode["defect_id"]]
    return {
        "valid": True, "catch": True, "false_positive": False, "precise": True,
        "localization": True, "extraction_useful": True, "success": True,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    mean = lambda k: round(sum(r[k] for r in results) / n, 4) if n else 0.0
    return {
        "n": n,
        "valid_query_rate": mean("valid"),
        "catch_rate": mean("catch"),
        "false_positive_rate": mean("false_positive"),
        "precision_closed": mean("precise"),
        "localization_accuracy": mean("localization"),
        "extraction_useful": mean("extraction_useful"),
        "overall_success": mean("success"),
    }


def load_model(base_model: str, adapter_dir: str | None):
    from slm.train_sft import load_4bit_model, load_tokenizer, CausalLMCollator
    tok = load_tokenizer(base_model)
    model = load_4bit_model(base_model)
    if adapter_dir and adapter_dir not in ("none", "None", ""):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload() if False else model  # keep LoRA active
    model.eval()
    model.to("cuda")
    return tok, model


def generate(tok, model, system: str, user: str, *, do_sample=False,
             max_new_tokens: int = 256, temperature: float = 0.7):
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        repetition_penalty=1.05,
    )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    generated_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(generated_tokens, skip_special_tokens=True)


def run_split(episodes: list[dict], tok, model, do_sample=False) -> list[dict]:
    results = []
    for ep in episodes:
        gen = generate(tok, model, ep["system"], ep["user"], do_sample=do_sample)
        results.append(grade_episode(ep, gen))
    return results


def evaluate(test_jsonl: str, base_model: str, adapter_dir: str | None,
             sample_limit: int | None = None) -> list[dict]:
    eps = [json.loads(l) for l in open(test_jsonl, encoding="utf-8") if l.strip()]
    eps = [e for e in eps if e["split"] == "test"]
    if sample_limit:
        eps = eps[:sample_limit]
    baseline = [grade_baseline(e) for e in eps]
    tok, model = load_model(base_model, adapter_dir)
    try:
        model_res = run_split(eps, tok, model, do_sample=False)
    finally:
        del model
        torch.cuda.empty_cache()
    return {"episodes": eps, "baseline": baseline, "model": model_res}


def main():
    ap = argparse.ArgumentParser(description="Evaluate an SFT adapter vs baselines.")
    ap.add_argument("--test_jsonl", default="data/generated/sft.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (None => base model)")
    ap.add_argument("--sample_limit", type=int, default=None)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--best_of_n", type=int, default=0,
                    help="if >0, measure best-of-N selection (greedy eval skipped)")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8") if l.strip()]
    eps = [e for e in eps if e["split"] == "test"]
    if args.sample_limit:
        eps = eps[: args.sample_limit]

    baseline = summarize([grade_baseline(e) for e in eps])
    tok, model = load_model(args.base_model, args.adapter)
    try:
        if args.best_of_n > 0:
            bon = run_best_of_n(eps, tok, model, args.best_of_n, args.temperature)
            out = {
                "n": len(eps),
                "base_model": args.base_model,
                "adapter": args.adapter,
                "oracle_baseline": baseline,
                "best_of_n": summarize(bon["results"]),
                "best_of_n_k": args.best_of_n,
            }
        else:
            mod = summarize(run_split(eps, tok, model))
            out = {
                "n": len(eps),
                "base_model": args.base_model,
                "adapter": args.adapter,
                "oracle_baseline": baseline,
                "model": mod,
            }
    finally:
        del model
        torch.cuda.empty_cache()

    print(json.dumps(out, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
