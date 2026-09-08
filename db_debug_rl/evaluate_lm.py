"""Benchmark an LM on db-debug-rl external episodes (execution-verified).

For every held-out episode the model must find the implicated diagnostic query
and root cause: its JSON response is parsed and graded by the mutation oracle
(valid = SQL runs; catch = catches the planted rows on the trap DB; FP = fires
on the clean DB; loc = table + primary suspect column named; success = all of
them). A returned walkthrough is verified step by step as a secondary metric.

Usage:
    .venv/bin/python -m db_debug_rl.evaluate_lm \
        --test_jsonl data/generated/external_episodes.jsonl \
        --base_model Qwen/Qwen2.5-0.5B-Instruct \
        [--adapter checkpoints/sft-0.5b-ext/lora] \
        --json_out eval/external-benchmark.json [--tag name]
        [--sample_limit N] [--best_of_n 4 --bos_sample_limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from db_debug_rl.pipelines import PIPELINES
from db_debug_rl.defects import DEFECTS_BY_ID
from db_debug_rl import oracle as orc
from db_debug_rl import walkthrough as wt
from db_debug_rl.reward import parse_output


def load_model(base_model: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(
        adapter or base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = model.merge_and_unload()
        model.eval()
    return model, tok


def generate(model, tok, system: str, user: str, max_new_tokens: int = 640,
             temperature: float = 0.0) -> str:
    do_sample = temperature > 0
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(
        model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def grade_text(env: orc.EnvData, text: str) -> dict[str, Any]:
    parsed = parse_output(text)
    diag = str(parsed.get("diagnostic_sql") or "") if parsed else ""
    rc = _as_dict(parsed.get("root_cause")) if parsed else {}
    verdict = orc.grade_response(env, rc, diag)
    w_parsed = wt.parse_walkthrough(text)
    w_score = 0.0
    if w_parsed:
        wv = wt.verify_walkthrough(env.pipeline, w_parsed, env, env.suspect_pk)
        w_score = wv["score"]
    return {"valid": bool(verdict["sql"]["valid"]),
            "catch": bool(verdict["sql"]["catches"]),
            "fp": bool(verdict["sql"]["false_positives"]),
            "precise": bool(verdict["sql"]["precise"]),
            "loc": bool(verdict["localization"]["correct"]),
            "success": bool(verdict["success"]),
            "walkthrough": w_score,
            "raw": text}


def pick_best(env: orc.EnvData, texts: list[str]) -> dict[str, Any]:
    graded = [grade_text(env, t) for t in texts]
    order = ("success", "precise", "catch", "valid")
    return max(graded, key=lambda g: tuple(1 if g[k] else 0 for k in order)
               + (g["walkthrough"],))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", default="data/generated/external_episodes.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--json_out", default="eval/external-benchmark.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sample_limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--best_of_n", type=int, default=0)
    ap.add_argument("--bos_sample_limit", type=int, default=12)
    args = ap.parse_args()

    episodes = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8")
                if l.strip()]
    test = [e for e in episodes if e["split"] == "test"]
    if args.sample_limit:
        test = test[:args.sample_limit]
    if not test:
        raise SystemExit("no test episodes in the jsonl")

    model, tok = load_model(args.base_model, args.adapter)
    report: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter": args.adapter or "-",
        "tag": args.tag or (Path(args.adapter).parent.name if args.adapter else "base"),
        "n_test": len(test), "rows": [], "wall_s": 0.0}
    t_all = time.time()
    agg: dict[str, float] = {}
    n = len(test)
    for i, ep in enumerate(test):
        t0 = time.time()
        pipe = PIPELINES[ep["pipeline_id"]]
        defect = DEFECTS_BY_ID[ep["defect_id"]]
        env = orc.build_env_data(pipe, defect, seed=ep["seed"], k_planted=3)
        try:
            if env.planted_pks != ep["planted_pks"]:
                raise RuntimeError(f"{ep['episode_id']}: planted pks drifted")
            if args.best_of_n and i < args.bos_sample_limit:
                texts = [generate(model, tok, ep["system"], ep["user"],
                                  args.max_new_tokens, 0.7)
                         for _ in range(args.best_of_n)]
                g = pick_best(env, texts)
            else:
                g = grade_text(env, generate(model, tok, ep["system"], ep["user"],
                                             args.max_new_tokens, 0.0))
        finally:
            env.close()
        for k in ("valid", "catch", "fp", "precise", "loc", "success"):
            agg[k] = agg.get(k, 0.0) + (1.0 if g[k] else 0.0)
        agg["walkthrough"] = agg.get("walkthrough", 0.0) + g["walkthrough"]
        row = {"episode_id": ep["episode_id"], "defect_id": ep["defect_id"],
               "db": ep["db"],
               **{k: g[k] for k in ("valid", "catch", "fp", "precise", "loc",
                                    "success", "walkthrough")},
               "latency_s": round(time.time() - t0, 1)}
        report["rows"].append(row)
        print(f"  [{i+1}/{n}] {ep['defect_id']:6s} {ep['db']:14s} "
              f"success={g['success']} walk={g['walkthrough']:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    report["metrics"] = {k: round(v / n, 4) for k, v in agg.items()}
    report["metrics"]["n"] = n
    report["wall_s"] = round(time.time() - t_all, 1)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["metrics"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
