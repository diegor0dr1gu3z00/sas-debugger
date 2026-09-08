"""Episode + SFT-data generation over the external production databases.

An episode bundles (a) the suspect row from the trap pipeline, (b) schema +
lineage context, (c) the gold multi-turn trace: hypotheses -> diagnostic SQL ->
localisation, plus a lineage walkthrough. Every stored episode is
oracle-verified (the gold trace grades success=True with reward >= 0.99 and the
walkthrough verifies to 1.0), so the dataset is execution-confirmed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from db_debug_rl.pipelines import PIPELINES
from db_debug_rl.defects import DEFECTS, Defect
from db_debug_rl import oracle as orc
from db_debug_rl import walkthrough as wt

SCHEMA_VERSION = "db-debug-rl-episode/v2"


def _row_text(row: dict[str, Any], max_fields: int = 18) -> str:
    items = [f"{k}={v}" for k, v in list(row.items())[:max_fields]]
    return " | ".join(items)


def gold_trace(defect: Defect, env: orc.EnvData) -> dict[str, Any]:
    pipe = PIPELINES[defect.pipeline_id]
    transformation = f"{defect.category} rule ({pipe.final_table}): {defect.description}"
    reason = (f"recomputing the metric from the source tables disagrees with "
              f"{pipe.final_table}.{defect.columns[0]} for the suspect row")
    hypotheses = [
        {"table": pipe.final_table, "columns": [defect.columns[0]],
         "claim": f"{defect.columns[0]} is not reproducible from the upstream joins"},
    ]
    trace = {
        "hypotheses": hypotheses,
        "root_cause": {"table": pipe.final_table, "columns": list(defect.columns),
                       "transformation": transformation, "reason": reason},
        "diagnostic_sql": orc.gold_diagnostic_sql(defect),
    }
    return trace


def gold_user_text(env: orc.EnvData) -> str:
    from db_debug_rl.db import schema_summary, schema_text
    pipe = env.pipeline
    row = orc.suspect_row(env)
    return (
        f"{schema_text(schema_summary(env.trap))}\n\n"
        f"PIPELINE LINEAGE ({pipe.final_table} is the reported table)\n{pipe.lineage}\n\n"
        f"SUSPECT ROW\n  {pipe.pk} = {env.suspect_pk}\n  {_row_text(row)}\n\n"
        "TASK\n  Investigate with read-only SQL, state your hypothesis, then "
        "localise the root cause (table, columns, transformation) and print the "
        "diagnostic SQL as JSON.")


def build_episode(defect: Defect, seed: int, split: str,
                  episode_id: str, k_planted: int = 3,
                  cache: "orc.PipelineCache | None" = None) -> dict[str, Any] | None:
    pipe = PIPELINES[defect.pipeline_id]
    env = orc.build_env_data(pipe, defect, seed=seed, k_planted=k_planted, cache=cache)
    try:
        trace = gold_trace(defect, env)
        verdict = orc.grade_response(env, trace["root_cause"], trace["diagnostic_sql"])
        w_steps = wt.gold_walkthrough(pipe, env.suspect_pk,
                                      suspect_missing=orc.row_is_missing(env))
        w_verdict = wt.verify_walkthrough(pipe, w_steps, env, env.suspect_pk)
        if not verdict["success"] or w_verdict["score"] < 1.0:
            return None
        user = gold_user_text(env)
        return {
            "schema": SCHEMA_VERSION,
            "episode_id": episode_id,
            "db": pipe.db,
            "pipeline_id": pipe.pipeline_id,
            "defect_id": defect.defect_id,
            "defect_kind": defect.kind,
            "dimension": defect.dimension,
            "category": defect.category,
            "severity": defect.severity,
            "split": split,
            "seed": seed,
            "planted_pks": env.planted_pks,
            "system": _system(),
            "user": user,
            "assistant": json.dumps({**trace, "walkthrough": {"steps": w_steps}},
                                    ensure_ascii=False),
            "gold": {**trace, "walkthrough": {"steps": w_steps}},
            "oracle": {"response": verdict, "walkthrough": w_verdict},
        }
    finally:
        env.close()


def _system() -> str:
    from db_debug_rl.env import SYSTEM_PROMPT
    from db_debug_rl.walkthrough import WALKTHROUGH_PROMPT
    return SYSTEM_PROMPT + "\n\nWhen asked for a walkthrough, use this contract:\n" + WALKTHROUGH_PROMPT


def generate_episodes(defects: list[Defect] | None = None, *, seeds_per_defect: int = 6,
                      test_frac: float = 0.2, seed0: int = 9001) -> list[dict[str, Any]]:
    import math
    defects = defects if defects is not None else DEFECTS
    episodes: list[dict[str, Any]] = []
    cache = orc.PipelineCache()
    try:
        for di, defect in enumerate(defects):
            n_test = max(1, math.ceil(seeds_per_defect * test_frac))
            for j in range(seeds_per_defect):
                seed = seed0 + di * 10_007 + j * 911
                split = "test" if j < n_test else "train"
                ep = build_episode(defect, seed, split, f"{defect.defect_id}_{j:03d}",
                                   cache=cache)
                if ep is not None:
                    episodes.append(ep)
                else:
                    print(f"  ! {defect.defect_id} seed={seed} failed verification",
                          file=sys.stderr)
    finally:
        cache.close_all()
    return episodes


def write_jsonl(eps: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_stats(eps: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    splits = Counter(e["split"] for e in eps)
    per_db = Counter(e["db"] for e in eps)
    per_kind = Counter(e["defect_kind"] for e in eps)
    return {"total": len(eps), "splits": dict(splits),
            "test_frac": round(splits.get("test", 0) / max(1, len(eps)), 4),
            "databases": dict(sorted(per_db.items())),
            "defect_kinds": dict(sorted(per_kind.items()))}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path("data/generated/external_episodes.jsonl")
    eps = generate_episodes(seeds_per_defect=n)
    write_jsonl(eps, out)
    print(json.dumps(split_stats(eps), indent=2))
    print(f"wrote {len(eps)} episodes -> {out}")
