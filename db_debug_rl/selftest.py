"""Self-test for db-debug-rl: verifies the mutation/oracle contract on every
external pipeline x defect, drives the multi-turn environment with a scripted
debugging policy, and validates gold walkthroughs. Prints a JSON report;
non-zero exit on any failure.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

from db_debug_rl.pipelines import PIPELINES
from db_debug_rl.defects import DEFECTS
from db_debug_rl import oracle as orc
from db_debug_rl import walkthrough as wt
from db_debug_rl.env import DbgRLEnv, EpisodeSpec
from db_debug_rl.pipelines import Pipeline
from db_debug_rl.defects import Defect


def _scripted_policy(pipe: Pipeline, defect: Defect):
    diag = orc.gold_diagnostic_sql(defect)
    root_cause = {"table": pipe.final_table, "columns": list(defect.columns),
                  "transformation": defect.description}

    def policy(obs: dict, history: list) -> dict:
        if not history:
            return {"action": "sql",
                    "sql": f'SELECT * FROM "{pipe.final_table}" LIMIT 3'}
        if len(history) == 1:
            return {"action": "hypothesis", "table": pipe.final_table,
                    "columns": [defect.columns[0]],
                    "claim": f"{defect.columns[0]} not reproducible upstream"}
        return {"action": "localize", "root_cause": root_cause, "diagnostic_sql": diag}
    return policy


def check_pair(pipe: Pipeline, defect: Defect, seed: int = 11,
               cache: "orc.PipelineCache | None" = None) -> dict[str, Any]:
    out: dict[str, Any] = {"defect_id": defect.defect_id, "pipeline": pipe.pipeline_id,
                           "db": pipe.db, "failures": []}
    t0 = time.time()
    env = orc.build_env_data(pipe, defect, seed=seed, cache=cache)
    try:
        flagged = env.trap.execute(defect.oracle_sql).fetchall()
        clean_flagged = env.clean.execute(defect.oracle_sql).fetchall()
        gold_clean = env.clean.execute(defect.gold_sql).fetchall()
        gold_trap = env.trap.execute(defect.gold_sql).fetchall()
        out["flagged_rows"] = len(flagged)
        out["planted"] = env.planted_pks
        if not flagged:
            out["failures"].append("oracle catches nothing on trap")
        if clean_flagged:
            out["failures"].append(f"oracle fires on clean ({len(clean_flagged)} rows)")
        if gold_clean:
            out["failures"].append(f"gold diag fires on clean ({len(gold_clean)} rows)")
        caught = {str(r[0]) for r in gold_trap}
        if not any(p in caught for p in env.planted_pks):
            out["failures"].append("gold diag misses planted rows")

        gold_sql = orc.gold_diagnostic_sql(defect)
        root_cause = {"table": pipe.final_table, "columns": list(defect.columns),
                      "transformation": defect.description}
        verdict = orc.grade_response(env, root_cause, gold_sql)
        if not verdict["success"]:
            out["failures"].append(f"gold trace fails: {verdict}")

        nondiag = orc.grade_sql(
            env, f"SELECT 'NONDIAG' AS probe FROM \"{pipe.final_table}\" LIMIT 1")
        if nondiag["catches"]:
            out["failures"].append("non-diagnostic lookup caught planted rows")
        if nondiag["precise"]:
            out["failures"].append("non-diagnostic lookup treated as precise")

        spec = EpisodeSpec(pipe.pipeline_id, defect.defect_id, seed=seed)
        rl = DbgRLEnv(spec, provided_env=env)
        rl.reset()
        obs = rl._obs()
        history: list[dict] = []
        success, total, forced = False, 0.0, False
        for _ in range(spec.max_turns):
            action = _scripted_policy(pipe, defect)(obs, history)
            obs, r, done, info = rl.step(action)
            total += r
            history.append({"obs": obs, "action": action})
            if done:
                success = bool(info.get("success"))
                forced = "reason" in info
                break
        out["env_success"] = success
        out["env_reward"] = round(total, 4)
        if not success:
            out["failures"].append("scripted policy failed to succeed")
        if forced:
            out["failures"].append("scripted policy hit a budget guard")

        w = wt.gold_walkthrough(pipe, env.suspect_pk,
                                suspect_missing=orc.row_is_missing(env))
        wv = wt.verify_walkthrough(pipe, w, env, env.suspect_pk)
        out["walkthrough_score"] = wv["score"]
        if wv["score"] < 1.0:
            out["failures"].append(f"gold walkthrough fails: {wv}")

        out["build_s"] = round(time.time() - t0, 2)
        return out
    finally:
        env.close()


def main() -> int:
    from db_debug_rl.defects import defects_for_pipeline
    results, failures = [], []
    t0 = time.time()
    cache = orc.PipelineCache()
    try:
        for pipe in PIPELINES.values():
            for defect in defects_for_pipeline(pipe.pipeline_id):
                t1 = time.time()
                try:
                    res = check_pair(pipe, defect, cache=cache)
                except Exception as exc:  # noqa: BLE001
                    res = {"defect_id": defect.defect_id,
                           "pipeline": pipe.pipeline_id, "db": pipe.db,
                           "failures": [f"exception: {type(exc).__name__}: {exc}"]}
                res.setdefault("build_s", round(time.time() - t1, 2))
                for f in res["failures"]:
                    failures.append(f"{res['defect_id']}: {f}")
                print(f"  {res['defect_id']:6s} {res['pipeline']:20s} "
                      f"{'OK' if not res['failures'] else 'FAIL'} "
                      f"({res.get('build_s', '?')}s)", flush=True)
                results.append(res)
    finally:
        cache.close_all()
    report = {"n_defects": len(DEFECTS), "n_pipelines": len(PIPELINES),
              "failures": failures, "wall_s": round(time.time() - t0, 1),
              "results": results}

    from db_debug_rl.rollout import batch_rollout
    light = [d for d in DEFECTS
             if PIPELINES[d.pipeline_id].weight == "light"][:3]
    specs = [EpisodeSpec(d.pipeline_id, d.defect_id, seed=7) for d in light]
    rolls = [batch_rollout([s], _scripted_policy(PIPELINES[s.pipeline_id],
                                                 next(d for d in light
                                                      if d.defect_id == s.defect_id)))[0]
             for s in specs]
    for spec, roll in zip(specs, rolls):
        ok = roll["success"] and roll["total_reward"] >= 0.99
        if not ok:
            failures.append(f"rollout {spec.defect_id}: success={roll['success']} "
                            f"reward={roll['total_reward']}")
        print(f"  rollout {spec.defect_id:6s} success={roll['success']} "
              f"turns={roll['turns']} reward={roll['total_reward']}")
    report["rollouts_ok"] = not any("rollout" in f for f in failures)
    status = "PASS" if not failures else f"FAIL ({len(failures)})"
    print(f"SELFTEST {status}")
    out_path = "data/generated/db_debug_rl_selftest.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
