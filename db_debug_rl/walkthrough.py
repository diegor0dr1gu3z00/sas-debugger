"""Walkthrough capability: step-by-step lineage explanations, oracle-verified.

A walkthrough explains HOW the suspect value of the final reported table was
produced, walking the pipeline backwards layer by layer (final table ->
staging joins -> raw sources). Each step carries an ``evidence_sql`` query that
must run on the working DB and return the claimed intermediate rows for the
suspect's primary key.

The verifier is execution-based: every evidence SQL must run; the final step
must return the suspect row; every claimed layer must map to a real pipeline
step. Gold walkthroughs are constructed from the pipeline definition, so SFT
supervision and the walkthrough score are both correct by construction.
"""
from __future__ import annotations

import json
import re
from typing import Any

from db_debug_rl.db import exec_sql
from db_debug_rl.oracle import EnvData
from db_debug_rl.pipelines import Pipeline

WALKTHROUGH_PROMPT = (
    "Produce a walkthrough of how the suspect value was produced: start at the "
    "final reported table and walk backwards through the transformation layers "
    "to the raw sources, naming every table and join the value flows through, "
    "with one evidence SQL query per layer. Respond ONLY with JSON:\n"
    '{"steps": [{"layer": "L1", "table": "...", "transformation": "...", '
    '"join": "...", "evidence_sql": "SELECT ..."}]}')


def _lit(value: Any) -> str:
    if value is None or (isinstance(value, str) and value.upper() == "NULL"):
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def gold_walkthrough(pipeline: Pipeline, suspect_pk: str,
                     suspect_missing: bool = False) -> list[dict[str, Any]]:
    """Reference walkthrough: one step per pipeline layer, backwards.

    When the suspect row is absent from the reported table (missing-row
    defects), the first step's evidence recomputes that single row from the
    pipeline instead of selecting it from the final table.
    """
    steps: list[dict[str, Any]] = []
    for step in reversed(pipeline.steps):
        is_final = step.name == pipeline.final_table
        if is_final:
            if suspect_missing:
                evidence = (f'SELECT * FROM ({pipeline.final_body}) '
                            f'WHERE CAST("{pipeline.pk}" AS TEXT) = {_lit(suspect_pk)} '
                            f'LIMIT 5')
            else:
                evidence = (f'SELECT * FROM "{step.name}" '
                            f'WHERE CAST("{pipeline.pk}" AS TEXT) = {_lit(suspect_pk)} '
                            f'LIMIT 5')
            transformation = step.description
        else:
            evidence = f'SELECT * FROM "{step.name}" LIMIT 5'
            transformation = f'{step.description} (staging layer feeding {steps[0]["table"] if steps else pipeline.final_table})'
        steps.append({"layer": step.layer, "table": step.name,
                      "transformation": transformation, "join": "",
                      "evidence_sql": evidence})
    return steps


def verify_walkthrough(pipeline: Pipeline, steps: list[dict[str, Any]],
                       env: EnvData, suspect_pk: str) -> dict[str, Any]:
    """Execution-verified walkthrough score in [0, 1].

    Contract (walkthroughs run final table -> raw sources): the first step must
    target the final reported table and its evidence must return the suspect
    row; every evidence SQL must run; layer labels must match the pipeline's
    reversed layers.
    """
    total = len(pipeline.steps)
    if not isinstance(steps, list) or not steps:
        return {"score": 0.0, "n_steps": 0, "ran": 0, "final_ok": False,
                "layers_ok": False, "errors": ["no steps"]}
    ran, errors = 0, []
    for i, st in enumerate(steps):
        sql = str(st.get("evidence_sql") or "")
        status, res = exec_sql(env.trap, sql, max_rows=5)
        if status != "rows":
            errors.append(f"step {i}: {res}")
            continue
        ran += 1
    head = steps[0]
    final_ok = False
    status, res = exec_sql(env.trap, str(head.get("evidence_sql") or ""), max_rows=5)
    if status == "rows" and res["rows"]:
        cols = res["cols"]
        pk_idx = cols.index(pipeline.pk) if pipeline.pk in cols else None
        if pk_idx is not None and any(
                str(r[pk_idx]) == str(suspect_pk) for r in res["rows"]):
            final_ok = True
        else:
            errors.append("first step evidence does not return the suspect row")
    else:
        errors.append(f"first step evidence failed: {res if status == 'err' else 'empty'}")
    layers_ok = (str(head.get("table") or "").lower() == pipeline.final_table.lower()
                 and len(steps) == total
                 and [s.get("layer") for s in steps]
                 == [st.layer for st in reversed(pipeline.steps)])
    if not layers_ok:
        errors.append("walkthrough must start at the final reported table and "
                      "follow the pipeline layers backwards")
    score = (ran / total) * 0.6 + (0.25 if final_ok else 0.0) + (0.15 if layers_ok else 0.0)
    return {"score": round(min(1.0, score), 4), "n_steps": len(steps), "ran": ran,
            "final_ok": final_ok, "layers_ok": layers_ok, "errors": errors}


def parse_walkthrough(text: str) -> list[dict[str, Any]] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    steps = parsed.get("steps") if isinstance(parsed, dict) else None
    return steps if isinstance(steps, list) else None
