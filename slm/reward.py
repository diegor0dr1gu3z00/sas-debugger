"""Stage-2 GRPO reward function (prepared, GATED — not launched).

The SFT trajectory (deliverable #4) must be the gate: GRPO runs only if SFT +
best-of-N plateaus BELOW target with a reward that is >= 0.9 execution-
verifiable (see TRAINING_REPORT.md).  The reward below is deliberately
*execution-verifiable*: every component is computed by running the model's
emitted SQL on the trap/clean databases, so the reward never depends on an
LLM judge.

Design
------
``compute_reward(env, output)`` returns a bounded scalar in [0, 1]:
    0.40 * sql_catch      (diagnostic SQL, run on trap, catches >=1 planted row)
    0.20 * sql_precision  (same SQL returns 0 rows on the clean DB)
    0.20 * localization   (table + suspect columns match the planted defect)
    0.20 * valid_output   (response parses as JSON with the required keys)
The scalar is fully determinable from the environment (ground truth is built-in).
"""

from __future__ import annotations

import json
import re
from typing import Any

from vendor.defect_catalog import DEFECTS_BY_ID
from vendor.generate_db import build_clean_conn, build_trap

from slm import oracle

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

REWARD_WEIGHTS = {
    "sql_catch": 0.40,
    "sql_precision": 0.20,
    "localization": 0.20,
    "valid_output": 0.20,
}


class RolloutEnv:
    """A GRPO rollout environment: one planted defect + its clean/trap DBs."""

    def __init__(self, defect_id: str, seed: int, n_rows: int = 1500):
        self.defect = DEFECTS_BY_ID[defect_id]
        self.seed = seed
        self.n_rows = n_rows
        self.trap, self.planted_pks = build_trap(
            self.defect, n_rows=n_rows, seed=seed, k=1)
        self.clean = build_clean_conn(n_rows, seed)

    def close(self) -> None:
        self.trap.close()
        self.clean.close()


def parse_output(text: str) -> dict[str, Any] | None:
    """Extract the JSON object from a generated response; None on parse fail."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def compute_reward(env: RolloutEnv, output: str) -> dict[str, Any]:
    """Bounded, execution-verifiable scalar reward plus explainable pieces."""
    parsed = parse_output(output)
    diag_sql = (parsed.get("diagnostic_sql") or "").strip() if parsed else ""
    root_cause = (parsed.get("root_cause") or {}) if parsed else {}
    has_sql = bool(diag_sql)
    has_loc = bool((root_cause.get("columns") or []))
    valid_output = 1.0 if (parsed is not None and has_sql and has_loc) else 0.0

    sql_grade = (oracle.grade_sql(env.defect, diag_sql, env.clean, env.trap,
                                  env.planted_pks) if has_sql else
                 {"valid": False, "catches": False, "false_positives": True,
                  "precise": False})
    loc_grade = oracle.grade_localization(root_cause, env.defect)

    if sql_grade.get("catches"):
        sql_catch = 1.0
    elif sql_grade.get("valid"):
        sql_catch = 0.30   # runs but catches nothing — partial progress signal
    else:
        sql_catch = 0.0

    if sql_grade.get("valid") and not sql_grade.get("false_positives"):
        sql_precision = 1.0
    else:
        sql_precision = 0.0

    comp = {
        "valid_output": valid_output,
        "sql_catch": sql_catch,
        "sql_precision": sql_precision,
        "localization": 1.0 if loc_grade["correct"] else 0.0,
        "sql_valid": bool(sql_grade.get("valid")),
        "sql_error": sql_grade.get("error", ""),
    }
    reward = sum(REWARD_WEIGHTS[k] * comp[k] for k in REWARD_WEIGHTS)
    return {"reward": round(float(reward), 4), "components": comp,
            "sql_grade": sql_grade, "loc_grade": loc_grade}


def verify_reward_sanity() -> dict[str, Any]:
    """Self-check: gold trace => reward ~= 1.0; empty/random => ~0.0."""
    from vendor.generate_db import PANEL_TABLE
    gold, junk, empty = 0.0, 0.0, 0.0
    for defect_id in ("D01", "D14", "D47", "D59", "DA"):
        env = RolloutEnv(defect_id, seed=7)
        d = env.defect
        gold_trace = {
            "diagnostic_sql": d.oracle_sql,
            "root_cause": oracle.defect_root_cause(d),
            "extraction_request": "",
        }
        gold += compute_reward(env, json.dumps(gold_trace))["reward"]
        junk += compute_reward(env, "not json at all")["reward"]
        empty += compute_reward(env, json.dumps({"diagnostic_sql": "",
                                                 "root_cause": {}}))["reward"]
        env.close()
    n = 5
    return {"mean_gold": round(gold / n, 3), "mean_junk": round(junk / n, 3),
            "mean_empty": round(empty / n, 3)}
