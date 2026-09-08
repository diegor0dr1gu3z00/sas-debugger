"""Terminal + shaped reward for db-debug-rl (execution-verifiable).

Same contract as ``slm.reward``: bounded in [0, 1], no LLM judge.
    0.40 * sql_catch      (diagnostic SQL catches >=1 planted row on the trap DB)
    0.20 * sql_precision  (same SQL returns 0 rows on the clean DB)
    0.20 * localization   (table + primary suspect column named)
    0.20 * valid_output   (localisation + diagnostic provided in usable form)
plus a small hypothesis-shaping bonus (<= 0.10) for correct hypotheses formed
during the episode, clamped to [0, 1].
"""
from __future__ import annotations

import json
import re
from typing import Any

from db_debug_rl.oracle import EnvData, grade_sql, grade_localization

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a response; tolerates truncation by
    trimming to the last balanced closing brace."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    candidates = [m.group(0)]
    tail = m.group(0)
    last = tail.rfind("}")
    if last != -1:
        candidates.append(tail[:last + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None

REWARD_WEIGHTS = {
    "sql_catch": 0.40,
    "sql_precision": 0.20,
    "localization": 0.20,
    "valid_output": 0.20,
}
HYPOTHESIS_BONUS_CAP = 0.10


def compute_reward(env: EnvData, localization: dict[str, Any] | None,
                   diagnostic_sql: str | None,
                   n_correct_hypotheses: int = 0) -> dict[str, Any]:
    localization = localization or {}
    diagnostic_sql = (diagnostic_sql or "").strip()
    has_loc = bool(localization.get("table") and localization.get("columns"))
    has_sql = bool(diagnostic_sql)
    valid_output = 1.0 if (has_loc and has_sql) else 0.0

    if has_sql:
        sql_grade = grade_sql(env, diagnostic_sql)
    else:
        sql_grade = {"valid": False, "catches": False, "false_positives": True,
                     "precise": False, "error": "no diagnostic sql"}
    loc_grade = grade_localization(localization, env.defect)

    if sql_grade.get("catches"):
        sql_catch = 1.0
    elif sql_grade.get("valid"):
        sql_catch = 0.30
    else:
        sql_catch = 0.0
    sql_precision = 1.0 if (sql_grade.get("valid")
                            and not sql_grade.get("false_positives")) else 0.0

    comp = {
        "valid_output": valid_output,
        "sql_catch": sql_catch,
        "sql_precision": sql_precision,
        "localization": 1.0 if loc_grade["correct"] else 0.0,
        "sql_valid": bool(sql_grade.get("valid")),
        "sql_error": sql_grade.get("error", ""),
    }
    bonus = min(HYPOTHESIS_BONUS_CAP, 0.05 * max(0, n_correct_hypotheses))
    reward = min(1.0, sum(REWARD_WEIGHTS[k] * comp[k] for k in REWARD_WEIGHTS) + bonus)
    return {"reward": round(float(reward), 4), "components": comp, "bonus": bonus,
            "sql_grade": sql_grade, "loc_grade": loc_grade}
