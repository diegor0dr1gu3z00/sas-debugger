"""db-debug-rl: a multi-turn RL environment that teaches a small LM to debug
discrepancies inside large, real, join-heavy databases.

The agent is given a production-style database (from ``data/external/sql``), a
materialized reporting pipeline (staging -> joins -> aggregation), and ONE
suspect row of the final table whose value disagrees with the reference
recomputation. It must investigate with read-only SQL, form hypotheses, and
finish with a localisation (table / columns / transformation) plus a diagnostic
SQL query that flags every defective row.

Ground truth is correct by construction: we authored the pipeline AND the
defect, so the oracle always knows the true table / column / transformation.
This is the same mutation-testing contract as ``vendor/defect_catalog`` and
``slm/oracle``, generalised from the synthetic bank schema to real databases.
"""
from db_debug_rl.db import (open_db, schema_summary, schema_text, exec_sql,
                            readonly_guard)
from db_debug_rl.pipelines import PIPELINES, Pipeline, Step, get_pipeline
from db_debug_rl.defects import DEFECTS, DEFECTS_BY_ID, Defect, defects_for_pipeline
from db_debug_rl.oracle import EnvData, build_env_data, grade_sql, grade_localization, grade_response
from db_debug_rl.env import DbgRLEnv, EpisodeSpec
from db_debug_rl.reward import compute_reward

__all__ = [
    "PIPELINES", "Pipeline", "Step", "get_pipeline",
    "DEFECTS", "DEFECTS_BY_ID", "Defect", "defects_for_pipeline",
    "EnvData", "build_env_data", "grade_sql", "grade_localization", "grade_response",
    "DbgRLEnv", "EpisodeSpec", "compute_reward",
    "open_db", "schema_summary", "schema_text", "exec_sql", "readonly_guard",
]
