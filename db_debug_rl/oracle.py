"""Deterministic grading oracle for db-debug-rl (external databases).

Builds the clean/trap in-memory working DBs for a (pipeline, defect) pair,
plants the discrepancy, and grades candidate diagnostic SQL + localisations
exactly like ``slm.oracle``: the diagnostic must run on the trap DB, catch at
least one planted row, return 0 rows on the clean DB, and the localisation
must name the true table and the primary suspect column.
"""
from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from db_debug_rl.db import open_db
from db_debug_rl.pipelines import Pipeline, PIPELINES
from db_debug_rl.defects import Defect

SQL_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "sql"
TRUTH = "zz_truth"
_TRUTH_RE = re.compile(r"\bzz_\w*", re.IGNORECASE)
_EPS = 0.011


def recompute_diff_sql(pipeline: Pipeline, col: str, numeric: bool) -> str:
    """The agent-writable reconciliation diff: final table vs clean recomputation."""
    f, pk = pipeline.final_table, pipeline.pk
    neq = f'f."{col}" IS NOT t."{col}"'
    if numeric:
        neq += (f' OR (f."{col}" IS NOT NULL AND t."{col}" IS NOT NULL AND '
                f'ABS(CAST(f."{col}" AS REAL) - CAST(t."{col}" AS REAL)) > {_EPS})')
    return (
        f'SELECT CAST(f."{pk}" AS TEXT) AS flagged_pk FROM "{f}" f '
        f'LEFT JOIN ({pipeline.final_body}) t '
        f'ON t."{pk}" IS f."{pk}" '
        f'WHERE t."{pk}" IS NULL OR {neq} '
        f'UNION ALL '
        f'SELECT CAST(t."{pk}" AS TEXT) FROM ({pipeline.final_body}) t '
        f'LEFT JOIN "{f}" f ON f."{pk}" IS t."{pk}" '
        f'WHERE f."{pk}" IS NULL')


def _connect_memory() -> sqlite3.Connection:
    return sqlite3.connect(":memory:", uri=True)


def _materialize(pipeline: Pipeline) -> sqlite3.Connection:
    """Materialize every pipeline step into an in-memory working DB.

    The source database stays attached as ``src``; staging tables land in main.
    A hidden ``zz_truth`` table holds the clean recomputation of the final
    table (oracle-only; the agent is blocked from referencing it).
    """
    conn = _connect_memory()
    conn.execute(f"ATTACH DATABASE 'file:{SQL_DIR / (pipeline.db + '.db')}?mode=ro' AS src")
    for step in pipeline.steps[:-1]:
        conn.execute(f'CREATE TABLE "{step.name}" AS {step.body}')
    for ix in pipeline.indexes:
        conn.execute(ix)
    conn.execute(f'CREATE TABLE "{pipeline.final_table}" AS {pipeline.final_body}')
    conn.execute(f'CREATE INDEX ix_final_pk ON "{pipeline.final_table}"("{pipeline.pk}")')
    conn.execute(f'CREATE TABLE {TRUTH} AS {pipeline.final_body}')
    conn.execute(f'CREATE INDEX ix_truth_pk ON {TRUTH}("{pipeline.pk}")')
    return conn


class PipelineCache:
    """Bounded LRU of clean materializations (one per pipeline).

    In-memory DBs for the heavy pipelines approach ~1.5 GB, so the cache keeps
    at most ``maxsize`` connections and evicts the least recently used.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self._cache: dict[str, sqlite3.Connection] = {}
        self._order: list[str] = []
        self.maxsize = maxsize

    def get(self, pipeline: Pipeline) -> sqlite3.Connection:
        pid = pipeline.pipeline_id
        if pid in self._cache:
            self._order.remove(pid)
            self._order.append(pid)
            return self._cache[pid]
        while len(self._cache) >= self.maxsize:
            old = self._order.pop(0)
            self._cache.pop(old).close()
        conn = _materialize(pipeline)
        self._cache[pid] = conn
        self._order.append(pid)
        return conn

    def close_all(self) -> None:
        for conn in self._cache.values():
            conn.close()
        self._cache.clear()
        self._order.clear()


_DEFAULT_CACHE = PipelineCache()


def _trap_from(base: sqlite3.Connection, pipeline: Pipeline,
               defect: Defect) -> sqlite3.Connection:
    trap = _connect_memory()
    base.backup(trap)
    trap.execute(f"ATTACH DATABASE 'file:{SQL_DIR / (pipeline.db + '.db')}?mode=ro' AS src")
    trap.execute(f'DROP TABLE "{pipeline.final_table}"')
    trap.execute(f'CREATE TABLE "{pipeline.final_table}" AS {defect.trap_sql}')
    trap.execute(f'CREATE INDEX ix_final_pk ON "{pipeline.final_table}"("{pipeline.pk}")')
    return trap


@dataclass
class EnvData:
    """One planted debugging environment (clean + trap + ground truth).

    ``clean`` may be a SHARED cached connection (PipelineCache); only ``trap``
    is private to the episode. Closing an EnvData never closes ``clean``.
    """
    pipeline: Pipeline
    defect: Defect
    clean: sqlite3.Connection
    trap: sqlite3.Connection
    planted_pks: list[str]
    suspect_pk: str

    def close(self) -> None:
        self.trap.close()


def build_env_data(pipeline: Pipeline, defect: Defect, seed: int = 0,
                   k_planted: int = 3, cache: PipelineCache | None = None) -> EnvData:
    """Build one planted debugging environment.

    Planted rows are sampled from the GOLD diagnostic's own result stream, so
    the gold trace always catches them. ``cache`` lets callers share the
    (expensive) clean materialization across defects of the same pipeline.
    """
    if defect.pipeline_id != pipeline.pipeline_id:
        raise ValueError(f"{defect.defect_id} does not belong to {pipeline.pipeline_id}")
    cache = cache or _DEFAULT_CACHE
    clean = cache.get(pipeline)
    trap = _trap_from(clean, pipeline, defect)
    gold_stream = [str(r[0]) for r in trap.execute(
        gold_diagnostic_sql(defect)).fetchmany(200)]
    flagged_n = trap.execute(
        f'SELECT COUNT(*) FROM ({defect.oracle_sql})').fetchone()[0]
    if not gold_stream or flagged_n < 1:
        trap.close()
        raise RuntimeError(
            f"{defect.defect_id}: mutation produced no flagged rows — the defect "
            "does not bite (check the trap SQL)")
    rng = random.Random(seed)
    pool = gold_stream[:50]
    planted = pool if len(pool) <= k_planted else rng.sample(pool, k_planted)
    suspect_pk = planted[0]
    return EnvData(pipeline, defect, clean, trap, planted, suspect_pk)


def gold_diagnostic_sql(defect: Defect) -> str:
    """The gold diagnostic: the short agent-writable invariant for the defect.

    Row-level invariants on the reported table's corroborating columns where
    possible, otherwise a small upstream recomputation/anti-join (see
    ``defects.GOLD_SQL``). Verified by the selftest to catch the planted rows
    on the trap DB and return 0 rows on the clean DB.
    """
    return defect.gold_sql


def suspect_row(env: EnvData) -> dict[str, Any]:
    """The suspect row as observed by the captain.

    For value-drift defects it comes from the trap (reported) table. For
    missing-row defects (inner-vs-left / dropped filters) the row is absent
    from the reported table, so it comes from the reference recomputation —
    which is exactly what the captain sees when reconciling.
    """
    pk, table = env.pipeline.pk, env.pipeline.final_table
    cur = env.trap.execute(f'SELECT * FROM "{table}" WHERE CAST("{pk}" AS TEXT) = ?',
                           (str(env.suspect_pk),))
    row = cur.fetchone()
    if row is not None:
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    cur = env.clean.execute(f'SELECT * FROM {TRUTH} WHERE CAST("{pk}" AS TEXT) = ?',
                            (str(env.suspect_pk),))
    row = cur.fetchone()
    if row is None:
        return {}
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def row_is_missing(env: EnvData) -> bool:
    """True when the suspect row is absent from the trap's reported table."""
    pk, table = env.pipeline.pk, env.pipeline.final_table
    cur = env.trap.execute(f'SELECT 1 FROM "{table}" WHERE CAST("{pk}" AS TEXT) = ?',
                           (str(env.suspect_pk),))
    return cur.fetchone() is None


def grade_sql(env: EnvData, sql: str) -> dict[str, Any]:
    """Execution-verified grade of a candidate diagnostic SQL query."""
    rule: dict[str, Any] = {"valid": False, "catches": False, "false_positives": False,
                            "precise": False, "n_caught": 0, "n_clean": 0,
                            "error": "", "uses_truth": False, "catches_planted": False}
    if not sql or not sql.strip():
        rule["error"] = "empty query"
        return rule
    if _TRUTH_RE.search(sql):
        rule["error"] = "query must not reference the hidden truth tables"
        rule["uses_truth"] = True
        return rule
    from db_debug_rl.db import exec_sql
    status_t, res_t = exec_sql(env.trap, sql, max_rows=50, timeout_s=120)
    if status_t == "err":
        rule["error"] = res_t
        return rule
    rule["valid"] = True
    caught = {str(r[0]) for r in res_t["rows"] if r and r[0] is not None}
    rule["n_caught"] = len(caught)
    rule["catches_planted"] = any(p in caught for p in env.planted_pks)
    rule["catches"] = rule["catches_planted"]
    status_c, res_c = exec_sql(env.clean, sql, max_rows=50, timeout_s=120)
    if status_c == "rows":
        rule["n_clean"] = len(res_c["rows"])
        rule["false_positives"] = rule["n_clean"] > 0
    else:
        rule["false_positives"] = True
    rule["precise"] = rule["catches"] and not rule["false_positives"]
    return rule


def grade_localization(localization: dict[str, Any], defect: Defect) -> dict[str, Any]:
    if isinstance(localization, str):
        try:
            import json as _json
            parsed = _json.loads(localization)
            localization = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            localization = {}
    target_cols = {c.lower() for c in defect.columns}
    model_cols = {str(c).lower() for c in (localization.get("columns") or [])}
    model_table = (localization.get("table") or "").lower().strip()
    table_ok = model_table == PIPELINES[defect.pipeline_id].final_table.lower()
    overlap = target_cols & model_cols
    primary_ok = defect.columns[0].lower() in model_cols
    col_ok = len(overlap) >= 1
    correct = table_ok and col_ok and primary_ok
    return {"correct": bool(correct), "table_ok": bool(table_ok),
            "columns_correct": bool(col_ok), "primary_identified": bool(primary_ok),
            "overlap": sorted(overlap), "model_table": model_table,
            "expected_table": PIPELINES[defect.pipeline_id].final_table}


def grade_response(env: EnvData, localization: dict[str, Any], sql: str) -> dict[str, Any]:
    sql_grade = grade_sql(env, sql)
    loc_grade = grade_localization(localization, env.defect)
    return {"sql": sql_grade, "localization": loc_grade,
            "success": bool(sql_grade["valid"] and sql_grade["catches"]
                            and not sql_grade["false_positives"] and loc_grade["correct"])}
