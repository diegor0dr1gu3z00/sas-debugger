"""Deterministic grading oracle for the SAS-reconcile debugging task.

Ground truth is correct *by construction*: we inject a single known defect into
a synthetic PostgreSQL/SQLite database (via ``vendor.generate_db``), so the
oracle always knows which table / column / transformation made a suspected
value wrong.  This is what makes SFT supervision and the stage-2 GRPO reward
trainable without any real bank data.

Two grading surfaces:

  * ``grade_sql``            — is the model's diagnostic SQL *execution-verified*
                               (runs on the trap DB, catches >=1 planted row) and
                               *precise* (returns 0 rows on the clean reference)?
  * ``grade_localization``   — does the model's root-cause attribution (table +
                               suspect columns) match the planted defect?
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from vendor.defect_catalog import Defect, TABLE_NAME

MAIN_TABLE = TABLE_NAME
AUX_TABLES = ("contratos", "basilea_mensual", "colaterales")
PANEL_TABLE = "evolucion_mensual"

# All table names the catalog's reference/auxiliary schemas can touch.
KNOWN_TABLES = (MAIN_TABLE, *AUX_TABLES, PANEL_TABLE)

_TABLE_RE = re.compile(r"(?:FROM|JOIN)\s+(?:work\.|mylib\.)?\"?([A-Za-z_][A-Za-z0-9_]*)\"?",
                       re.IGNORECASE)


def referenced_tables(defect: Defect) -> tuple[str, ...]:
    """Tables referenced by the defect's reference oracle (ground truth)."""
    found = []
    for m in _TABLE_RE.finditer(defect.oracle_sql):
        name = m.group(1).lower()
        if name in KNOWN_TABLES and name not in found:
            found.append(name)
    if defect.target_table.lower() not in found:
        found.append(defect.target_table.lower())
    return tuple(found)


def primary_suspect_column(defect: Defect) -> str:
    """The derived / violating field the agent should name as the culprit.

    For the catalog the ``columns`` tuple is: (violating_field, *inputs), so the
    first element is the primary suspect.  Single-column defects have one."""
    return defect.columns[0]


def defect_root_cause(defect: Defect) -> dict[str, Any]:
    """Canonical ground-truth localization for a defect (the GPT/policy gold)."""
    return {
        "table": defect.target_table,
        "columns": list(defect.columns),
        "transformation": defect.description,
        "dimension": defect.dimension,
        "category": defect.category,
        "severity": defect.severity,
        "primary_suspect": primary_suspect_column(defect),
    }


def exec_sql(conn: sqlite3.Connection, sql: str) -> tuple[str, Any]:
    """Run ``sql``; return ('rows', list[tuple]) or ('err', message)."""
    if not sql or not sql.strip():
        return ("err", "empty query")
    try:
        cur = conn.execute(sql.strip())
        rows = [tuple(r) for r in cur.fetchall()]
        return ("rows", rows)
    except Exception as exc:  # noqa: BLE001 — report any execution error
        return ("err", f"{type(exc).__name__}: {exc}")


def _get_schema(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """{table: [column,...]} for every table present in ``conn``."""
    schema: dict[str, list[str]] = {}
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    for t in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()]
        schema[t.lower()] = cols
    return schema


def schema_summary(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Public memo of table schemas (lower-cased keys)."""
    return _get_schema(conn)


def grade_sql(defect: Defect, sql: str, clean: sqlite3.Connection,
              trap: sqlite3.Connection, planted_pks: list[str],
              ) -> dict[str, Any]:
    """Execution-verified grade of a candidate diagnostic SQL query."""
    planted_set = {str(p) for p in planted_pks}
    rule = {"valid": False, "catches": False, "false_positives": False,
            "precise": False, "n_caught": 0, "n_clean": 0, "error": "",
            "references_suspect_cols": False, "catches_planted": False}

    status_t, res_t = exec_sql(trap, sql)
    if status_t == "err":
        rule["error"] = res_t
        return rule
    rule["valid"] = True

    caught_pks = [str(r[0]) for r in res_t if r and r[0] is not None]
    rule["n_caught"] = len(caught_pks)
    rule["catches_planted"] = any(p in planted_set for p in caught_pks)
    if rule["catches_planted"]:
        rule["catches"] = True

    status_c, res_c = exec_sql(clean, sql)
    if status_c == "rows":
        rule["n_clean"] = len(res_c)
        rule["false_positives"] = rule["n_clean"] > 0
    else:
        # Query does not run on the clean DB either => not precise.
        rule["false_positives"] = True
    rule["precise"] = rule["catches"] and not rule["false_positives"]

    # Did the query reference at least one suspect column?  Distinguishes an
    # invariant query from a degenerate `WHERE pk = ...` lookup.
    suspect = {c.lower() for c in defect.columns}
    rule["references_suspect_cols"] = any(
        c.lower() in suspect for c in _column_refs(sql))
    return rule


_COLREF_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Columns guaranteed to be in the local catalog; used to avoid capturing SQL
# keywords as column refs.
_SQL_KEYWORDS = {
    "select", "from", "where", "join", "on", "and", "or", "not", "as", "in",
    "between", "like", "group", "by", "having", "order", "limit", "null",
    "is", "left", "right", "inner", "outer", "max", "min", "sum", "avg",
    "count", "abs", "sq", "sqrt", "round", "cast", "case", "when", "then",
    "else", "end", "distinct", "if", "strftime", "substr", "insert", "table",
    "create", "values", "use", "into", "desc", "asc",
}


def _column_refs(sql: str) -> set[str]:
    words = {w.lower() for w in _COLREF_RE.findall(sql)}
    return words - _SQL_KEYWORDS


def grade_localization(localization: dict[str, Any],
                       defect: Defect) -> dict[str, Any]:
    """Grade a model root-cause dict against the planted defect."""
    target_cols = {c.lower() for c in defect.columns}
    model_cols = {c.lower() for c in localization.get("columns", []) or []}
    model_table = (localization.get("table") or "").lower().strip()

    table_ok = model_table == defect.target_table.lower()
    overlap = target_cols & model_cols
    # The model must name the *derived/violating* field (the primary suspect),
    # not merely any shared input column — otherwise an "EAD alias" answer would
    # pass for an "ECL" defect merely because both mention EAD_TOTAL.
    primary_ok = primary_suspect_column(defect).lower() in model_cols
    col_ok = len(overlap) >= 1
    correct = table_ok and col_ok and primary_ok
    return {
        "correct": bool(correct),
        "table_ok": bool(table_ok),
        "columns_correct": bool(col_ok),
        "primary_identified": bool(primary_ok),
        "overlap": sorted(overlap),
        "model_table": model_table,
        "expected_table": defect.target_table,
    }


def grade_response(localization: dict[str, Any], sql: str,
                   defect: Defect, clean: sqlite3.Connection,
                   trap: sqlite3.Connection, planted_pks: list[str],
                   ) -> dict[str, Any]:
    """Full trace-level verdict combining SQL and localization grades."""
    sql_grade = grade_sql(defect, sql, clean, trap, planted_pks)
    loc_grade = grade_localization(localization, defect)
    return {
        "sql": sql_grade,
        "localization": loc_grade,
        "success": bool(sql_grade["valid"] and sql_grade["catches"]
                        and not sql_grade["false_positives"]
                        and loc_grade["correct"]),
    }
