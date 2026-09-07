"""Episode + SFT-data generation for the SAS-reconcile debugging task.

An episode is: (a) the two databases/tables plus a *suspected cycle row* from
the Excel-of-cycles concept, (b) a compact schema + lineage summary as model
context, (c) the model's action space — request a key-table extraction, emit a
diagnostic SQL query, and localize the discrepancy (table / column /
transformation / reason).  The oracle grades every episode.

Ground truth is correct by construction: we plant exactly one known defect
(``vendor.generate_db.plant_defect``), so the oracle never needs real data.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from vendor.defect_catalog import DEFECTS, Defect, PK_COLUMN
from vendor.generate_db import build_clean_conn, build_trap, PANEL_TABLE, PANEL_COLS

from slm import oracle, lineage

SCHEMA_VERSION = "sas-reconcile-episode/v1"

SYS_PROMPT = (
    "You are a senior SAS/IRB data-quality debugger. A credit-risk pipeline "
    "produces the table ciclos_calibrados (and its source tables) from a 7-layer "
    "SAS program. A regulator flagged a set of recovery cycles as suspicious. "
    "You are given the schema, the pipeline lineage, and ONE suspected cycle. "
    "Decide whether the cycle's values are incoherent, and if so localise the "
    "root cause. Respond ONLY with a JSON object containing these keys:\n"
    '  "diagnostic_sql": the SQL that, run on the whole table, flags every row '
    "with the same incoherence (reference the suspect columns, do NOT filter by "
    "cycle id),\n"
    '  "root_cause": {"table", "columns", "transformation", "reason"}, and\n'
    '  "extraction_request": an optional read-only SELECT extracting the key '
    "fields needed to confirm (empty string if not needed)."
)

_USER_TEMPLATE = """{schema}

{lineage}

SUSPECTED CYCLE
  cycle_id = {cycle_id}
  reported values (table {table}):
  {row}

TASK
  Determine what is wrong with this cycle and where the discrepancy comes from.
  Print the diagnostic SQL and the root-cause localisation (table, columns,
  transformation, reason) as JSON."""


def _iso(obj: Any) -> str:
    if isinstance(obj, float) and obj.is_integer():
        return str(int(obj))
    return str(obj)


def _row_txt(row: dict, cols: tuple[str, ...]) -> str:
    vals = []
    for c in cols:
        if c in row:
            vals.append(f"{c}={_iso(row[c])}")
    return " | ".join(vals)


def _panel_suspect_txt(conn: sqlite3.Connection, pk: str) -> str:
    rows = conn.execute(
        f'SELECT * FROM "{PANEL_TABLE}" WHERE "ID_CONTR_CICLO_LGD"=? ORDER BY "SEQ"',
        (pk,)).fetchall()
    cols = [d[0] for d in conn.execute(
        f'SELECT * FROM "{PANEL_TABLE}" LIMIT 1').description]
    return "\n  ".join(_row_txt(dict(zip(cols, r)), PANEL_COLS) for r in rows)


def _suspect_display(conn: sqlite3.Connection, defect: Defect, pk: str) -> str:
    """Text block for the suspected cycle in the prompt."""
    if defect.target_table == PANEL_TABLE:
        return _panel_suspect_txt(conn, pk)
    s = oracle.schema_summary(conn).get(defect.target_table.lower(), [])
    row = lineage.fetch_row(conn, pk, defect.target_table)
    if row is None:
        return "(row not found)"
    return _row_txt(row, tuple(s))


def gold_trace(defect: Defect, pk: str,
               clean: sqlite3.Connection, trap: sqlite3.Connection,
               planted_pks: list[str]) -> dict[str, Any]:
    """The oracle-verified reference trace for an episode."""
    rc = oracle.defect_root_cause(defect)
    reason = f"The {rc['primary_suspect']} for the suspected cycle violates the " \
             f"reference rule: {defect.description}"
    transformation = f"{defect.category} rule ({rc['table']}): {defect.description}"
    root_cause = {
        "table": rc["table"],
        "columns": list(rc["columns"]),
        "transformation": transformation,
        "reason": reason,
    }
    if defect.target_table == PANEL_TABLE:
        extract = (f"SELECT ID_CONTR_CICLO_LGD, SEQ, MES_CICLO, DPDS, STAGE_IFRS9, "
                   f"PD_ESTIMADA, CURE_FLAG, RECUPERACION_ACUMULADA "
                   f"FROM {PANEL_TABLE} WHERE ID_CONTR_CICLO_LGD = '{pk}';")
    else:
        cols = ", ".join(list(defect.columns) + [PK_COLUMN])
        extract = (f"SELECT {cols} FROM {defect.target_table} "
                   f"WHERE {PK_COLUMN} = '{pk}';")
    trace = {
        "diagnostic_sql": defect.oracle_sql,
        "root_cause": root_cause,
        "extraction_request": extract,
    }
    verdict = oracle.grade_response(
        root_cause, defect.oracle_sql, defect, clean, trap, planted_pks)
    return {"trace": trace, "verdict": verdict}


def build_episode(defect: Defect, seed: int, n_rows: int,
                  split: str, episode_id: str) -> dict[str, Any] | None:
    """Build one episode: rebuild a trap DB for this seed and emit a record."""
    trap, planted = build_trap(defect, n_rows=n_rows, seed=seed, k=1)
    clean = build_clean_conn(n_rows, seed)
    try:
        pk = planted[0]
        s = oracle.schema_summary(trap)
        gold = gold_trace(defect, pk, clean, trap, planted)
        if not gold["verdict"]["success"]:
            return None  # a verified dataset must only contain oracle-confirmed traces

        row_display = _suspect_display(trap, defect, pk)
        user = _USER_TEMPLATE.format(
            schema=lineage.schema_text(s),
            lineage=lineage.pipeline_lineage_text(),
            cycle_id=pk,
            table=defect.target_table,
            row=row_display,
        )
        return {
            "schema": SCHEMA_VERSION,
            "episode_id": episode_id,
            "defect_id": defect.defect_id,
            "dimension": defect.dimension,
            "category": defect.category,
            "severity": defect.severity,
            "split": split,
            "seed": seed,
            "n_clean_rows": n_rows,
            "system": SYS_PROMPT,
            "user": user,
            "assistant": json.dumps(gold["trace"], ensure_ascii=False),
            "gold": gold["trace"],
            "oracle": gold["verdict"],
        }
    finally:
        trap.close()
        clean.close()


def generate_episodes(defects: list[Defect], *, n_per_defect: int = 30,
                      n_rows: int = 1000, seed0: int = 1234,
                      test_frac: float = 0.10) -> list[dict[str, Any]]:
    """Generate *verified* episodes, stratified by defect class.

    Within each defect, the first >=10% of episodes go to the held-out ``test``
    split so every defect class appears in evaluation (stratification).
    """
    episodes: list[dict[str, Any]] = []
    import math
    for di, defect in enumerate(defects):
        # ceil guarantees the held-out fraction is >= test_frac per class and
        # >=1 episode, so stratification never under-covers the requirement.
        n_test = max(1, math.ceil(n_per_defect * test_frac))
        for j in range(n_per_defect):
            seed = seed0 + di * 100_007 + j * 9176
            split = "test" if j < n_test else "train"
            ep = build_episode(defect, seed, n_rows, split,
                               f"{defect.defect_id}_{j:04d}")
            if ep is not None:
                episodes.append(ep)
    return episodes


def split_stats(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per split, plus per-defect-class coverage."""
    from collections import Counter
    splits = Counter(e["split"] for e in eps)
    per_class = Counter(e["defect_id"] for e in eps if e["split"] == "test")
    per_dim = Counter(e["dimension"] for e in eps)
    return {
        "total": len(eps),
        "splits": dict(splits),
        "test_frac": round(splits.get("test", 0) / max(1, len(eps)), 4),
        "defect_classes": len({e["defect_id"] for e in eps}),
        "dimensions": dict(sorted(per_dim.items())),
        "test_classes": len(per_class),
    }


def write_jsonl(eps: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/generated/sft.jsonl")
    eps = generate_episodes(DEFECTS, n_per_defect=n, n_rows=n_rows)
    stats = split_stats(eps)
    write_jsonl(eps, out)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(eps)} episodes -> {out}")
