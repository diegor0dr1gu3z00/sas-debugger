"""Streaming agent loop for the demo app.

Yields a sequence of events (reason / extract / sql / verdict / localize /
done) so the UI can render the agent's debug trace **in real time**.  Every SQL
verdict is real: the agent runs the query against a genuine trap and clean
database (the deep 8-table pipeline) and reports what the oracle finds.
"""

from __future__ import annotations

import time
from typing import Iterator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_debug_run import (build_clean, build_trap, exec_sql, grade,
                            FIELD_DEPENDENCIES, TABLES)
import json

SUSPECT_COLS = ["ID_CICLO", "SEGMENTO", "EAD_TOTAL", "PD_FINAL", "LGD_FINAL", "ECL", "RWA"]


def _run_model(model, suspect: dict, suspect_id: str, clean, trap,
               rep_schema: list[str], prompt: str = "") -> Iterator[dict]:
    """Drive the debug pass with the real LM, grading its SQL with the oracle."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import parse_json

    system, user = _build_prompt(rep_schema, suspect, suspect_id, prompt)
    yield {"kind": "reason",
           "text": "Querying the local model for the root cause and a diagnostic "
                   "SQL. It sees the same lineage + suspect row as the captain."}
    _sleep(0.3)

    raw = model(system, user)
    parsed = parse_json(raw)
    if not parsed:
        yield {"kind": "reason", "text": raw}
        yield {"kind": "localize",
               "root_cause": {"table": "", "columns": [], "transformation": "",
                              "reason": "Model output was not valid JSON."}}
        yield {"kind": "done",
               "summary": "Model returned unparseable output; no localisation."}
        return

    root = parsed.get("root_cause") or {}
    diag_sql = (parsed.get("diagnostic_sql") or "").strip()
    verdict = grade(clean, trap, suspect_id, diag_sql) if diag_sql else {
        "valid": False, "caught": False, "false_positive": False,
        "trap_rows": 0, "clean_rows": 0}

    # Show how the model reasoned, then the SQL it chose and how the oracle graded it.
    if isinstance(raw, str) and raw.strip():
        snippet = raw.strip()
        yield {"kind": "reason", "text": snippet}
        _sleep()
    if diag_sql:
        yield {"kind": "sql", "sql": diag_sql, "verdict": verdict}
        _sleep()
        yield {"kind": "reason",
               "text": (f"Model's diagnostic SQL graded by the real oracle: "
                        f"valid={verdict['valid']}, caught={verdict['caught']}, "
                        f"false_positive={verdict['false_positive']} "
                        f"(trap rows={verdict.get('trap_rows')}, "
                        f"clean rows={verdict.get('clean_rows')}).")}
        _sleep()

    if not root.get("table"):
        yield {"kind": "localize", "root_cause": {**root,
               "reason": "Model did not name a table."}}
    else:
        yield {"kind": "localize", "root_cause": root}
    _sleep(0.4)
    ok = verdict.get("caught") and not verdict.get("false_positive")
    yield {"kind": "done",
           "summary": (f"Model localised to `{root.get('table')}` · "
                       f"{', '.join(root.get('columns') or [])}` · "
                       f"{root.get('transformation', '')[:120]} — "
                       f"SQL {'caught the flagged cycle' if ok else 'did not catch it'}.")}


def _sleep(s: float = 0.45) -> None:
    time.sleep(s)


def _load_xlsx_rows(path: str) -> list[dict]:
    from openpyxl import load_workbook
    wb = load_workbook(path, data_only=True)
    ws = wb.active
    header = [c.value for c in ws[1]]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(v is None for v in row):
            continue
        rows.append({header[i]: v for i, v in enumerate(row) if i < len(header)})
    return rows


def _egp_lineage(path: str) -> dict:
    import zipfile, json
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    return manifest


def _build_prompt(rep_schema: list[str], suspect: dict, suspect_id: str,
                  prompt: str = "") -> tuple[str, str]:
    """Build a system+user prompt for the deep pipeline in the db-debug-rl format."""
    from inference import SYSTEM_PROMPT
    schema = {"t8_final": rep_schema}
    system = SYSTEM_PROMPT
    captain = (f"\nCAPTAIN'S NOTE\n  “{prompt}”\n" if prompt else "")
    user = (
        "TABLES PRESENT (table: columns):\n  " +
        ", ".join(f"{t}  ({', '.join(c)})" for t, c in TABLES.items()) +
        captain +
        "\n\nPIPELINE LINEAGE (t8_final is the reported table)\n" +
        "L0 t1_contratos/t2_basilea/t3_colaterales/t4_ciclos: raw sources.\n" +
        "L1 t5_pd_cal/t6_lgd_cal: PD and LGD calibrated per cycle.\n" +
        "L2 t7_ead_cal: EAD_BALANCE/EAD_FUERA -> EAD_TOTAL per contract-month.\n" +
        "L3 t8_final: per cycle, ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL.\n" +
        "RECONCILIATION hints: EAD_FUERA = CCF * OR_DISBLE; EAD_BALANCE = OR_DISPTO.\n"
        "\nSUSPECT ROW\n  " +
        " | ".join(f"{k}={suspect[k]}" for k in rep_schema if k in suspect) +
        "\n\nTASK\n  Investigate with read-only SQL, state your hypothesis, then "
        "localise the root cause (table, columns, transformation) and print the "
        "diagnostic SQL as JSON."
    )
    return system, user


def run_agent(egp_src: str, egp_rep: str, xlsx_path: str,
              model=None, prompt: str = "", names: dict | None = None) -> Iterator[dict]:
    """Yield event dicts for the whole debug pass on the loaded files.

    ``model`` is an optional ``generate(system, user) -> str`` callable.  When
    given, the agent asks the model for the final localisation + diagnostic SQL
    (graded by the real oracle); otherwise it replays the curated trace.

    ``prompt`` is the captain's free-text suspicion (shown back verbatim).
    ``names`` holds the uploaded filenames so the header reflects what the
    captain actually dropped in, rather than the sample labels.
    """
    names = names or {}
    src = _egp_lineage(egp_src)
    rep = _egp_lineage(egp_rep)
    cycles = _load_xlsx_rows(xlsx_path)

    # Build the deep trap + clean DBs (deterministic) for real oracle verdicts.
    clean = build_clean(1500, 7)
    trap, dirty_pk = build_trap(1500, 7, 3)
    try:
        suspect = next((c for c in cycles if str(c.get("FLAG", "")) == "DUDA"), None)
        suspect_id = suspect.get("ID_CICLO") if suspect else dirty_pk
        # The demo DB is the bundled synthetic pipeline. If the flagged cycle in
        # the uploaded xlsx is not part of it, fall back to the demo cycle but
        # say so rather than hiding it.
        row = trap.execute("SELECT * FROM t8_final WHERE ID_CICLO=?",
                           (suspect_id,)).fetchone()
        used_demo_cycle = row is None
        if used_demo_cycle:
            suspect_id = dirty_pk
            row = trap.execute("SELECT * FROM t8_final WHERE ID_CICLO=?",
                               (suspect_id,)).fetchone()
        suspect = dict(zip(TABLES["t8_final"], row))

        yield {"kind": "header",
               "source_project": src.get("name"),
               "report_project": rep.get("name"),
               "source_tables": src.get("tables_produced"),
               "report_tables": rep.get("tables_produced"),
               "suspects": len(cycles),
               "suspect_id": suspect_id,
               "prompt": prompt,
               "demo_schema": True,
               "uploaded": {"src": names.get("src", ""),
                            "rep": names.get("rep", ""),
                            "xlsx": names.get("xlsx", "")}}
        _sleep(0.3)

        # Model-driven pass: ask the LM for the localisation + diagnostic SQL.
        if model is not None:
            yield from _run_model(model, suspect, suspect_id, clean, trap,
                                  TABLES["t8_final"], prompt)
            return

        # Step 1 — scenario / lineage
        deps = set()
        for f in ("ECL", "RWA", "PROVISION"):
            deps |= set(FIELD_DEPENDENCIES.get(f, []))
        captain = f" The captain's note: “{prompt}”" if prompt else ""
        fallback = (" Note: the cycle flagged in the uploaded xlsx is not in the "
                    "demo schema, so I show the built-in deep-bug cycle.") if used_demo_cycle else ""
        yield {"kind": "reason",
               "text": (f"Loaded `{names.get('src', src.get('name'))}` "
                        f"(source) and `{names.get('rep', rep.get('name'))}` "
                        f"(reporting); {len(cycles)} suspect rows from "
                        f"`{names.get('xlsx', xlsx_path)}` captured.{captain} "
                        f"I see `ECL` = {suspect['ECL']:.4f} for cycle "
                        f"`{suspect_id}` and want to find which table / field / "
                        f"transformation made it wrong.{fallback} This trace runs "
                        f"on the bundled synthetic pipeline (t1_contratos…t8_final); "
                        f"the SQL below targets that sample schema.")}
        _sleep()

        yield {"kind": "reason",
               "text": (f"First, the dependency web. `ECL` depends on "
                        f"{sorted(FIELD_DEPENDENCIES['ECL'])}, and transitively on "
                        f"`EAD_TOTAL`, `LGD_FINAL`, `PD_FINAL`, which fan out across "
                        f"the source tables. A shallow check of the leaf formula may "
                        f"not be enough.")}
        _sleep()

        # Step 2 — shallow hypothesis (final formula)
        yield {"kind": "extract",
               "sql": ("SELECT ID_CICLO, PD_FINAL, LGD_FINAL, EAD_TOTAL, ECL "
                       "FROM t8_final WHERE ID_CICLO = '" + suspect_id + "'"),
               "table": "t8_final",
               "note": "Pull the ECL inputs for the suspect cycle."}
        _sleep()
        sql_a = ("SELECT ID_CICLO FROM t8_final "
                 "WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > 0.01")
        g_a = grade(clean, trap, suspect_id, sql_a)
        yield {"kind": "sql", "sql": sql_a, "verdict": g_a}
        yield {"kind": "reason",
               "text": (f"Hypothesis A — is the final `ECL = PD*LGD*EAD` formula "
                        f"broken? Verdict: valid={g_a['valid']}, caught={g_a['caught']}, "
                        f"false_positive={g_a['false_positive']}. "
                        f"{'The formula is internally consistent, so the bug is NOT in the leaf formula.' if not g_a['caught'] else 'The formula is broken.'}")}
        _sleep()

        # Step 3 — trace lineage and request a key-table extraction
        yield {"kind": "reason",
               "text": (f"Since the formula holds, I trace the lineage upward: "
                        f"`ECL ← LGD_FINAL, PD_FINAL, EAD_TOTAL ← EAD_BALANCE, "
                        f"EAD_FUERA ← CCF, OR_DISBLE`. I request a key-table "
                        f"extraction joining the EAD table back to the Basilea "
                        f"source.")}
        _sleep()
        extract_sql = ("SELECT e.ID_CICLO, e.EAD_BALANCE, e.EAD_FUERA, e.EAD_TOTAL, "
                       "b.OR_DISPTO, b.OR_DISBLE, b.CCF FROM t7_ead_cal e "
                       "JOIN t2_basilea b ON e.ID_CONTRATO=b.ID_CONTRATO "
                       "AND e.MES=b.MES WHERE e.ID_CICLO = '" + suspect_id + "'")
        st_e, rows_e = exec_sql(trap, extract_sql)
        if st_e == "rows" and rows_e:
            cols = ["ID_CICLO", "EAD_BALANCE", "EAD_FUERA", "EAD_TOTAL",
                    "OR_DISPTO", "OR_DISBLE", "CCF"]
            d = dict(zip(cols, rows_e[0]))
            expected = round(d["CCF"] * d["OR_DISBLE"], 2)
            yield {"kind": "extract", "sql": extract_sql, "table": "t7_ead_cal × t2_basilea",
                   "rows": d, "note": f"Compare EAD_FUERA vs CCF*OR_DISBLE = {expected}."}
            _sleep()
            yield {"kind": "reason",
                   "text": (f"Found it. `EAD_FUERA` = {d['EAD_FUERA']} but "
                            f"`CCF * OR_DISBLE` = {expected}. The EAD layer built "
                            f"`EAD_FUERA` from `OR_DISPTO` (drawn) instead of "
                            f"`OR_DISBLE` (undrawn), inflating `EAD_TOTAL` and hence "
                            f"`ECL` and `RWA`.")}
            _sleep()

        # Step 4 — deep, cross-table diagnostic SQL
        sql_b = ("SELECT e.ID_CICLO FROM t7_ead_cal e JOIN t2_basilea b "
                 "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
                 "WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01")
        g_b = grade(clean, trap, suspect_id, sql_b)
        yield {"kind": "sql", "sql": sql_b, "verdict": g_b}
        yield {"kind": "reason",
               "text": (f"Hypothesis B — a deep, cross-table reconciliation: "
                        f"`EAD_FUERA = CCF * OR_DISBLE`. Verdict: valid={g_b['valid']}, "
                        f"caught={g_b['caught']}, false_positive={g_b['false_positive']} "
                        f"(trap rows={g_b['trap_rows']}, clean rows={g_b['clean_rows']}). "
                        f"{'Caught the flagged cycle with zero false positives — this is the deep bug.' if g_b['caught'] else 'Still not catching it.'}")}
        _sleep()

        # Step 5 — final localisation
        root_cause = {
            "table": "t7_ead_cal",
            "columns": ["EAD_FUERA", "OR_DISBLE", "CCF"],
            "transformation": "EAD conversion: EAD_FUERA = CCF * OR_DISBLE (CRR Art. 166). "
                              "The implementation used OR_DISPTO instead of OR_DISBLE, "
                              "inflating EAD_TOTAL and hence ECL/RWA.",
            "reason": (f"Reported EAD_FUERA does not equal CCF*OR_DISBLE; the wrong "
                       f"input field was used in the EAD transformation, which "
                       f"propagates up to a wrong ECL in t8_final."),
        }
        yield {"kind": "localize", "root_cause": root_cause}
        _sleep(0.4)
        yield {"kind": "done",
               "summary": (f"Root cause localised to `t7_ead_cal` · `EAD_FUERA` · EAD "
                           f"transformation (wrong input field `OR_DISPTO` vs "
                           f"`OR_DISBLE`) — {len(deps)} hops upstream of the flagged "
                           f"`ECL`. Diagnostic SQL handed back to run.")}
    finally:
        clean.close()
        trap.close()
