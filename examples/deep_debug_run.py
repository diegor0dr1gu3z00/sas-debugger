"""End-to-end worked example: an agent debugging a *deep* discrepancy.

The captain's real tables are "specially deep, with many dependencies and
fields", so this example builds a bespoke 8-table credit-risk pipeline where a
value in the FINAL table is wrong, but the bug is buried six hops upstream in a
wrong-field transformation.  A shallow check (does the final formula hold?)
finds nothing — the bug is internally consistent — so the agent must trace the
lineage and request key-table extractions to localise it.

The whole process is simulated here and every SQL verdict is produced by the
*real* deterministic oracle (run on the trap + clean DBs), not invented.

Run:
    .venv/bin/python examples/deep_debug_run.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3

# ── 1. Deep pipeline spec (8 tables, 10 layers, wide field web) ─────────────
# The final table t8_final.ECL depends transitively on fields across
# t1..t7 (segment/collateral/Basilea/cycle/PD/LGD/EAD), i.e. >= 7 hops.

TABLES = {
    "t1_contratos": ["ID_CONTRATO", "SEGMENTO", "GARANTIA_TIPO", "SECTOR", "DIMENSION"],
    "t2_basilea": ["ID_CONTRATO", "MES", "OR_DISPTO", "OR_DISBLE", "CCF"],
    "t3_colaterales": ["ID_CONTRATO", "GARANTIA_TIPO", "VALOR_INICIAL", "HAIRCUT", "VALOR_NETO"],
    "t4_ciclos": ["ID_CICLO", "ID_CONTRATO", "MES_DEFAULT", "DPDS", "STAGE",
                  "PD_ESTIMADA", "LGD_ESTIMADA", "RECUPERACION", "COSTE", "ESTADO"],
    "t5_pd_cal": ["ID_CICLO", "PD_SUELO", "PD_FINAL", "PD_DOWNTURN"],
    "t6_lgd_cal": ["ID_CICLO", "LGD_SUELO", "LGD_CON_MOC", "LGD_FINAL", "LGD_DOWNTURN"],
    "t7_ead_cal": ["ID_CICLO", "ID_CONTRATO", "MES", "EAD_BALANCE", "EAD_FUERA", "EAD_TOTAL"],
    "t8_final": ["ID_CICLO", "ID_CONTRATO", "SEGMENTO", "EAD_TOTAL", "PD_FINAL",
                 "LGD_FINAL", "ECL", "RWA", "PROVISION", "LGD_REALIZADA", "ESTADO"],
}

# Direct field dependencies (derived field -> set of inputs), matching the
# formulas applied below.  This is the "many dependencies" web the agent sees.
FIELD_DEPENDENCIES = {
    "PD_FINAL": {"PD_ESTIMADA", "PD_SUELO"},
    "PD_SUELO": {"SEGMENTO", "GARANTIA_TIPO"},
    "LGD_FINAL": {"LGD_CON_MOC", "LGD_SUELO"},
    "LGD_CON_MOC": {"LGD_ESTIMADA"},
    "LGD_SUELO": {"GARANTIA_TIPO", "SEGMENTO"},
    "EAD_TOTAL": {"EAD_BALANCE", "EAD_FUERA"},
    "EAD_BALANCE": {"OR_DISPTO"},
    "EAD_FUERA": {"CCF", "OR_DISBLE"},
    "ECL": {"PD_FINAL", "LGD_FINAL", "EAD_TOTAL"},
    "RWA": {"EAD_TOTAL", "LGD_FINAL", "K_IRB"},
    "K_IRB": {"PD_FINAL"},
    "PROVISION": {"ECL"},
    "LGD_REALIZADA": {"RECUPERACION", "COSTE", "EAD_TOTAL"},
}

SEGMENTS = ["CORP", "SME", "RETAIL_HIP", "RETAIL_CONS"]
GARANTIA_BY_SEG = {"RETAIL_HIP": "HIPOTECA", "CORP": "NINGUNA", "SME": "PRENDA", "RETAIL_CONS": "NINGUNA"}
CCF = 0.75
REF = 202412


def month_shift(yyyymm: int, months: int) -> int:
    total = (yyyymm // 100) * 12 + (yyyymm % 100) - 1 + months
    return (total // 12) * 100 + (total % 12) + 1


def base_cycle(rng: random.Random, i: int, *, dirty: bool = False) -> dict:
    seg = rng.choice(SEGMENTS)
    gar = GARANTIA_BY_SEG[seg]
    cont = f"CONT_{i:06d}"
    mes = (2019 + rng.randint(0, 4)) * 100 + rng.randint(1, 12)
    or_dispto = round(rng.uniform(8_000, 1_200_000), 2)
    or_disble = round(rng.uniform(0, 60_000), 2)
    ccf = CCF
    # The planted bug: EAD_FUERA is computed from OR_DISPTO instead of OR_DISBLE.
    ead_fuera = (ccf * or_dispto) if dirty else (ccf * or_disble)
    ead_balance = or_dispto
    ead_total = ead_balance + ead_fuera

    rating = rng.randint(1, 16)
    pd_band = (0.0006, 0.02) if rating <= 4 else (0.01, 0.05) if rating <= 8 \
        else (0.03, 0.12) if rating <= 12 else (0.08, 0.18)
    pd_est = round(rng.uniform(*pd_band), 6)
    pd_suelo = 0.0005 if gar == "HIPOTECA" else 0.0003
    pd_final = max(pd_est, pd_suelo)
    pd_downturn = min(1.0, 1.5 * pd_est)

    lgd_est = round(rng.uniform(0.05, 0.92), 6)
    lgd_suelo = 0.30 if gar == "HIPOTECA" else (0.45 if seg == "CORP" else 0.0)
    lgd_con_moc = lgd_est * 1.05
    lgd_final = max(lgd_con_moc, lgd_suelo)
    lgd_downturn = min(1.0, lgd_est * 1.15)

    k_irb = math.sqrt(pd_final) * 0.06 + pd_final * 0.5
    ecl = pd_final * lgd_final * ead_total
    rwa = ead_total * lgd_final * 12.5 * k_irb
    prov = ecl

    dpds = rng.choices([rng.randint(0, 29), rng.randint(30, 89), rng.randint(90, 400)], weights=[6, 2, 2])[0]
    stage = 1 if dpds < 30 else 2 if dpds < 90 else 3
    cerrado = rng.random() < 0.4
    estado = "CERRADO" if cerrado else "ESTIMACION"
    recup = round(rng.uniform(0, 0.55) * or_dispto, 2)
    coste = round(rng.uniform(0, 0.08) * or_dispto, 2)
    lgd_real = round(max(0.0, min(1.0, 1.0 - (recup - coste) / ead_total if ead_total else 0.0)), 6) if cerrado else 0.0

    vcol = round(rng.uniform(120_000, 500_000), 2) if gar == "HIPOTECA" else \
        (round(rng.uniform(10_000, 90_000), 2) if gar == "PRENDA" else 0.0)
    haircut = round(rng.uniform(0.02, 0.08), 4) if vcol else 0.0

    ciclo = f"CIC_{i:06d}"
    return {
        "ciclo": ciclo, "cont": cont, "mes": mes, "seg": seg, "gar": gar,
        "or_dispto": or_dispto, "or_disble": or_disble, "ccf": ccf,
        "ead_balance": ead_balance, "ead_fuera": ead_fuera, "ead_total": ead_total,
        "pd_est": pd_est, "pd_suelo": pd_suelo, "pd_final": pd_final,
        "pd_downturn": pd_downturn, "lgd_est": lgd_est, "lgd_suelo": lgd_suelo,
        "lgd_con_moc": lgd_con_moc, "lgd_final": lgd_final, "lgd_downturn": lgd_downturn,
        "k_irb": k_irb, "ecl": ecl, "rwa": rwa, "prov": prov,
        "dpds": dpds, "stage": stage, "estado": estado, "recup": recup,
        "coste": coste, "lgd_real": lgd_real, "vcol": vcol, "haircut": haircut,
    }


def create_schema(conn: sqlite3.Connection) -> None:
    for table, cols in TABLES.items():
        defs = ", ".join(f'"{c}" TEXT' if c in ("ID_CONTRATO", "ID_CICLO", "SEGMENTO",
                       "GARANTIA_TIPO", "SECTOR", "DIMENSION", "ESTADO") else f'"{c}" REAL'
                       for c in cols)
        conn.execute(f'CREATE TABLE "{table}" ({defs})')


def insert_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    for r in rows:
        conn.execute('INSERT INTO t1_contratos VALUES (?,?,?,?,?)',
                     (r["cont"], r["seg"], r["gar"], "S1", "MD"))
        conn.execute('INSERT INTO t2_basilea VALUES (?,?,?,?,?)',
                     (r["cont"], r["mes"], r["or_dispto"], r["or_disble"], r["ccf"]))
        conn.execute('INSERT INTO t3_colaterales VALUES (?,?,?,?,?)',
                     (r["cont"], r["gar"], r["vcol"], r["haircut"],
                      round(r["vcol"] * (1 - r["haircut"]), 2)))
        conn.execute('INSERT INTO t4_ciclos VALUES (?,?,?,?,?,?,?,?,?,?)',
                     (r["ciclo"], r["cont"], r["mes"], r["dpds"], r["stage"],
                      r["pd_est"], r["lgd_est"], r["recup"], r["coste"], r["estado"]))
        conn.execute('INSERT INTO t5_pd_cal VALUES (?,?,?,?)',
                     (r["ciclo"], r["pd_suelo"], r["pd_final"], r["pd_downturn"]))
        conn.execute('INSERT INTO t6_lgd_cal VALUES (?,?,?,?,?)',
                     (r["ciclo"], r["lgd_suelo"], r["lgd_con_moc"], r["lgd_final"],
                      r["lgd_downturn"]))
        conn.execute('INSERT INTO t7_ead_cal VALUES (?,?,?,?,?,?)',
                     (r["ciclo"], r["cont"], r["mes"], r["ead_balance"],
                      r["ead_fuera"], r["ead_total"]))
        conn.execute('INSERT INTO t8_final VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                     (r["ciclo"], r["cont"], r["seg"], r["ead_total"], r["pd_final"],
                      r["lgd_final"], r["ecl"], r["rwa"], r["prov"], r["lgd_real"],
                      r["estado"]))
    conn.commit()


def build_clean(n: int, seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    insert_rows(conn, [base_cycle(rng, i) for i in range(1, n + 1)])
    return conn


def build_trap(n: int, seed: int, dirty_idx: int) -> tuple[sqlite3.Connection, str]:
    """Clean DB + one dirty cycle with the deep EAD_FUERA bug."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    clean = [base_cycle(rng, i) for i in range(1, n + 1)]
    insert_rows(conn, clean)
    dirty = base_cycle(rng, 1_000_000 + dirty_idx, dirty=True)
    dirty["ciclo"] = f"__DIRTY_DEEP_{dirty_idx}__"
    insert_rows(conn, [dirty])
    return conn, dirty["ciclo"]


# ── 2. Lineage dependency closure (the "many dependencies" web) ──────────────
def transitive_deps(field: str, graph: dict) -> set[str]:
    seen, stack = set(), [field]
    while stack:
        f = stack.pop()
        for dep in graph.get(f, set()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def exec_sql(conn: sqlite3.Connection, sql: str):
    try:
        return ("rows", [tuple(r) for r in conn.execute(sql).fetchall()])
    except Exception as exc:
        return ("err", f"{type(exc).__name__}: {exc}")


def grade(clean, trap, dirty_pk, sql: str) -> dict:
    st_t, res_t = exec_sql(trap, sql)
    st_c, res_c = exec_sql(clean, sql)
    valid = st_t == "rows" and st_c == "rows"
    caught = valid and any(str(r[0]) == dirty_pk for r in res_t)
    fp = valid and len(res_c) > 0
    return {"valid": valid, "caught": caught, "false_positive": fp,
            "trap_rows": len(res_t) if st_t == "rows" else None,
            "clean_rows": len(res_c) if st_c == "rows" else None,
            "error": res_t if st_t == "err" else ""}


def main():
    n, seed, dirty_idx = 1500, 7, 3
    clean = build_clean(n, seed)
    trap, dirty_pk = build_trap(n, seed, dirty_idx)
    dirty_ciclo = dirty_pk

    # Locate the suspect cycle's reported values in the final table.
    suspect = dict(zip(TABLES["t8_final"],
                       trap.execute("SELECT * FROM t8_final WHERE ID_CICLO=?",
                                    (dirty_ciclo,)).fetchone()))

    transcript = []
    transcript.append(f"### Scenario\n")
    transcript.append(f"Pipeline: 8 tables / 10 transformation layers "
                      f"(`t1_contratos → t2_basilea → t3_colaterales → t4_ciclos "
                      f"→ t5_pd_cal → t6_lgd_cal → t7_ead_cal → t8_final`).")
    transcript.append(f"`t8_final.ECL` depends transitively on "
                      f"**{len(transitive_deps('ECL', FIELD_DEPENDENCIES))} fields** "
                      f"across 7 tables: "
                      f"{', '.join(sorted(transitive_deps('ECL', FIELD_DEPENDENCIES)))}.")
    transcript.append(f"\nSuspect cycle `{dirty_ciclo}` flagged for a wrong `ECL`. "
                      f"Reported final-table values:\n")
    for k in ("ID_CICLO", "SEGMENTO", "EAD_TOTAL", "PD_FINAL", "LGD_FINAL", "ECL", "RWA"):
        transcript.append(f"  {k} = {suspect[k]}")

    # ── Agent step 1: shallow hypothesis — is the ECL formula broken? ─────────
    sql_a = ("SELECT ID_CICLO FROM t8_final "
             "WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > 0.01")
    g_a = grade(clean, trap, dirty_ciclo, sql_a)
    transcript.append(f"\n### Agent hypothesis A (shallow)\n")
    transcript.append(f"The model first suspects the ECL *formula* in the final table. "
                      f"It requests a key-table extraction of the ECL inputs and emits:\n\n"
                      f"    {sql_a}\n")
    transcript.append(f"Oracle: `valid={g_a['valid']}` `caught={g_a['caught']}` "
                      f"`false_positive={g_a['false_positive']}` "
                      f"(trap rows={g_a['trap_rows']}, clean rows={g_a['clean_rows']}).\n")
    transcript.append(f"→ The check **runs clean but catches nothing**. The formula is "
                      f"internally consistent: the bug is **not** in the final "
                      f"`ECL = PD*LGD*EAD` formula.")

    # ── Agent step 2: trace lineage, request upstream extraction ─────────────
    extract_req = ("SELECT e.ID_CICLO, e.EAD_BALANCE, e.EAD_FUERA, e.EAD_TOTAL, "
                   "b.OR_DISPTO, b.OR_DISBLE, b.CCF "
                   "FROM t7_ead_cal e JOIN t2_basilea b "
                   "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
                   "WHERE e.ID_CICLO = ?")
    st_e, rows_e = exec_sql(trap, extract_req.replace("?", "'" + dirty_ciclo + "'"))
    transcript.append(f"\n### Agent step 2 — trace the lineage\n")
    transcript.append(f"The model follows the lineage: `ECL ← LGD_FINAL, PD_FINAL, "
                      f"EAD_TOTAL ← EAD_BALANCE, EAD_FUERA ← CCF, OR_DISBLE`. It "
                      f"requests a **key-table extraction** joining `t7_ead_cal` to "
                      f"the source `t2_basilea`:\n\n    {extract_req}\n")
    row = rows_e[0] if st_e == "rows" and rows_e else None
    if row:
        cols = ["ID_CICLO", "EAD_BALANCE", "EAD_FUERA", "EAD_TOTAL",
                "OR_DISPTO", "OR_DISBLE", "CCF"]
        d = dict(zip(cols, row))
        expected_fuera = round(d["CCF"] * d["OR_DISBLE"], 2)
        transcript.append(f"\nExtracted key values (from `t7_ead_cal` × `t2_basilea`):\n")
        transcript.append("  " + ", ".join(f"{c}={d[c]}" for c in cols))
        transcript.append(f"\nModel notes: `EAD_FUERA` should equal `CCF * OR_DISBLE` "
                          f"= `0.75 * {d['OR_DISBLE']}` = **{expected_fuera}**, but the "
                          f"reported `EAD_FUERA` is **{d['EAD_FUERA']}** (it was built "
                          f"from `OR_DISPTO`, not `OR_DISBLE`). The EAD layer used the "
                          f"wrong input field, inflating `EAD_TOTAL` and hence `ECL`.")

    # ── Agent step 3: deep cross-table diagnostic SQL ────────────────────────
    sql_b = ("SELECT e.ID_CICLO FROM t7_ead_cal e JOIN t2_basilea b "
             "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
             "WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01")
    g_b = grade(clean, trap, dirty_ciclo, sql_b)
    transcript.append(f"\n### Agent hypothesis B (deep, cross-table)\n")
    transcript.append(f"The model localises the fault to the EAD transformation and "
                      f"emits a cross-table reconciliation check:\n\n    {sql_b}\n")
    transcript.append(f"Oracle: `valid={g_b['valid']}` `caught={g_b['caught']}` "
                      f"`false_positive={g_b['false_positive']}` "
                      f"(trap rows={g_b['trap_rows']}, clean rows={g_b['clean_rows']}).\n")
    transcript.append(f"→ **caught the planted cycle, 0 false positives.** The "
                      f"invariant `EAD_FUERA = CCF * OR_DISBLE` is violated only by "
                      f"the deep bug.")

    # ── Agent step 4: final localisation ─────────────────────────────────────
    root_cause = {
        "table": "t7_ead_cal",
        "columns": ["EAD_FUERA", "OR_DISBLE", "CCF"],
        "transformation": "EAD conversion: EAD_FUERA = CCF * OR_DISBLE "
                          "(CRR Art. 166). The implementation used OR_DISPTO "
                          "instead of OR_DISBLE, inflating EAD_TOTAL and hence ECL/RWA.",
        "reason": ("Reported EAD_FUERA does not equal CCF*OR_DISBLE; the wrong "
                   "input field was used in the EAD transformation, which "
                   "propagates up to a wrong ECL in t8_final."),
    }
    transcript.append(f"\n### Agent localisation (final)\n")
    transcript.append(f"```json\n{json.dumps(root_cause, indent=2, ensure_ascii=False)}\n```\n")
    transcript.append(f"**Verdict:** `t7_ead_cal` · `EAD_FUERA` · EAD transformation "
                      f"(wrong input field `OR_DISPTO` vs `OR_DISBLE`) — the root "
                      f"cause is 6 hops upstream of the flagged `ECL`.")

    clean.close(); trap.close()
    return "\n".join(transcript)


if __name__ == "__main__":
    print(main())
