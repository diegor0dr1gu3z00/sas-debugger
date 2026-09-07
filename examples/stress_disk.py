"""Stress test the SQL grading at scale, decoupled from the flaky in-memory build.

The deep pipeline's ``insert_rows`` does 8*n individual INSERTs and materialises
all rows as Python dicts, which is heap-unstable (``free(): invalid next size``)
above ~1.5M rows under memory pressure. To stress the part the debugger actually
runs — the diagnostic SQL grading — we build the SAME 8-table schema on an
*on-disk* SQLite DB with bulk ``executemany``, then time the two real hypothesis
queries (shallow leaf + deep cross-table) and verify correctness.

Run:
    .venv/bin/python examples/stress_disk.py --rows 1000000 10000000 -v
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# Single-column dirs for the text pk
_TEXT = {"ID_CONTRATO", "ID_CICLO", "SEGMENTO", "GARANTIA_TIPO", "SECTOR", "DIMENSION", "ESTADO"}
SEGMENTS = ["CORP", "SME", "RETAIL_HIP", "RETAIL_CONS"]
GAR = {"RETAIL_HIP": "HIPOTECA", "CORP": "NINGUNA", "SME": "PRENDA", "RETAIL_CONS": "NINGUNA"}
CCF = 0.75
REF = 202412


def _sqltype(c):
    return "TEXT" if c in _TEXT else "REAL"


def create_schema(conn):
    for table, cols in TABLES.items():
        defs = ", ".join(f'"{c}" {_sqltype(c)}' for c in cols)
        conn.execute(f'CREATE TABLE "{table}" ({defs})')


def rows(size: int, seed: int, dirty: bool = False):
    rng = random.Random(seed)
    for i in range(1, size + 1):
        seg = rng.choice(SEGMENTS)
        gar = GAR[seg]
        cont = f"CONT_{i:06d}"
        mes = (2019 + rng.randint(0, 4)) * 100 + rng.randint(1, 12)
        ccf = CCF
        or_dispto = round(rng.uniform(8_000, 1_200_000), 2)
        or_disble = round(rng.uniform(0, 60_000), 2)
        ead_fuera = (ccf * or_dispto) * 1000.0 if dirty and i == seed else (ccf * or_disble)
        ead_balance = or_dispto
        ead_total = ead_balance + ead_fuera
        rating = rng.randint(1, 16)
        pd_band = (0.0006, 0.02) if rating <= 4 else (0.01, 0.05) if rating <= 8 \
            else (0.03, 0.12) if rating <= 12 else (0.08, 0.18)
        pd_est = round(rng.uniform(*pd_band), 6)
        pd_suelo = 0.0005 if gar == "HIPOTECA" else 0.0003
        pd_final = max(pd_est, pd_suelo)
        lgd_est = round(rng.uniform(0.05, 0.92), 6)
        lgd_suelo = 0.30 if gar == "HIPOTECA" else (0.45 if seg == "CORP" else 0.0)
        lgd_con_moc = lgd_est * 1.05
        lgd_final = max(lgd_con_moc, lgd_suelo)
        k_irb = pd_final ** 0.5 * 0.06 + pd_final * 0.5
        ecl = pd_final * lgd_final * ead_total
        rwa = ead_total * lgd_final * 12.5 * k_irb
        dpds = rng.choice([rng.randint(0, 29), rng.randint(30, 89), rng.randint(90, 400)])
        stage = 1 if dpds < 30 else 2 if dpds < 90 else 3
        estado = "CERRADO" if rng.random() < 0.4 else "ESTIMACION"
        recup = round(rng.uniform(0, 0.55) * or_dispto, 2)
        coste = round(rng.uniform(0, 0.08) * or_dispto, 2)
        vcol = round(rng.uniform(120_000, 500_000), 2) if gar == "HIPOTECA" else \
            (round(rng.uniform(10_000, 90_000), 2) if gar == "PRENDA" else 0.0)
        haircut = round(rng.uniform(0.02, 0.08), 4) if vcol else 0.0
        ciclo = f"CIC_{i:06d}"
        yield {
            "t1_contratos": (cont, seg, gar, "S1", "MD"),
            "t2_basilea": (cont, mes, or_dispto, or_disble, ccf),
            "t3_colaterales": (cont, gar, vcol, haircut, round(vcol * (1 - haircut), 2)),
            "t4_ciclos": (ciclo, cont, mes, dpds, stage, pd_est, lgd_est, recup, coste, estado),
            "t5_pd_cal": (ciclo, pd_suelo, pd_final, min(1.0, 1.5 * pd_est)),
            "t6_lgd_cal": (ciclo, lgd_suelo, lgd_con_moc, lgd_final, min(1.0, lgd_est * 1.15)),
            "t7_ead_cal": (ciclo, cont, mes, ead_balance, ead_fuera, ead_total),
            "t8_final": (ciclo, cont, seg, ead_total, pd_final, lgd_final, ecl, rwa, ecl,
                   round(max(0.0, min(1.0, 1.0 - (recup - coste) / ead_total if ead_total else 0.0)), 6) if estado == "CERRADO" else 0.0,
                   estado),
        }


def build(path: str, size: int, seed: int, dirty: bool):
    t0 = time.perf_counter()
    conn = sqlite3.connect(path)
    for pragma in ("PRAGMA journal_mode=OFF", "PRAGMA synchronous=OFF",
                   "PRAGMA temp_store=MEMORY"):
        conn.execute(pragma)
    create_schema(conn)
    buckets = {t: [] for t in TABLES}
    dirty_pk = f"CIC_{seed:06d}" if dirty else None
    for r in rows(size, seed, dirty):
        for t in TABLES:
            buckets[t].append(r[t])
    for t, cols in TABLES.items():
        placeholders = ", ".join(["?"] * len(cols))
        conn.executemany(f'INSERT INTO "{t}" VALUES ({placeholders})', buckets[t])
        buckets[t] = None
    conn.commit()
    return conn, dirty_pk, time.perf_counter() - t0


def grade(conn, sql):
    t0 = time.perf_counter()
    cur = conn.execute(sql)
    rows = cur.fetchall()
    return len(rows), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[1_000_000, 10_000_000],
                    help="row counts (one process per scale recommended)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dirty", type=int, default=1, help="plant a bad EAD_FUERA row")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/sas-stress"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    HYP_A = ("SELECT ID_CICLO FROM t8_final "
             "WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > 0.01")
    HYP_B = ("SELECT e.ID_CICLO FROM t7_ead_cal e JOIN t2_basilea b "
             "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
             "WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01")

    out = []
    print(f"{'rows':>12} {'build':>9} {'file':>9} {'hypA':>9} {'hypB':>10} "
          f"{'hypB_rows':>10} {'correct':>7}")
    for size in args.rows:
        db = args.tmp / f"sas_{size}.db"
        if db.exists():
            db.unlink()
        try:
            conn, dirty_pk, build_t = build(str(db), size, args.seed, bool(args.dirty))
        except Exception as exc:  # noqa: BLE001
            print(f"{size:>12}  BUILD ERROR: {type(exc).__name__}: {exc}")
            out.append({"rows": size, "error": str(exc)})
            continue
        try:
            na, ta = grade(conn, HYP_A)
            nb, tb = grade(conn, HYP_B)
        finally:
            conn.close()
        fs = db.stat().st_size / 1e6
        rec = {"rows": size, "build_s": round(build_t, 2), "db_mb": round(fs, 1),
               "hypA_s": round(ta, 3), "hypB_s": round(tb, 3),
               "hypB_rows": nb, "hypA_rows": na,
               "correct": (na == 0 and nb == (1 if args.dirty else 0))}
        out.append(rec)
        print(f"{size:>12} {rec['build_s']:>9.2f} {rec['db_mb']:>9.1f} "
              f"{ta:>9.3f} {tb:>10.3f} {nb:>10} {str(rec['correct']):>7}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()