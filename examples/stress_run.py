"""Stress test the deep pipeline at scale — timing + correctness.

Reuses the exact builders in ``deep_debug_run`` (8-table trap + clean DBs) but at
arbitrary row counts, then runs the real oracle grade on the two hypotheses
(shallow leaf formula, deep cross-table EAD) and reports timings + correctness.

Run:
    .venv/bin/python examples/stress_run.py [--rows N ...] [--seed S] [--idx I]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_debug_run import build_clean, build_trap, exec_sql, grade  # noqa: E402

SCOPE = ["build_clean", "build_trap", "hypA_leaf", "hypB_cross", "total"]


def bench(name: str, fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    return out, dt


def run_scale(rows: int, seed: int, dirty_idx: int) -> dict:
    rec = {"rows": rows, "seed": seed, "dirty_idx": dirty_idx}

    clean, t_clean = bench("build_clean", build_clean, rows, seed)
    rec["build_clean"] = t_clean
    rec["clean_rows"] = clean.execute(
        "SELECT COUNT(*) FROM t8_final").fetchone()[0]

    trap_data, t_trap = bench("build_trap", build_trap, rows, seed, dirty_idx)
    trap, _dirty_pk = trap_data
    rec["build_trap"] = t_trap
    rec["trap_rows"] = trap.execute(
        "SELECT COUNT(*) FROM t8_final").fetchone()[0]
    dirty_ciclo = trap.execute("SELECT ID_CICLO FROM t8_final "
                               "WHERE ID_CICLO LIKE '%DIRTY%'").fetchone()[0]

    # Hypothesis A — shallow leaf formula (valid runs, catches nothing)
    sql_a = ("SELECT ID_CICLO FROM t8_final "
             "WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > 0.01")
    (g_a), t_a = bench("hypA", grade, clean, trap, dirty_ciclo, sql_a)
    rec["hypA"] = t_a
    rec["hypA_ok"] = g_a

    # Hypothesis B — deep cross-table reconciliation (catches the planted bug)
    sql_b = ("SELECT e.ID_CICLO FROM t7_ead_cal e JOIN t2_basilea b "
             "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
             "WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01")
    (g_b), t_b = bench("hypB", grade, clean, trap, dirty_ciclo, sql_b)
    rec["hypB"] = t_b
    rec["hypB_ok"] = g_b
    rec["hypB_true"] = {"valid": g_b["valid"], "caught": g_b["caught"],
                        "false_positive": g_b["false_positive"]}

    rec["select_trap_via_B"] = smaller_query_bench(trap, rows)

    clean.close(); trap.close()
    rec["total"] = rec["build_clean"] + rec["build_trap"] + rec["hypB"]
    rec["rows"] = rows
    return rec


def smaller_query_bench(trap, rows: int) -> float:
    """Time a realistic key-table extraction read (join + where) as a proxy for
    grading on large DBs (the SELECT itself, not 2x full-table scans)."""
    _, dt = bench("extract", exec_sql, trap,
                  "SELECT e.ID_CICLO FROM t7_ead_cal e JOIN t2_basilea b "
                  "ON e.ID_CONTRATO=b.ID_CONTRATO AND e.MES=b.MES "
                  "WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01 "
                  "LIMIT 10")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[1_500, 25_000, 250_000, 1_000_000],
                    help="row counts to test (comma allowed: 1500,25000)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dirty_idx", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None,
                    help="write ONE result record to this file (one scale per "
                         "process avoids cumulative in-memory SQLite pages)")
    ap.add_argument("--memory", action="store_true",
                    help="report peak RSS and exit without grading (probe)")
    args = ap.parse_args()

    print(f"{'rows':>12} {'build_clean':>11} {'build_trap':>11} "
          f"{'hypA':>9} {'hypB':>9} {'extract':>9} {'total':>9} "
          f"{'catch':>6} {'FP':>5}")
    if args.memory:
        # Probe peak RSS for the clean+trap pair at one scale, then exit.
        import resource
        import os
        t = time.perf_counter()
        c = build_clean(args.rows[0], args.seed)
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        tr, _ = build_trap(args.rows[0], args.seed, args.dirty_idx)
        r2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        c.close(); tr.close()
        print(json.dumps({"rows": args.rows[0], "clean_s": round(time.perf_counter() - t, 2),
                          "peak_rss_after_clean_gb": round(r1, 2),
                          "peak_rss_both_gb": round(r2, 2)}))
        return

    results = []
    for rows in args.rows:
        try:
            r = run_scale(rows, args.seed, args.dirty_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"{rows:>12}  ERROR: {type(exc).__name__}: {exc}")
            results.append({"rows": rows, "error": str(exc)})
            continue
        results.append(r)
        ok = r["hypB_true"]
        print(f"{r['rows']:>12} {r['build_clean']:>11.3f} {r['build_trap']:>11.3f} "
              f"{r['hypA']:>9.3f} {r['hypB']:>9.3f} {r['hypB']:>9.3f} "
              f"{r['total']:>9.3f} "
              f"{str(ok['caught']):>6} {str(ok['false_positive']):>5}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()