"""Create the demo's sample assets: two corresponding .egp projects + an .xlsx.

- ``src_basilea.egp``  — the upstream project that builds the *source* tables
  (contracts, Basilea exposure, collateral) correctly.
- ``rep_lgd.egp``      — the downstream *reporting* project that builds the EAD /
  final LGD table and carries the deep transformation bug (EAD_FUERA built from
  OR_DISPTO instead of OR_DISBLE).
- ``ciclos_sospechosos.xlsx`` — the Excel of "cycles" the captain has doubts
  about; one row (the deep-bug cycle) is flagged.

Each .egp is a ZIP (the SAS Enterprise Guide format) of the SAS programs plus a
``manifest.json`` describing the tables it produces and the transformation
layers, so the app can render the lineage.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "app_assets"

# The 7-10 layer lineage narrative (mirrors the deep pipeline).
LAYERS = [
    ("L1", "contracts", "t1_contratos: SEGMENTO / GARANTIA_TIPO per contract (drives PD_SUELO & LGD_SUELO)."),
    ("L2", "basilea", "t2_basilea: OR_DISPTO (drawn), OR_DISBLE (undrawn), CCF (Basel exposure)."),
    ("L3", "collateral", "t3_colaterales: VALOR_INICIAL, HAIRCUT -> VALOR_NETO (drives LGD floor)."),
    ("L4", "cycle state", "t4_ciclos: PD_ESTIMADA, LGD_ESTIMADA, DPDS, STAGE, recoveries/cost."),
    ("L5", "PD", "t5_pd_cal: PD_SUELO, PD_FINAL = MAX(PD_ESTIMADA, PD_SUELO)."),
    ("L6", "LGD", "t6_lgd_cal: LGD_SUELO, LGD_CON_MOC = LGD_ESTIMADA*1.05, LGD_FINAL = MAX(LGD_CON_MOC, LGD_SUELO)."),
    ("L7", "EAD", "t7_ead_cal: EAD_BALANCE = OR_DISPTO; EAD_FUERA = CCF*OR_DISBLE; EAD_TOTAL = sum."),
    ("L8", "ECL/RWA", "t8_final: ECL = PD_FINAL*LGD_FINAL*EAD_TOTAL; RWA; PROVISION; LGD_REALIZADA."),
]

# The transitive dependency web for the suspected ECL (what the app shows).
FIELD_DEPENDENCIES = {
    "ECL": ["PD_FINAL", "LGD_FINAL", "EAD_TOTAL"],
    "PD_FINAL": ["PD_ESTIMADA", "PD_SUELO"],
    "PD_SUELO": ["SEGMENTO", "GARANTIA_TIPO"],
    "LGD_FINAL": ["LGD_CON_MOC", "LGD_SUELO"],
    "LGD_SUELO": ["GARANTIA_TIPO", "SEGMENTO"],
    "LGD_CON_MOC": ["LGD_ESTIMADA"],
    "EAD_TOTAL": ["EAD_BALANCE", "EAD_FUERA"],
    "EAD_BALANCE": ["OR_DISPTO"],
    "EAD_FUERA": ["CCF", "OR_DISBLE"],
    "RWA": ["EAD_TOTAL", "LGD_FINAL", "K_IRB"],
    "PROVISION": ["ECL"],
}

SRC_SAS = r"""/* src_basilea.sas — build the correct source tables (contracts / Basel / collateral) */
LIBNAME mylib '/data/irb';
DATA mylib.t1_contratos;
    LENGTH ID_CONTRATO $32 SEGMENTO $16 GARANTIA_TIPO $16;
    INPUT ID_CONTRATO $ SEGMENTO $ GARANTIA_TIPO $ SECTOR $ DIMENSION $;
    DATALINES;
    ;
RUN;
DATA mylib.t2_basilea;
    LENGTH ID_CONTRATO $32; INPUT ID_CONTRATO $ MES OR_DISPTO OR_DISBLE CCF;
    DATALINES;
    ;
RUN;
DATA mylib.t3_colaterales;
    LENGTH ID_CONTRATO $32 GARANTIA_TIPO $16;
    INPUT ID_CONTRATO $ GARANTIA_TIPO $ VALOR_INICIAL HAIRCUT VALOR_NETO;
    DATALINES;
    ;
RUN;
"""

REP_SAS = r"""/* rep_lgd.sas — build EAD + final LGD table (BUG: EAD_FUERA uses OR_DISPTO) */
LIBNAME mylib '/data/irb';
DATA work.t7_ead_cal;
    SET mylib.t2_basilea;
    EAD_BALANCE = OR_DISPTO;
    /* BUG: should be CCF * OR_DISBLE; wrong input field used */
    EAD_FUERA = CCF * OR_DISPTO;
    EAD_TOTAL = EAD_BALANCE + EAD_FUERA;
RUN;
DATA mylib.t8_final;
    SET work.t7_ead_cal;
    ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL;
    RWA = EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB;
    PROVISION = ECL;
RUN;
"""


def make_egp(name: str, sas_code: str, tables: list[str],
             layers: list[tuple[str, str, str]], bug: str | None) -> Path:
    manifest = {
        "name": name,
        "tables_produced": tables,
        "layers": [{"id": i, "name": n, "detail": d} for i, n, d in layers],
        "field_dependencies": FIELD_DEPENDENCIES,
        "known_bug": bug,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}.egp"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("programs/src.sas", sas_code)
    return path


def make_xlsx(rows: list[dict]) -> Path:
    from openpyxl import Workbook
    ASSETS.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "ciclos_sospechosos"
    cols = list(rows[0].keys())
    ws.append(cols)
    for r in rows:
        ws.append([r[c] for c in cols])
    path = ASSETS / "ciclos_sospechosos.xlsx"
    wb.save(path)
    return path


def main() -> None:
    # Reuse the deep pipeline's generator to produce a real suspect cycle and a
    # realistic sample of cycles for the Excel.
    import sys
    sys.path.insert(0, str(HERE))
    from deep_debug_run import build_trap, build_clean

    trap, dirty_pk = build_trap(1500, 7, 3)
    clean = build_clean(1500, 7)
    # Suspect cycle row (the deep-bug cycle) + a sample of clean cycles.
    dirty_row = dict(zip(
        ["ID_CICLO", "ID_CONTRATO", "SEGMENTO", "EAD_TOTAL", "PD_FINAL",
         "LGD_FINAL", "ECL", "RWA", "PROVISION", "LGD_REALIZADA", "ESTADO"],
        trap.execute("SELECT * FROM t8_final WHERE ID_CICLO=?",
                     (dirty_pk,)).fetchone()))
    clean_rows = []
    for r in trap.execute("SELECT * FROM t8_final LIMIT 8").fetchall():
        clean_rows.append(dict(zip(
            ["ID_CICLO", "ID_CONTRATO", "SEGMENTO", "EAD_TOTAL", "PD_FINAL",
             "LGD_FINAL", "ECL", "RWA", "PROVISION", "LGD_REALIZADA", "ESTADO"], r)))
    rows = clean_rows + [dirty_row]
    for r in rows:
        r.setdefault("FLAG", "OK")
    rows[-1]["FLAG"] = "DUDA"  # the captain's flagged cycle

    src = make_egp("src_basilea", SRC_SAS,
                   ["t1_contratos", "t2_basilea", "t3_colaterales"],
                   LAYERS[:3], None)
    rep = make_egp("rep_lgd", REP_SAS,
                   ["t7_ead_cal", "t8_final"], LAYERS[6:],
                   "t7_ead_cal.EAD_FUERA built from OR_DISPTO instead of OR_DISBLE")
    xls = make_xlsx(rows)

    trap.close(); clean.close()
    print(f"wrote {src}\nwrote {rep}\nwrote {xls}")


if __name__ == "__main__":
    main()
