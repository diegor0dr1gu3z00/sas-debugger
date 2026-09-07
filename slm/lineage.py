"""Compact schema + lineage summary used as model context in each episode.

The narrative mirrors the vendored reference ``ciclos_calibrados_pipeline.sas``
so the model sees the same 7 transformation layers a real SAS project would
produce.  It is plain text: small SLMs must not be asked to parse an AST.
"""

from __future__ import annotations

import sqlite3

_LAYERS = (
    ("L0", "raw sources",
     "CICLOS (one row per recovery cycle), CONTRATOS (contract/fusion flags), "
     "BASILEA_MENSUAL (monthly Basel exposure: OR_EAD/OR_DISPTO/OR_DISBLE), "
     "COLATERALES (collateral valuation)."),
    ("L1", "staging",
     "Type coercion + completeness fix; COALESCE blanks to missing for PD, LGD, "
     "recoveries; normalise ADJUDICACION_FLAG to '0'/'1'."),
    ("L2", "fusion + contract enrichment",
     "MERGE CICLOS -> CONTRATOS on ID_CONTRATO; keep all cycles (IF a); "
     "COALESCE SW_FUSION to 0."),
    ("L3", "Basilea OR_EAD join (cross-table)",
     "LEFT JOIN de-duplicated BASILEA (MAX aggregation on ID_FUSION_FINAL+period) "
     "for fused groups, plus a direct contract join; pulls OR_EAD/OR_DISPTO/"
     "VALOR_COLATERAL_INICIAL/HAIRCUT into the cycle."),
    ("L4", "PD calibration + floors",
     "PD_SUELO depends on segment (RETAIL_HIP 0.0005 else 0.0003); "
     "PD_FINAL = MAX(PD_ESTIMADA, PD_SUELO); PD_DOWNTURN = MIN(1, PD_ESTIMADA*1.5)."),
    ("L5", "LGD floors / MoC",
     "LGD_SUELO by collateral (HIPOTECA 0.30, CORP 0.45, else 0); "
     "MOC = 0.05*LGD_ESTIMADA; LGD_CON_MOC = LGD_ESTIMADA + MOC; "
     "LGD_FINAL = MAX(LGD_CON_MOC, LGD_SUELO)."),
    ("L6", "EAD / CCF",
     "CCF_ESTIMADO = 0.75; EAD_BALANCE = OR_DISPTO (or SALDO_PENDIENTE); "
     "EAD_FUERA_BALANCE = CCF_ESTIMADO*OR_DISBLE; EAD_TOTAL = balance + off-balance."),
    ("L7", "ECL / RWA / staging (final)",
     "K_IRB = SQRT(PD_FINAL)*0.06 + PD_FINAL*0.5; "
     "RWA = EAD_TOTAL*LGD_FINAL*12.5*K_IRB; ECL = PD_FINAL*LGD_FINAL*EAD_TOTAL; "
     "PROVISION = ECL; IFRS-9 backstop DPDS>=30 & STAGE=1 -> STAGE=2."),
)

_PIPELINE_TEMPLATE = "\n".join(
    f"{mark}  {name:<24s} {desc}" for mark, name, desc in _LAYERS
)


def pipeline_lineage_text() -> str:
    """Human-readable lineage narrative for the model context."""
    return (
        "PIPELINE LINEAGE (ciclos_calibrados is the final table; every field has\n"
        ">= 7 hops from the raw sources):\n" + _PIPELINE_TEMPLATE
    )


def schema_text(schema: dict[str, list[str]]) -> str:
    """Compact, model-friendly rendering of the table schemas."""
    lines = ["TABLES PRESENT (schema field:type):"]
    for table in sorted(schema):
        cols = ", ".join(schema[table])
        lines.append(f"  {table}  ({cols})")
    return "\n".join(lines)


def suspect_row_text(row: dict, columns: tuple[str, ...]) -> str:
    """The suspected cycle's observed values, up to a budget."""
    items = [f"{c}={row[c]}" for c in columns if c in row]
    return "| ".join(items)


def fetch_row(conn: sqlite3.Connection, pk: str,
              table: str) -> dict | None:
    """Fetch a single row by primary key from ``table``."""
    try:
        cur = conn.execute(
            f'SELECT * FROM "{table}" WHERE "ID_CONTR_CICLO_LGD" = ?', (pk,))
        r = cur.fetchone()
        if r is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, r))
    except Exception:
        return None
