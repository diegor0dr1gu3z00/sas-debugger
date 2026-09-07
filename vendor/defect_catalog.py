# VENDORED_PRIOR_ART
# ==================
# This file is vendored (copied verbatim, then adapted in place only where
# noted) from the RegLLM prior-art project at ~/Documents/PwC/regllm
# (DQC/eval/defect_catalog.py). It is the ground-truth defect catalog the
# SFT/GRPO oracle grades against. Attribution: RegLLM development team.
# Original module docstring follows.

"""Ground-truth defect catalog for the DQC eval harness.

Each ``Defect`` pairs a *coherence invariant* (what a good DQC must verify)
with a deterministic ``mutate(row)`` that plants exactly that incoherence into
one clean row, AND a ``dimension`` tag from the industry data-quality
taxonomies (DAMA / ISO 8000 / BCBS 239). The catalog is the oracle the eval
harness scores against:

  * **recall**    — does the agent's SQL return >=1 row on the mutated row?
  * **precision** — does the agent's SQL return 0 rows on the clean DB?
  * **oracle_sql** — the reference check that catches the defect (sanity).

Dimension coverage (so the harness can surface *model deficiencies* by area):

  completeness | validity | accuracy | consistency | timeliness
  uniqueness   | plausibility | conformity

Every oracle is verified by the harness self-test to return 0 rows on a clean
DB (built by ``generate_db.build_clean_db``) and >=1 row on its own trap.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

TABLE_NAME = "ciclos_calibrados"
PK_COLUMN = "ID_CONTR_CICLO_LGD"

# DQ dimensions (DAMA / ISO 8000 / BCBS 239)
DIMENSIONS = (
    "completeness", "validity", "accuracy", "consistency", "timeliness",
    "uniqueness", "plausibility", "conformity",
)

# Column type affinity used by generate_db when materializing SQLite.
REAL_COLS = {
    "PD_ESTIMADA", "PD_SUELO", "PD_FINAL", "PD_DOWNTURN",
    "LGD_ESTIMADA", "LGD_REALIZADA", "LGD_SUELO", "MOC", "LGD_CON_MOC", "LGD_FINAL",
    "MOC_CAT_A", "MOC_CAT_B", "MOC_CAT_C", "LGD_DOWNTURN",
    "OR_EAD", "OR_DISPTO", "OR_DISBLE", "SALDO_PENDIENTE", "EAD",
    "EAD_BALANCE", "CCF_ESTIMADO", "EAD_FUERA_BALANCE", "EAD_TOTAL",
    "EAD_TOTAL_EUR", "TIPO_CAMBIO",
    "VALOR_COLATERAL_INICIAL", "VALOR_COLATERAL", "HAIRCUT", "LTV",
    "K_IRB", "M_VENCIMIENTO", "RWA", "ECL", "PROVISION",
    "TIPO_INTERES_ORIGINAL", "TIPO_INTERES_ACTUAL", "INTERESES_ACUMULADOS",
    "ADJUDICACION_VALOR", "RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO",
    "TASA_DESCUENTO",
}
INT_COLS = {
    "MES_CICLO", "RATING_GRADO", "DPDS", "STAGE_IFRS9", "CURE_FLAG", "SW_FUSION",
    "MES_DEFAULT", "MES_CIERRE_CICLO", "MES_VALORACION_COLATERAL",
    "VENTANA_OBSERVACION_YEARS", "FLAG_NC",
}

FORMULA_EPS = 0.01  # tolerance for floating formula invariants
ALLOWED_SEGMENTS = ("CORP", "SME", "RETAIL_HIP", "RETAIL_CONS")
REFERENCE_PERIOD = 202412  # data beyond this is "future"/stale


@dataclass(frozen=True)
class Defect:
    defect_id: str
    dimension: str        # one of DIMENSIONS
    category: str         # formula | consistencia | referencial | cross_table | rango
    severity: str         # HIGH | MED | LOW
    description: str      # natural-language spec shown to the agent
    columns: tuple[str, ...]
    oracle_sql: str       # the reference check — returns violating rows
    mutate: Callable[[dict], dict] = field(default=lambda r: dict(r), repr=False)
    regulation_ref: str = ""
    decoy: bool = False   # True => single-column, must score r_coherence=0
    plants_duplicate_pk: bool = False  # planted as a verbatim PK duplicate
    plants_from_existing: bool = False  # planted by mutating a real clean row
    pick_where: str = ""  # SQL filter on eligible clean rows (cross-table)
    cross_table: bool = False  # oracle joins ciclos_calibrados to a source table
    target_table: str = "ciclos_calibrados"  # table the defect is planted into
    # For panel defects: mutate a clean monthly series (list[dict], rng) -> series
    panel_mutate: Callable[[list, Any], list] = field(
        default=lambda s, rng: s, repr=False)


# ── mutators — each plants exactly its own incoherence into a clean row ──────

def _d01_ecl(row):  # consistency — formula
    r = dict(row); r["ECL"] = r["PD_FINAL"] * r["LGD_FINAL"] * r["EAD_TOTAL"] + 1000.0; return r

def _d02_rwa(row):
    r = dict(row); r["RWA"] = r["EAD_TOTAL"] * r["LGD_FINAL"] * 12.5 * r["K_IRB"] + 5000.0; return r

def _d03_pd_floor(row):
    r = dict(row); r["PD_FINAL"] = r["PD_SUELO"] / 2; return r

def _d04_lgd_floor(row):
    r = dict(row); r["LGD_SUELO"] = 0.45; r["LGD_FINAL"] = 0.20; return r

def _d05_pd_max(row):
    r = dict(row); r["PD_FINAL"] = r["PD_ESTIMADA"] / 2; return r

def _d06_stage3_low_dpd(row):
    r = dict(row); r["STAGE_IFRS9"] = 3; r["DPDS"] = 5; return r

def _d07_valor_sin_flag(row):
    r = dict(row); r["ADJUDICACION_VALOR"] = 50000.0; r["ADJUDICACION_FLAG"] = "0"; return r

def _d08_flag_sin_valor(row):
    r = dict(row); r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_VALOR"] = 0.0; return r

def _d09_valor_sin_tipo(row):
    r = dict(row); r["ADJUDICACION_VALOR"] = 12345.0; r["ADJUDICACION_TIPO"] = ""; return r

def _d10_recuperacion_sobre_ead(row):  # plausibility
    r = dict(row)
    ead = float(r.get("EAD_TOTAL") or 0)
    r["RECUPERACION_ACUMULADA"] = ead * 1.8 if ead > 0 else 1_000_000.0
    r["COSTE_TOTAL_ACUMULADO"] = 0.0
    return r

def _d11_cerrado_sin_terminacion(row):
    r = dict(row); r["ESTADO_CICLO"] = "CERRADO"; r["TERMINACION"] = ""; return r

def _d12_ead_total(row):
    r = dict(row); r["EAD_TOTAL"] = r["EAD_BALANCE"] + r["EAD_FUERA_BALANCE"] + 5000.0; return r

def _d13_ead_fuerasaldo(row):
    r = dict(row); r["EAD_FUERA_BALANCE"] = r["CCF_ESTIMADO"] * r["OR_DISBLE"] + 999.0; return r

def _d14_retail_hip_sin_hipoteca(row):  # conformity (cross-table)
    r = dict(row); r["SEGMENTO"] = "RETAIL_HIP"; r["COLATERAL_TIPO"] = "NINGUNA"; return r

def _d15_hipoteca_suelo_bajo(row):
    r = dict(row); r["COLATERAL_TIPO"] = "HIPOTECA"; r["LGD_SUELO"] = 0.10; return r

def _d16_kirb(row):
    r = dict(row); r["K_IRB"] = math.sqrt(r["PD_FINAL"]) * 0.06 + r["PD_FINAL"] * 0.5 + 0.05; return r

def _d17_fusion_dup(row):  # uniqueness / cardinality
    r = dict(row); r["SW_FUSION"] = 1; r["EAD_BALANCE"] = r["OR_EAD"] * 2.0; return r

def _d18_pd_nulo_activo(row):  # completeness
    r = dict(row); r["ESTADO_CICLO"] = "ESTIMACION"; r["PD_ESTIMADA"] = None; return r

def _d19_segmento_invalido(row):  # validity
    r = dict(row); r["SEGMENTO"] = "WHOLESALE"; return r

def _d20_stage_invalido(row):  # validity
    r = dict(row); r["STAGE_IFRS9"] = 5; return r

def _d21_mes_futuro(row):  # timeliness
    r = dict(row); r["MES_CICLO"] = 210012; return r

def _d22_ead_vs_basilea(row):  # accuracy vs authoritative source
    r = dict(row); r["EAD_BALANCE"] = r["OR_DISPTO"] + 100000.0; return r

def _d24_fusion_flag_inconsistente(row):  # conformity / integrity
    r = dict(row); r["SW_FUSION"] = 0; r["ID_FUSION_FINAL"] = "FUS_GHOST"; return r

def _d25_grade_pd_no_monotono(row):  # plausibility / monotonicity
    r = dict(row); r["RATING_GRADO"] = 16; r["PD_ESTIMADA"] = 0.002; return r

def _month_shift(yyyymm: int, months: int) -> int:
    total = (yyyymm // 100) * 12 + (yyyymm % 100) - 1 + months
    return (total // 12) * 100 + (total % 12) + 1


def _force_secured(r: dict) -> dict:
    """Make a row secured without tripping other collateral invariants.

    RETAIL_HIP rows are already HIPOTECA in the clean generator; everything
    else becomes PRENDA (which carries no LGD-floor constraint, so D14/D15
    stay silent) with a plausible, *fresh* valuation.
    """
    if r["COLATERAL_TIPO"] == "NINGUNA":
        r["COLATERAL_TIPO"] = "PRENDA"
        r["VALOR_COLATERAL_INICIAL"] = 50_000.0
        r["HAIRCUT"] = 0.10
        r["VALOR_COLATERAL"] = 45_000.0
        r["MES_VALORACION_COLATERAL"] = r["MES_CICLO"]
    return r


def _d26_downturn_bajo(row):  # consistency — downturn LGD undercuts long-run
    r = dict(row); r["LGD_DOWNTURN"] = r["LGD_ESTIMADA"] * 0.7; return r

def _d27_moc_sum(row):  # consistency — MoC != sum of categories
    r = dict(row); r["MOC_CAT_A"] = (r["MOC_CAT_A"] or 0.0) + 0.05; return r

def _d28_ventana_sin_flag(row):  # conformity — short window not flagged
    r = dict(row); r["VENTANA_OBSERVACION_YEARS"] = 3; r["FLAG_NC"] = 0; return r

def _d29_valoracion_estancada(row):  # timeliness — stale collateral valuation
    r = _force_secured(dict(row))
    r["MES_VALORACION_COLATERAL"] = _month_shift(r["MES_CICLO"], -48)
    return r

def _d30_fx_mal_convertido(row):  # accuracy — EUR conversion broken
    r = dict(row)
    r["EAD_TOTAL_EUR"] = r["EAD_TOTAL"] * (r["TIPO_CAMBIO"] or 1.0) + 25_000.0
    return r

def _d31_cierre_antes_default(row):  # plausibility — impossible date order
    r = dict(row)
    r["ESTADO_CICLO"] = "CERRADO"
    r["CURE_FLAG"] = 0
    r["TERMINACION"] = "FALLIDO"
    r["MES_CIERRE_CICLO"] = _month_shift(r["MES_DEFAULT"], -3)
    return r

def _d32_cura_sin_flag(row):  # conformity — cured closure without cure flag
    r = dict(row)
    r["ESTADO_CICLO"] = "CERRADO"
    r["TERMINACION"] = "CURA"
    r["CURE_FLAG"] = 0
    r["MES_CIERRE_CICLO"] = r["MES_CIERRE_CICLO"] or _month_shift(r["MES_DEFAULT"], 6)
    return r

def _d34_divisa_invalida(row):  # validity — currency outside domain
    r = dict(row); r["DIVISA"] = "XXX"; return r

def _d35_colateral_sin_valor(row):  # completeness — secured without valuation
    r = _force_secured(dict(row))
    r["VALOR_COLATERAL"] = None
    return r


def _d36_provision_ecl(row):  # consistency — provision decoupled from ECL
    r = dict(row); r["PROVISION"] = r["ECL"] + 500.0; return r

def _d37_pd_downturn_bajo(row):  # consistency — downturn PD under estimate
    r = dict(row); r["PD_DOWNTURN"] = r["PD_ESTIMADA"] / 2; return r

def _d38_lgd_con_moc(row):  # consistency — MoC not applied to LGD
    r = dict(row); r["LGD_CON_MOC"] = r["LGD_ESTIMADA"]; return r

def _d39_vencimiento_fuera(row):  # validity — maturity outside [1,5]
    r = dict(row); r["M_VENCIMIENTO"] = 7.0; return r

def _d40_ead_alias(row):  # consistency — EAD alias diverges from EAD_TOTAL
    r = dict(row); r["EAD"] = r["EAD_TOTAL"] + 1_000.0; return r

def _d41_descuento_cero(row):  # plausibility — recoveries without discount rate
    r = dict(row)
    r["TASA_DESCUENTO"] = 0.0
    r["RECUPERACION_ACUMULADA"] = min(5_000.0, 0.5 * (r["EAD_TOTAL"] or 10_000.0))
    return r

def _d42_causa_sin_dpd(row):  # conformity — 90-DPD cause with low DPDs
    r = dict(row); r["CAUSA_DEFAULT"] = "90_DIAS_VENCIDO"; r["DPDS"] = 10
    r["STAGE_IFRS9"] = 1  # keep D06/D20 silent: stage 1 is valid for DPDS=10
    return r

def _d43_tipo_persona(row):  # conformity — legal nature vs segment
    r = dict(row)
    r["TIPO_PERSONA"] = "F" if r["SEGMENTO"] in ("CORP", "SME") else "J"
    return r

def _d44_calibration_segment(row):  # conformity — calibration bucket unmapped
    r = dict(row); r["CALIBRATION_SEGMENT"] = "UNMAPPED_X"; return r


# ── cross-table mutators (operate on a copy of a real clean row) ────────────

def _d48_orphan_contrato(row):  # referential integrity — no parent contract
    r = dict(row); r["ID_CONTRATO"] = "CONT_GHOST"; return r

def _d49_or_ead_reconc(row):  # accuracy — OR_EAD disagrees with BASILEA source
    r = dict(row)
    r["OR_EAD"] = (r["OR_EAD"] or 0.0) + 50_000.0
    r["SW_FUSION"] = 0            # keep D17 (fusion dedup) silent
    r["ID_FUSION_FINAL"] = None   # keep D24 (fusion flag) silent
    return r

def _d50_segmento_reconc(row):  # consistency — SEGMENTO differs from CONTRATOS
    r = dict(row)
    new = "CORP" if r["SEGMENTO"] != "CORP" else "SME"
    r["SEGMENTO"] = new
    r["TIPO_PERSONA"] = "J"          # both CORP and SME are legal persons (D43 ok)
    r["CALIBRATION_SEGMENT"] = f"{new}_MED"   # keep D44 (bucket prefix) silent
    return r

def _d51_cliente_reconc(row):  # consistency — ID_CLIENTE differs from CONTRATOS
    r = dict(row); r["ID_CLIENTE"] = "CLI_MISMATCH"; return r

def _d52_colateral_valor_reconc(row):  # accuracy — collateral value vs source
    r = dict(row)
    r["VALOR_COLATERAL_INICIAL"] = (r["VALOR_COLATERAL_INICIAL"] or 0.0) + 100_000.0
    return r

def _d53_fecha_alta_reconc(row):  # consistency — origination date vs CONTRATOS
    r = dict(row); r["FECHA_ALTA_CONTRATO"] = "1900-01-01"; return r


# ── date-interrelation mutators ─────────────────────────────────────────────

def _d54_alta_tras_default(row):  # plausibility — contract opened after default
    r = dict(row); r["FECHA_ALTA_CONTRATO"] = "2099-12-31"; return r

def _d55_adj_fuera_ventana(row):  # plausibility — foreclosure before default
    r = dict(row)
    r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_TIPO"] = "SUBASTA"
    r["ADJUDICACION_VALOR"] = 30_000.0
    r["FECHA_ADJUDICACION"] = "1990-01-01"   # long before any default
    return r

def _d56_venta_antes_adj(row):  # plausibility — sold before foreclosed
    r = dict(row)
    r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_TIPO"] = "DACCION"
    r["ADJUDICACION_VALOR"] = 40_000.0
    r["FECHA_ADJUDICACION"] = "2023-06-01"
    r["FECHA_VENTA_COLATERAL"] = "2023-01-01"
    return r

def _d57_fecha_periodo(row):  # consistency — FECHA_DEFAULT month != MES_DEFAULT
    r = dict(row); r["FECHA_DEFAULT"] = "2000-07-15"; return r


# ── panel (monthly evolution) mutators — operate on a clean series ──────────

def _pd58_recup_baja(series, rng):  # recovery cumulative decreases MoM
    s = [dict(r) for r in series]
    s[1]["RECUPERACION_ACUMULADA"] = s[0]["RECUPERACION_ACUMULADA"] - 500.0
    return s

def _pd59_hueco_mes(series, rng):  # a month is missing from the series
    s = [dict(r) for r in series]
    for j in range(1, len(s)):
        s[j]["MES_CICLO"] = _month_shift(s[j]["MES_CICLO"], 1)
    return s

def _pd60_mes_duplicado(series, rng):  # same month appears twice for a cycle
    s = [dict(r) for r in series]
    s.append(dict(s[1]))
    return s

def _pd61_stage3_pd_baja(series, rng):  # STAGE=3 (default) with PD < 1.0
    s = [dict(r) for r in series]
    s[0]["PD_ESTIMADA"] = 0.5            # seq 1 is always in default
    return s

def _pd62_regresion_stage(series, rng):  # stage improves without a cure
    s = [dict(r) for r in series]
    s[-1]["STAGE_IFRS9"] = 1
    s[-1]["CURE_FLAG"] = 0
    return s

def _pd63_dpd_baja(series, rng):  # DPD falls without a cure
    s = [dict(r) for r in series]
    s[-1]["DPDS"] = max(0, s[-2]["DPDS"] - 40)
    s[-1]["STAGE_IFRS9"] = 3
    s[-1]["CURE_FLAG"] = 0
    s[-1]["PD_ESTIMADA"] = 1.0
    return s


# ── "weird data" cross-domain mutators (main table) ─────────────────────────

def _d64_colateral_en_tarjeta(row):  # conformity — a credit card with collateral
    r = dict(row)
    r["SEGMENTO"] = "RETAIL_CONS"; r["TIPO_PERSONA"] = "F"
    r["CALIBRATION_SEGMENT"] = "RETAIL_CONS_MED"
    r["PRODUCTO"] = "TARJETA"; r["COLATERAL_TIPO"] = "PRENDA"
    return r

def _d65_hipoteca_sin_garantia(row):  # conformity — mortgage loan, no mortgage
    r = dict(row)
    r["SEGMENTO"] = "RETAIL_CONS"; r["TIPO_PERSONA"] = "F"
    r["CALIBRATION_SEGMENT"] = "RETAIL_CONS_MED"
    r["PRODUCTO"] = "HIPOTECA_LN"; r["COLATERAL_TIPO"] = "NINGUNA"
    return r

def _d66_adjudicacion_sin_colateral(row):  # conformity — foreclose the unsecured
    r = dict(row)
    r["COLATERAL_TIPO"] = "NINGUNA"
    r["ADJUDICACION_FLAG"] = "1"; r["ADJUDICACION_TIPO"] = "SUBASTA"
    r["ADJUDICACION_VALOR"] = 30_000.0
    return r


def _d45_haircut_implausible(row):  # plausibility — haircut out of band
    r = _force_secured(dict(row)); r["HAIRCUT"] = 0.90; return r

def _d46_cliente_nulo(row):  # completeness — active cycle without client
    r = dict(row); r["ESTADO_CICLO"] = "ESTIMACION"; r["TERMINACION"] = ""
    r["MES_CIERRE_CICLO"] = None; r["CURE_FLAG"] = 0; r["ID_CLIENTE"] = None
    return r

def _d47_lgd_realizada_formula(row):  # accuracy — realised LGD off-formula
    r = dict(row)
    r["ESTADO_CICLO"] = "CERRADO"
    if not r.get("TERMINACION"):
        r["TERMINACION"] = "FALLIDO"; r["CURE_FLAG"] = 0
    r["LGD_REALIZADA"] = 0.01  # clean closed cycles sit in [0.45, 1.0]
    return r


def _da_ead_cero(row):  # validity decoy
    r = dict(row); r["EAD_TOTAL"] = 0.0; return r

def _db_lgd_realizada_neg(row):  # validity decoy
    r = dict(row); r["LGD_REALIZADA"] = -0.5; return r


DEFECTS: list[Defect] = [
    # ── consistency (cross-field / formula invariants) ───────────────────────
    Defect("D01", "consistency", "formula", "HIGH",
           "El ECL no coincide con la fórmula canónica ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL.",
           ("ECL", "PD_FINAL", "LGD_FINAL", "EAD_TOTAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > {FORMULA_EPS}",
           _d01_ecl, "IFRS 9 / CRR Art. 158"),
    Defect("D02", "consistency", "formula", "HIGH",
           "El RWA no coincide con RWA = EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB.",
           ("RWA", "EAD_TOTAL", "LGD_FINAL", "K_IRB"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(RWA - EAD_TOTAL*LGD_FINAL*12.5*K_IRB) > 1.0",
           _d02_rwa, "CRR Art. 153"),
    Defect("D03", "consistency", "referencial", "HIGH",
           "La PD final vulnera su suelo regulatorio: PD_FINAL < PD_SUELO.",
           ("PD_FINAL", "PD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_FINAL < PD_SUELO",
           _d03_pd_floor, "CRR Art. 160.1"),
    Defect("D04", "consistency", "referencial", "HIGH",
           "La LGD final vulnera su suelo por colateral: LGD_FINAL < LGD_SUELO.",
           ("LGD_FINAL", "LGD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE LGD_FINAL < LGD_SUELO",
           _d04_lgd_floor, "CRR Art. 161.1"),
    Defect("D05", "consistency", "consistencia", "MED",
           "La PD final es menor que la PD estimada: el suelo (max) no se aplicó.",
           ("PD_FINAL", "PD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_FINAL < PD_ESTIMADA",
           _d05_pd_max, "CRR Art. 160.1"),
    Defect("D06", "consistency", "consistencia", "HIGH",
           "Ciclo en STAGE_IFRS9=3 (default) con DPDS < 30 días.",
           ("STAGE_IFRS9", "DPDS"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE STAGE_IFRS9=3 AND DPDS<30",
           _d06_stage3_low_dpd, "CRR Art. 178.1(b)"),
    Defect("D07", "consistency", "consistencia", "HIGH",
           "Adjudicación con valor > 0 cuyo flag no está activo (= '0').",
           ("ADJUDICACION_VALOR", "ADJUDICACION_FLAG"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_VALOR>0 AND ADJUDICACION_FLAG='0'",
           _d07_valor_sin_flag),
    Defect("D08", "consistency", "consistencia", "MED",
           "Flag de adjudicación activo ('1') pero valor = 0.",
           ("ADJUDICACION_FLAG", "ADJUDICACION_VALOR"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_FLAG='1' AND ADJUDICACION_VALOR=0",
           _d08_flag_sin_valor),
    Defect("D09", "consistency", "consistencia", "HIGH",
           "Adjudicación con valor > 0 cuyo tipo está vacío.",
           ("ADJUDICACION_VALOR", "ADJUDICACION_TIPO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ADJUDICACION_VALOR>0 AND (ADJUDICACION_TIPO='' OR ADJUDICACION_TIPO IS NULL)",
           _d09_valor_sin_tipo),
    Defect("D11", "consistency", "consistencia", "HIGH",
           "Ciclo CERRADO sin causa de TERMINACION informada.",
           ("ESTADO_CICLO", "TERMINACION"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ESTADO_CICLO='CERRADO' AND (TERMINACION='' OR TERMINACION IS NULL)",
           _d11_cerrado_sin_terminacion),
    Defect("D12", "consistency", "formula", "MED",
           "EAD_TOTAL no coincide con EAD_BALANCE + EAD_FUERA_BALANCE.",
           ("EAD_TOTAL", "EAD_BALANCE", "EAD_FUERA_BALANCE"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(EAD_TOTAL - (EAD_BALANCE+EAD_FUERA_BALANCE)) > {FORMULA_EPS}",
           _d12_ead_total, "CRR Art. 166"),
    Defect("D13", "consistency", "formula", "MED",
           "EAD_FUERA_BALANCE no coincide con CCF_ESTIMADO * OR_DISBLE cuando hay undrawn.",
           ("EAD_FUERA_BALANCE", "CCF_ESTIMADO", "OR_DISBLE"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE OR_DISBLE>0 AND ABS(EAD_FUERA_BALANCE - CCF_ESTIMADO*OR_DISBLE) > {FORMULA_EPS}",
           _d13_ead_fuerasaldo, "CRR Art. 166.8"),
    Defect("D15", "consistency", "referencial", "HIGH",
           "Colateral HIPOTECA cuyo LGD_SUELO es inferior al mínimo del 30%.",
           ("COLATERAL_TIPO", "LGD_SUELO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE COLATERAL_TIPO='HIPOTECA' AND LGD_SUELO<0.30",
           _d15_hipoteca_suelo_bajo, "CRR Art. 161.1"),
    Defect("D16", "consistency", "formula", "MED",
           "K_IRB no coincide con SQRT(PD_FINAL)*0.06 + PD_FINAL*0.5.",
           ("K_IRB", "PD_FINAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(K_IRB - (SQRT(PD_FINAL)*0.06+PD_FINAL*0.5)) > 0.0001",
           _d16_kirb, "CRR Art. 153"),
    # ── uniqueness / cardinality ────────────────────────────────────────────
    Defect("D17", "uniqueness", "cross_table", "MED",
           "Contrato fusionado (SW_FUSION=1) cuyo EAD en balance duplica el OR_EAD "
           "original — síntoma de filas BASILEA duplicadas que sobrevivieron.",
           ("SW_FUSION", "EAD_BALANCE", "OR_EAD"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SW_FUSION=1 AND EAD_BALANCE > OR_EAD*1.5",
           _d17_fusion_dup, "BASILEA dedup"),
    # ── completeness ─────────────────────────────────────────────────────────
    Defect("D18", "completeness", "completitud", "HIGH",
           "PD_ESTIMADA vacía (NULL) en un ciclo activo (no CERRADO): campo "
           "mandatorio para el cálculo de capital.",
           ("PD_ESTIMADA", "ESTADO_CICLO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_ESTIMADA IS NULL AND ESTADO_CICLO<>'CERRADO'",
           _d18_pd_nulo_activo, "BCBS 239 P4 / EBA GL 2017/16"),
    # ── validity (domain membership) ────────────────────────────────────────
    Defect("D19", "validity", "rango", "HIGH",
           f"SEGMENTO fuera del dominio permitido {ALLOWED_SEGMENTS}.",
           ("SEGMENTO",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SEGMENTO NOT IN ('CORP','SME','RETAIL_HIP','RETAIL_CONS')",
           _d19_segmento_invalido, "CRR Art. 147"),
    Defect("D20", "validity", "rango", "HIGH",
           "STAGE_IFRS9 fuera del dominio permitido {1,2,3}.",
           ("STAGE_IFRS9",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE STAGE_IFRS9 NOT IN (1,2,3)",
           _d20_stage_invalido, "IFRS 9"),
    # ── timeliness ───────────────────────────────────────────────────────────
    Defect("D21", "timeliness", "consistencia", "MED",
           f"MES_CICLO posterior al periodo de referencia ({REFERENCE_PERIOD}): "
           "dato futuro / no valido para el corte.",
           ("MES_CICLO",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE MES_CICLO > {REFERENCE_PERIOD}",
           _d21_mes_futuro, "BCBS 239 P5"),
    # ── accuracy (vs authoritative source value) ────────────────────────────
    Defect("D22", "accuracy", "consistencia", "MED",
           "El EAD en balance difiere del importe dispuesto BASILEA (OR_DISPTO) "
           "más allá de la tolerancia: inexactitud frente a la fuente autorizada.",
           ("EAD_BALANCE", "OR_DISPTO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(EAD_BALANCE - OR_DISPTO) > 1.0",
           _d22_ead_vs_basilea, "BCBS 239 P3"),
    # ── conformity / referential integrity ──────────────────────────────────
    Defect("D14", "conformity", "cross_table", "HIGH",
           "Segmento RETAIL_HIP (hipotecario) cuyo colateral no es HIPOTECA: "
           "inconsistencia referencial entre segmento y tipo de garantía.",
           ("SEGMENTO", "COLATERAL_TIPO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SEGMENTO='RETAIL_HIP' AND COLATERAL_TIPO<>'HIPOTECA'",
           _d14_retail_hip_sin_hipoteca, "CRR Art. 147"),
    Defect("D24", "conformity", "consistencia", "MED",
           "SW_FUSION=0 (sin fusión) pero ID_FUSION_FINAL informado: flags de "
           "fusión incoherentes (integridad referencial).",
           ("SW_FUSION", "ID_FUSION_FINAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE SW_FUSION=0 AND ID_FUSION_FINAL IS NOT NULL",
           _d24_fusion_flag_inconsistente),
    # ── plausibility / reasonableness ───────────────────────────────────────
    Defect("D10", "plausibility", "consistencia", "MED",
           "(RECUPERACION + COSTE_TOTAL) supera el 150% del EAD: recuperación "
           "implausible frente a la exposición (razonabilidad).",
           ("RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO", "EAD_TOTAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE (RECUPERACION_ACUMULADA+COSTE_TOTAL_ACUMULADO) > 1.5*EAD_TOTAL",
           _d10_recuperacion_sobre_ead, "EBA GL 2017/16 §135"),
    Defect("D25", "plausibility", "consistencia", "MED",
           "Peor grado de rating (>=14) con PD estimada implausible baja (<1%): "
           "violación de monotonicidad rating↔PD (razonabilidad).",
           ("RATING_GRADO", "PD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE RATING_GRADO>=14 AND PD_ESTIMADA<0.01",
           _d25_grade_pd_no_monotono, "EBA GL 2017/16 §73"),
    # ── D26+: downturn / MoC / governance / dates / FX / uniqueness ─────────
    Defect("D26", "consistency", "referencial", "HIGH",
           "La LGD downturn es inferior a la LGD estimada a largo plazo: "
           "LGD_DOWNTURN < LGD_ESTIMADA.",
           ("LGD_DOWNTURN", "LGD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE LGD_DOWNTURN < LGD_ESTIMADA - 0.0001",
           _d26_downturn_bajo, "CRR Art. 181.1(b) / EBA GL 2017/16 §345"),
    Defect("D27", "consistency", "formula", "MED",
           "El MoC total no coincide con la suma de sus categorías "
           "(A: deficiencias de datos, B: representatividad, C: error general): "
           "MOC != MOC_CAT_A + MOC_CAT_B + MOC_CAT_C.",
           ("MOC", "MOC_CAT_A", "MOC_CAT_B", "MOC_CAT_C"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE ABS(MOC - (MOC_CAT_A+MOC_CAT_B+MOC_CAT_C)) > 0.0001",
           _d27_moc_sum, "EBA GL 2017/16 §43-44"),
    Defect("D28", "conformity", "consistencia", "HIGH",
           "Ventana de observación histórica < 5 años sin marcar el flag de "
           "no conformidad (FLAG_NC=0).",
           ("VENTANA_OBSERVACION_YEARS", "FLAG_NC"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE VENTANA_OBSERVACION_YEARS < 5 AND FLAG_NC = 0",
           _d28_ventana_sin_flag, "EBA GL 2017/16 §6.3.2.1"),
    Defect("D29", "timeliness", "consistencia", "MED",
           "Exposición con colateral cuya última valoración tiene más de 36 "
           "meses de antigüedad respecto al mes del ciclo (revaluación "
           "periódica incumplida).",
           ("COLATERAL_TIPO", "MES_VALORACION_COLATERAL", "MES_CICLO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE COLATERAL_TIPO <> 'NINGUNA' AND "
           f"(MES_CICLO/100*12 + MES_CICLO%100) - "
           f"(MES_VALORACION_COLATERAL/100*12 + MES_VALORACION_COLATERAL%100) > 36",
           _d29_valoracion_estancada, "CRR Art. 208.3"),
    Defect("D30", "accuracy", "formula", "MED",
           "El EAD convertido a EUR no coincide con EAD_TOTAL * TIPO_CAMBIO "
           "más allá de la tolerancia.",
           ("EAD_TOTAL_EUR", "EAD_TOTAL", "TIPO_CAMBIO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE ABS(EAD_TOTAL_EUR - EAD_TOTAL*TIPO_CAMBIO) > 1.0",
           _d30_fx_mal_convertido, "BCBS 239 P3"),
    Defect("D31", "plausibility", "consistencia", "HIGH",
           "Ciclo cerrado cuyo mes de cierre es anterior al mes de default: "
           "orden temporal imposible.",
           ("MES_CIERRE_CICLO", "MES_DEFAULT"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE MES_CIERRE_CICLO IS NOT NULL AND MES_CIERRE_CICLO < MES_DEFAULT",
           _d31_cierre_antes_default, "BCBS 239 P3"),
    Defect("D32", "conformity", "consistencia", "MED",
           "Ciclo cerrado por CURA sin el flag de curación activo "
           "(TERMINACION='CURA' con CURE_FLAG=0).",
           ("TERMINACION", "CURE_FLAG"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE TERMINACION = 'CURA' AND CURE_FLAG = 0",
           _d32_cura_sin_flag, "EBA GL 2017/16 §101"),
    Defect("D33", "uniqueness", "cardinality", "HIGH",
           "Clave primaria duplicada: el mismo ID_CONTR_CICLO_LGD aparece en "
           "más de una fila (duplicidad real, no síntoma).",
           (PK_COLUMN,),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"GROUP BY {PK_COLUMN} HAVING COUNT(*) > 1",
           regulation_ref="BCBS 239 P3", plants_duplicate_pk=True),
    Defect("D34", "validity", "rango", "MED",
           "DIVISA fuera del dominio permitido ('EUR','USD','GBP').",
           ("DIVISA",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE DIVISA NOT IN ('EUR','USD','GBP')",
           _d34_divisa_invalida, "ISO 4217 / BCBS 239 P3"),
    Defect("D35", "completeness", "completitud", "HIGH",
           "Exposición con colateral informado (COLATERAL_TIPO <> 'NINGUNA') "
           "sin valor de colateral vigente (VALOR_COLATERAL NULL).",
           ("COLATERAL_TIPO", "VALOR_COLATERAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE COLATERAL_TIPO <> 'NINGUNA' AND VALOR_COLATERAL IS NULL",
           _d35_colateral_sin_valor, "CRR Art. 199 / BCBS 239 P4"),
    Defect("D36", "consistency", "formula", "HIGH",
           "La provisión contable no coincide con la ECL calculada: "
           "PROVISION != ECL.",
           ("PROVISION", "ECL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(PROVISION - ECL) > {FORMULA_EPS}",
           _d36_provision_ecl, "IFRS 9"),
    Defect("D37", "consistency", "referencial", "HIGH",
           "La PD downturn es inferior a la PD estimada: PD_DOWNTURN < PD_ESTIMADA.",
           ("PD_DOWNTURN", "PD_ESTIMADA"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE PD_DOWNTURN < PD_ESTIMADA - 0.000001",
           _d37_pd_downturn_bajo, "CRR Art. 181.1.b"),
    Defect("D38", "consistency", "formula", "MED",
           "La LGD con MoC no incorpora el margen de cautela: "
           "LGD_CON_MOC != LGD_ESTIMADA + MOC.",
           ("LGD_CON_MOC", "LGD_ESTIMADA", "MOC"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE ABS(LGD_CON_MOC - (LGD_ESTIMADA + MOC)) > 0.0001",
           _d38_lgd_con_moc, "EBA GL 2017/16 §50"),
    Defect("D39", "validity", "rango", "MED",
           "Vencimiento efectivo fuera del rango regulatorio [1, 5] años.",
           ("M_VENCIMIENTO",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE M_VENCIMIENTO NOT BETWEEN 1.0 AND 5.0",
           _d39_vencimiento_fuera, "CRR Art. 162"),
    Defect("D40", "consistency", "formula", "MED",
           "El alias EAD no coincide con EAD_TOTAL (deriva de vistas "
           "desincronizadas).",
           ("EAD", "EAD_TOTAL"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE ABS(EAD - EAD_TOTAL) > {FORMULA_EPS}",
           _d40_ead_alias, "CRR Art. 166"),
    Defect("D41", "plausibility", "consistencia", "MED",
           "Recuperaciones acumuladas positivas con tasa de descuento nula o "
           "negativa: los flujos no pueden descontarse.",
           ("RECUPERACION_ACUMULADA", "TASA_DESCUENTO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE RECUPERACION_ACUMULADA > 0 AND TASA_DESCUENTO <= 0",
           _d41_descuento_cero, "EBA GL 2019/03 §85"),
    Defect("D42", "conformity", "consistencia", "MED",
           "Causa de default '90_DIAS_VENCIDO' con menos de 90 días de impago.",
           ("CAUSA_DEFAULT", "DPDS"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE CAUSA_DEFAULT = '90_DIAS_VENCIDO' AND DPDS < 90",
           _d42_causa_sin_dpd, "CRR Art. 178"),
    Defect("D43", "conformity", "consistencia", "LOW",
           "Naturaleza jurídica incoherente con el segmento: CORP/SME deben "
           "ser persona jurídica ('J') y retail persona física ('F').",
           ("TIPO_PERSONA", "SEGMENTO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE (SEGMENTO IN ('CORP','SME') AND TIPO_PERSONA <> 'J') "
           f"OR (SEGMENTO IN ('RETAIL_HIP','RETAIL_CONS') AND TIPO_PERSONA <> 'F')",
           _d43_tipo_persona, "CRR Art. 147"),
    Defect("D44", "conformity", "referencial", "MED",
           "Bucket de calibración no derivado del segmento: "
           "CALIBRATION_SEGMENT debe comenzar por SEGMENTO.",
           ("CALIBRATION_SEGMENT", "SEGMENTO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE CALIBRATION_SEGMENT NOT LIKE SEGMENTO || '%'",
           _d44_calibration_segment, "EBA GL 2017/16 §73"),
    Defect("D45", "plausibility", "rango", "MED",
           "Haircut implausible (> 50%) sobre una exposición con colateral: "
           "valoración prudencial fuera de toda banda regulatoria.",
           ("HAIRCUT", "COLATERAL_TIPO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE COLATERAL_TIPO <> 'NINGUNA' AND (HAIRCUT < 0 OR HAIRCUT > 0.5)",
           _d45_haircut_implausible, "CRR Art. 161.4"),
    Defect("D46", "completeness", "completitud", "HIGH",
           "Ciclo activo (no CERRADO) sin identificador de cliente: campo "
           "mandatorio para agregación por deudor.",
           ("ID_CLIENTE", "ESTADO_CICLO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE (ID_CLIENTE IS NULL OR ID_CLIENTE = '') AND ESTADO_CICLO <> 'CERRADO'",
           _d46_cliente_nulo, "BCBS 239 P3"),
    Defect("D47", "accuracy", "formula", "MED",
           "La LGD realizada de un ciclo cerrado no respeta la fórmula de "
           "backtesting LGD = 1 - (recuperaciones - costes) / EAD.",
           ("LGD_REALIZADA", "RECUPERACION_ACUMULADA", "COSTE_TOTAL_ACUMULADO",
            "EAD_TOTAL", "ESTADO_CICLO"),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} "
           f"WHERE ESTADO_CICLO = 'CERRADO' AND EAD_TOTAL > 0 AND "
           f"ABS(LGD_REALIZADA - MAX(0.0, MIN(1.0, "
           f"1.0 - (RECUPERACION_ACUMULADA - COSTE_TOTAL_ACUMULADO)/EAD_TOTAL))) > 0.001",
           _d47_lgd_realizada_formula, "EBA GL 2017/16 §135"),
    # ── cross-table: referential integrity + reconciliation ─────────────────
    Defect("D48", "conformity", "cross_table", "HIGH",
           "Ciclo reportado cuyo ID_CONTRATO no existe en la tabla CONTRATOS: "
           "integridad referencial rota (huérfano).",
           ("ID_CONTRATO",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "LEFT JOIN contratos c ON cc.ID_CONTRATO = c.ID_CONTRATO "
           "WHERE c.ID_CONTRATO IS NULL",
           _d48_orphan_contrato, "BCBS 239 P3",
           plants_from_existing=True, cross_table=True),
    Defect("D49", "accuracy", "cross_table", "HIGH",
           "El OR_EAD del ciclo no cuadra con el OR_EAD autorizado en "
           "BASILEA_MENSUAL para el mismo contrato y mes (reconciliación).",
           ("OR_EAD",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "JOIN basilea_mensual b ON cc.ID_CONTRATO = b.ID_CONTRATO "
           "AND cc.MES_CICLO = b.MES_CICLO "
           "WHERE ABS(cc.OR_EAD - b.OR_EAD) > 1.0",
           _d49_or_ead_reconc, "CRR Art. 166 / BCBS 239 P3",
           plants_from_existing=True, pick_where="SW_FUSION = 0",
           cross_table=True),
    Defect("D50", "consistency", "cross_table", "HIGH",
           "El SEGMENTO reportado en el ciclo difiere del SEGMENTO maestro del "
           "contrato en CONTRATOS: el mismo atributo se informa distinto en "
           "dos tablas.",
           ("SEGMENTO",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "JOIN contratos c ON cc.ID_CONTRATO = c.ID_CONTRATO "
           "WHERE cc.SEGMENTO <> c.SEGMENTO",
           _d50_segmento_reconc, "CRR Art. 147 / BCBS 239 P3",
           plants_from_existing=True, cross_table=True),
    Defect("D51", "consistency", "cross_table", "MED",
           "El ID_CLIENTE del ciclo no coincide con el del contrato en "
           "CONTRATOS (deudor mal asignado).",
           ("ID_CLIENTE",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "JOIN contratos c ON cc.ID_CONTRATO = c.ID_CONTRATO "
           "WHERE cc.ID_CLIENTE <> c.ID_CLIENTE",
           _d51_cliente_reconc, "BCBS 239 P3",
           plants_from_existing=True, cross_table=True),
    Defect("D52", "accuracy", "cross_table", "MED",
           "El valor de colateral del ciclo no cuadra con el registrado en "
           "COLATERALES para el contrato (reconciliación de garantías).",
           ("VALOR_COLATERAL_INICIAL",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "JOIN colaterales co ON cc.ID_CONTRATO = co.ID_CONTRATO "
           "WHERE ABS(cc.VALOR_COLATERAL_INICIAL - co.VALOR_COLATERAL_INICIAL) > 1.0",
           _d52_colateral_valor_reconc, "CRR Art. 208 / BCBS 239 P3",
           plants_from_existing=True, pick_where="COLATERAL_TIPO <> 'NINGUNA'",
           cross_table=True),
    Defect("D53", "consistency", "cross_table", "MED",
           "La fecha de alta del contrato en el ciclo difiere de la registrada "
           "en CONTRATOS.",
           ("FECHA_ALTA_CONTRATO",),
           "SELECT cc.ID_CONTR_CICLO_LGD FROM ciclos_calibrados cc "
           "JOIN contratos c ON cc.ID_CONTRATO = c.ID_CONTRATO "
           "WHERE cc.FECHA_ALTA_CONTRATO <> c.FECHA_ALTA_CONTRATO",
           _d53_fecha_alta_reconc, "BCBS 239 P3",
           plants_from_existing=True, cross_table=True),
    # ── date interrelations (start / default / close / adjudication / sale) ──
    Defect("D54", "plausibility", "consistencia", "HIGH",
           "La fecha de alta del contrato es posterior a la fecha de default: "
           "orden temporal imposible.",
           ("FECHA_ALTA_CONTRATO", "FECHA_DEFAULT"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE FECHA_ALTA_CONTRATO > FECHA_DEFAULT",
           _d54_alta_tras_default, "CRR Art. 178"),
    Defect("D55", "plausibility", "consistencia", "MED",
           "La fecha de adjudicación cae fuera de la ventana del ciclo "
           "(anterior al default o posterior al cierre).",
           ("FECHA_ADJUDICACION", "FECHA_DEFAULT", "FECHA_CIERRE_CICLO"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE ADJUDICACION_FLAG = '1' AND FECHA_ADJUDICACION IS NOT NULL AND "
           "(FECHA_ADJUDICACION < FECHA_DEFAULT OR (FECHA_CIERRE_CICLO IS NOT NULL "
           "AND FECHA_ADJUDICACION > FECHA_CIERRE_CICLO))",
           _d55_adj_fuera_ventana, "EBA GL 2017/16 §159"),
    Defect("D56", "plausibility", "consistencia", "MED",
           "La venta del colateral es anterior a la adjudicación: no se puede "
           "vender lo aún no adjudicado.",
           ("FECHA_VENTA_COLATERAL", "FECHA_ADJUDICACION"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE FECHA_VENTA_COLATERAL IS NOT NULL AND "
           "FECHA_VENTA_COLATERAL < FECHA_ADJUDICACION",
           _d56_venta_antes_adj, "EBA GL 2017/16 §159"),
    Defect("D57", "consistency", "consistencia", "MED",
           "El mes derivado de FECHA_DEFAULT no coincide con MES_DEFAULT: dos "
           "representaciones de la misma fecha en desacuerdo.",
           ("FECHA_DEFAULT", "MES_DEFAULT"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados WHERE "
           "CAST(substr(FECHA_DEFAULT,1,4)||substr(FECHA_DEFAULT,6,2) AS INTEGER) "
           "<> MES_DEFAULT",
           _d57_fecha_periodo, "BCBS 239 P3"),
    # ── panel: monthly evolution (temporal invariants) ──────────────────────
    Defect("D58", "plausibility", "consistencia", "HIGH",
           "La recuperación acumulada de un ciclo DISMINUYE de un mes al "
           "siguiente: un acumulado no puede decrecer.",
           ("RECUPERACION_ACUMULADA", "SEQ"),
           "SELECT DISTINCT e.ID_CONTR_CICLO_LGD FROM evolucion_mensual e "
           "JOIN evolucion_mensual p ON e.ID_CONTR_CICLO_LGD = p.ID_CONTR_CICLO_LGD "
           "AND p.SEQ = e.SEQ - 1 "
           "WHERE e.RECUPERACION_ACUMULADA < p.RECUPERACION_ACUMULADA",
           (lambda r: r), "EBA GL 2017/16 §135",  # row mutate unused for panel
           target_table="evolucion_mensual", panel_mutate=_pd58_recup_baja),
    Defect("D59", "completeness", "completitud", "MED",
           "La serie mensual del ciclo tiene un hueco: dos meses consecutivos "
           "no son contiguos.",
           ("MES_CICLO", "SEQ"),
           "SELECT DISTINCT e.ID_CONTR_CICLO_LGD FROM evolucion_mensual e "
           "JOIN evolucion_mensual p ON e.ID_CONTR_CICLO_LGD = p.ID_CONTR_CICLO_LGD "
           "AND p.SEQ = e.SEQ - 1 WHERE "
           "(e.MES_CICLO/100*12 + e.MES_CICLO%100) - "
           "(p.MES_CICLO/100*12 + p.MES_CICLO%100) <> 1",
           (lambda r: r), "BCBS 239 P4",
           target_table="evolucion_mensual", panel_mutate=_pd59_hueco_mes),
    Defect("D60", "uniqueness", "cardinality", "HIGH",
           "Un mismo mes aparece dos veces para el mismo ciclo en la serie "
           "mensual (duplicidad de observación).",
           ("ID_CONTR_CICLO_LGD", "MES_CICLO"),
           "SELECT ID_CONTR_CICLO_LGD FROM evolucion_mensual "
           "GROUP BY ID_CONTR_CICLO_LGD, MES_CICLO HAVING COUNT(*) > 1",
           (lambda r: r), "BCBS 239 P3",
           target_table="evolucion_mensual", panel_mutate=_pd60_mes_duplicado),
    Defect("D61", "consistency", "consistencia", "HIGH",
           "Observación mensual en STAGE_IFRS9=3 (default) con PD estimada "
           "inferior a 1.0: en default la PD debe ser 1.",
           ("STAGE_IFRS9", "PD_ESTIMADA"),
           "SELECT DISTINCT ID_CONTR_CICLO_LGD FROM evolucion_mensual "
           "WHERE STAGE_IFRS9 = 3 AND PD_ESTIMADA < 1.0",
           (lambda r: r), "CRR Art. 178",
           target_table="evolucion_mensual", panel_mutate=_pd61_stage3_pd_baja),
    Defect("D62", "consistency", "consistencia", "HIGH",
           "El stage IFRS-9 mejora de un mes al siguiente sin que exista una "
           "cura (CURE_FLAG=0): regresión de stage injustificada.",
           ("STAGE_IFRS9", "CURE_FLAG", "SEQ"),
           "SELECT DISTINCT e.ID_CONTR_CICLO_LGD FROM evolucion_mensual e "
           "JOIN evolucion_mensual p ON e.ID_CONTR_CICLO_LGD = p.ID_CONTR_CICLO_LGD "
           "AND p.SEQ = e.SEQ - 1 "
           "WHERE e.STAGE_IFRS9 < p.STAGE_IFRS9 AND e.CURE_FLAG = 0",
           (lambda r: r), "IFRS 9 B5.5.12",
           target_table="evolucion_mensual", panel_mutate=_pd62_regresion_stage),
    Defect("D63", "consistency", "consistencia", "MED",
           "Los días de impago (DPDS) disminuyen de un mes al siguiente sin "
           "cura: sólo una cura puede reducir la mora acumulada.",
           ("DPDS", "CURE_FLAG", "SEQ"),
           "SELECT DISTINCT e.ID_CONTR_CICLO_LGD FROM evolucion_mensual e "
           "JOIN evolucion_mensual p ON e.ID_CONTR_CICLO_LGD = p.ID_CONTR_CICLO_LGD "
           "AND p.SEQ = e.SEQ - 1 "
           "WHERE e.DPDS < p.DPDS AND e.CURE_FLAG = 0",
           (lambda r: r), "CRR Art. 178",
           target_table="evolucion_mensual", panel_mutate=_pd63_dpd_baja),
    # ── "weird data": cross-domain business-rule violations ──────────────────
    Defect("D64", "conformity", "consistencia", "HIGH",
           "Tarjeta de crédito (producto revolving no garantizado) con "
           "colateral asignado: las tarjetas no admiten garantía real.",
           ("PRODUCTO", "COLATERAL_TIPO"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE PRODUCTO = 'TARJETA' AND COLATERAL_TIPO <> 'NINGUNA'",
           _d64_colateral_en_tarjeta, "CRR Art. 194"),
    Defect("D65", "conformity", "consistencia", "HIGH",
           "Préstamo hipotecario (HIPOTECA_LN) sin colateral HIPOTECA: un "
           "producto hipotecario debe estar garantizado por la hipoteca.",
           ("PRODUCTO", "COLATERAL_TIPO"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE PRODUCTO = 'HIPOTECA_LN' AND COLATERAL_TIPO <> 'HIPOTECA'",
           _d65_hipoteca_sin_garantia, "CRR Art. 194"),
    Defect("D66", "conformity", "consistencia", "MED",
           "Adjudicación/ejecución sobre una exposición sin colateral: no se "
           "puede adjudicar una garantía inexistente.",
           ("ADJUDICACION_FLAG", "COLATERAL_TIPO"),
           "SELECT ID_CONTR_CICLO_LGD FROM ciclos_calibrados "
           "WHERE ADJUDICACION_FLAG = '1' AND COLATERAL_TIPO = 'NINGUNA'",
           _d66_adjudicacion_sin_colateral, "CRR Art. 199"),
    # ── decoys: single-column range, must score r_coherence = 0 ──────────────
    Defect("DA", "validity", "rango", "MED",
           "EAD_TOTAL <= 0 en un ciclo activo (control de rango, no coherencia).",
           ("EAD_TOTAL",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE EAD_TOTAL<=0",
           _da_ead_cero, decoy=True),
    Defect("DB", "validity", "rango", "LOW",
           "LGD_REALIZADA fuera de rango [0,1] (control de rango, no coherencia).",
           ("LGD_REALIZADA",),
           f"SELECT {PK_COLUMN} FROM {TABLE_NAME} WHERE LGD_REALIZADA NOT BETWEEN 0 AND 1",
           _db_lgd_realizada_neg, decoy=True),
]

DEFECTS_BY_ID: dict[str, Defect] = {d.defect_id: d for d in DEFECTS}
COHERENCE_DEFECTS = [d for d in DEFECTS if not d.decoy]
DECOY_DEFECTS = [d for d in DEFECTS if d.decoy]


def defects_by_dimension() -> dict[str, list[Defect]]:
    out: dict[str, list[Defect]] = {d: [] for d in DIMENSIONS}
    for d in DEFECTS:
        out.setdefault(d.dimension, []).append(d)
    return out


if __name__ == "__main__":
    by_dim = defects_by_dimension()
    print(f"{len(DEFECTS)} defects across {sum(1 for v in by_dim.values() if v)} dimensions")
    for dim in DIMENSIONS:
        ds = by_dim.get(dim, [])
        if ds:
            ids = ", ".join(d.defect_id for d in ds)
            print(f"  {dim:14s} ({len(ds)}): {ids}")
