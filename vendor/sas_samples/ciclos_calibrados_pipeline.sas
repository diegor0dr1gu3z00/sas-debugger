/*****************************************************************************
 * CICLOS_CALIBRADOS — Full PD & LGD estimation pipeline from BASILEA
 * ----------------------------------------------------------------------------
 * Stress-test database for the DQC (Data Quality Check) agent.
 *
 * 7 transformation layers, each building on the previous, so every field in
 * the final table CICLOS_CALIBRADOS has a lineage of >= 7 hops from the raw
 * BASILEA_MENSUAL / CONTRATOS / CICLOS / COLATERALES sources.
 *
 * The pipeline applies CORRECTIVE FIXES at every layer (regulatory floors,
 * COALESCE imputation, fusion de-duplication, type coercion, caps). A clearly
 * delimited DEFECT-INJECTION block at the end plants the ground-truth
 * incoherences that the DQC agent's checks must catch (see defect_catalog.py).
 *
 * Regulation:
 *   CRR Art. 147   — exposure class segmentation
 *   CRR Art. 160.1 — PD floor 0.03% (non-default exposures)
 *   CRR Art. 161.1 — LGD floors by collateral (mortgage 30%, corp 45%)
 *   CRR Art. 162   — effective maturity M in [1,5]
 *   CRR Art. 166   — EAD = on-balance + CCF * off-balance
 *   CRR Art. 178   — default definition (>= 90 DPD)
 *   CRR Art. 181   — LGD downturn
 *   EBA GL 2017/16 — PD/LGD estimation guidelines, MoC, calibration window
 *   IFRS 9 B5.5    — staging / 30-DPD backstop
 *****************************************************************************/

LIBNAME mylib '/data/irb/calibracion';

/* ===========================================================================
 * LAYER 0 — RAW SOURCE TABLES  (self-contained seed; production reads these
 *           from mylib. Each row is one recovery cycle / contract / month)
 * ===========================================================================*/

/* CICLOS — one row per recovery cycle (the unit of LGD calibration) */
DATA work.ciclos_raw;
    LENGTH ID_CONTR_CICLO_LGD ID_CONTRATO SEGMENTO CALIBRATION_SEGMENT $32;
    LENGTH PRODUCTO TERMINACION CAUSA_DEFAULT COLATERAL_TIPO $24;
    LENGTH MES_CICLO RATING_GRADO DPDS STAGE_IFRS9 CURE_FLAG 8;
    LENGTH PD_ESTIMADA LGD_ESTIMADA LGD_REALIZADA SALDO_PENDIENTE 8;
    LENGTH TIPO_INTERES_ORIGINAL TIPO_INTERES_ACTUAL INTERESES_ACUMULADOS 8;
    LENGTH ADJUDICACION_FLAG $1 ADJUDICACION_TIPO $12 ADJUDICACION_VALOR 8;
    LENGTH RECUPERACION_ACUMULADA COSTE_TOTAL_ACUMULADA TASA_DESCUENTO 8;
    LENGTH ESTADO_CICLO $12;
    INPUT ID_CONTR_CICLO_LGD $ ID_CONTRATO $ SEGMENTO $ CALIBRATION_SEGMENT $
          PRODUCTO $ TERMINACION $ CAUSA_DEFAULT $ COLATERAL_TIPO $
          MES_CICLO RATING_GRADO DPDS STAGE_IFRS9 CURE_FLAG
          PD_ESTIMADA LGD_ESTIMADA LGD_REALIZADA SALDO_PENDIENTE
          TIPO_INTERES_ORIGINAL TIPO_INTERES_ACTUAL INTERESES_ACUMULADOS
          ADJUDICACION_FLAG $ ADJUDICACION_TIPO $ ADJUDICACION_VALOR
          RECUPERACION_ACUMULADA COSTE_TOTAL_ACUMULADA TASA_DESCUENTO
          ESTADO_CICLO $;
    DATALINES;
CIC_0001 CONT_0001 CORP CORP_MED PRESTAMO FALLIDO 90_DIAS_VENCIDO NINGUNA 6 9 0 1 0 0.012 0.42 0.40 850000 0.060 0.062 4100 0 "" 0 210000 52000 0.0450 ESTIMACION
CIC_0002 CONT_0002 SME SME_MICRO PRESTAMO CURA 90_DIAS_VENCIDO NINGUNA 12 12 0 1 0 0.038 0.55 0.51 320000 0.071 0.071 0 0 "" 0 168000 39000 0.0502 CERRADO
CIC_0005 CONT_0005 RETAIL_HIP RETAIL_HIP HIPOTECA PRESTAMO CURA 90_DIAS_VENCIDO HIPOTECA 18 4 0 1 0 0.007 0.18 0.16 245000 0.029 0.031 0 0 "" 0 98000 9000 0.0310 CERRADO
CIC_0006 CONT_0006 CORP CORP_HIGH LINEA CERRADO 90_DIAS_VENCIDO NINGUNA 9 7 22 2 0 0.004 0.50 . 540000 0.040 0.040 2200 0 "" 0 290000 61000 0.0421 ESTIMACION
CIC_0007 CONT_0007 RETAIL_CONS RETAIL_CONS TARJETA FALLIDO IMPROBABLE_PAGO NINGUNA 15 14 35 3 0 0.092 0.85 0.82 18000 0.210 0.210 0 1 SUBASTA 12500 15500 4100 0.0710 ESTIMACION
;
RUN;

/* CONTRATOS — contract metadata + fusion flags */
DATA work.contratos_raw;
    LENGTH ID_CONTRATO ID_FUSION_FINAL ENTIDAD_ORIGEN $32;
    LENGTH SW_FUSION 8;
    INPUT ID_CONTRATO $ SW_FUSION ID_FUSION_FINAL $ ENTIDAD_ORIGEN $;
    DATALINES;
CONT_0001 0 . BANCO_A
CONT_0002 0 . BANCO_A
CONT_0005 0 . BANCO_B
CONT_0006 1 FUS_7712 BANCO_C
CONT_0007 0 . BANCO_A
;
RUN;

/* BASILEA_MENSUAL — monthly Basel exposure data (source of OR_EAD).
 * WARNING: ID_FUSION_FINAL is NOT unique here (see data/sas/toy_lgd meta). */
DATA work.basilea_mensual_raw;
    LENGTH ID_CONTRATO ID_FUSION_FINAL $32;
    LENGTH ID_FCH_DATOS OR_EAD OR_DISPTO OR_DISBLE 8;
    INPUT ID_CONTRATO $ ID_FUSION_FINAL $ ID_FCH_DATOS OR_EAD OR_DISPTO OR_DISBLE;
    DATALINES;
CONT_0001 . 201903 850000 845000 5000
CONT_0002 . 201910 320000 318000 2000
CONT_0005 . 201910 245000 240000 5000
CONT_0006 FUS_7712 201910 540000 538000 2000
CONT_A FUS_7712 201910 540000 538000 2000
CONT_0007 . 201910 18000 17900 100
;
RUN;

/* COLATERALES — collateral valuation (drives LGD_SUELO & LTV) */
DATA work.colaterales_raw;
    LENGTH ID_CONTRATO $32;
    LENGTH VALOR_COLATERAL_INICIAL VALOR_COLATERAL HAIRCUT 8;
    INPUT ID_CONTRATO $ VALOR_COLATERAL_INICIAL VALOR_COLATERAL HAIRCUT;
    DATALINES;
CONT_0001 0 0 0
CONT_0002 0 0 0
CONT_0005 280000 268000 0.04
CONT_0006 0 0 0
CONT_0007 0 0 0
;
RUN;

/* ===========================================================================
 * LAYER 1 — STAGING: type coercion + completeness fix
 *   FIX: coerce numerics, keep NULLs as missing (.), trim whitespace.
 * ===========================================================================*/
DATA work.stg_ciclos;
    SET work.ciclos_raw;
    PD_ESTIMADA        = COALESCE(PD_ESTIMADA, .);
    LGD_ESTIMADA       = COALESCE(LGD_ESTIMADA, .);
    ADJUDICACION_VALOR = COALESCE(ADJUDICACION_VALOR, 0);
    RECUPERACION_ACUMULADA = COALESCE(RECUPERACION_ACUMULADA, 0);
    COSTE_TOTAL_ACUMULADA  = COALESCE(COSTE_TOTAL_ACUMULADA, 0);
    IF ADJUDICACION_FLAG NOT IN ('0','1') THEN ADJUDICACION_FLAG = '0';
RUN;

/* ===========================================================================
 * LAYER 2 — FUSION + CONTRACT ENRICHMENT
 *   Join CICLOS to CONTRATOS to obtain SW_FUSION, ID_FUSION_FINAL,
 *   M_VENCIMIENTO_CTE. FIX: left-keep all cycles (IF a).
 * ===========================================================================*/
DATA work.ciclos_enr;
    MERGE work.stg_ciclos (IN=a) work.contratos_raw (IN=b);
    BY ID_CONTRATO;
    IF a;
    SW_FUSION = COALESCE(SW_FUSION, 0);
RUN;

/* ===========================================================================
 * LAYER 3 — BASILEA OR_EAD JOIN (the cross-table step)
 *   FIX the classic fusion-duplication bug: when SW_FUSION=1 a naive JOIN on
 *   ID_FUSION_FINAL yields N rows. We de-duplicate BASILEA to ONE row per
 *   fusion group + period via aggregation (MAX), then join.
 * ===========================================================================*/
PROC SQL;
    CREATE TABLE work.ciclos_ead AS
    SELECT  c.*,
            b.OR_EAD, b.OR_DISPTO, b.OR_DISBLE,
            col.VALOR_COLATERAL_INICIAL, col.VALOR_COLATERAL, col.HAIRCUT
    FROM work.ciclos_enr AS c
    LEFT JOIN (
        SELECT ID_FUSION_FINAL AS JOIN_KEY, ID_FCH_DATOS,
               MAX(OR_EAD) AS OR_EAD, MAX(OR_DISPTO) AS OR_DISPTO, MAX(OR_DISBLE) AS OR_DISBLE
        FROM work.basilea_mensual_raw
        WHERE ID_FUSION_FINAL IS NOT NULL
        GROUP BY ID_FUSION_FINAL, ID_FCH_DATOS
    ) b   ON c.ID_FUSION_FINAL = b.JOIN_KEY AND c.MES_CICLO = b.ID_FCH_DATOS
    LEFT JOIN work.basilea_mensual_raw b2
           ON c.ID_CONTRATO = b2.ID_CONTRATO AND c.MES_CICLO = b2.ID_FCH_DATOS
    LEFT JOIN work.colaterales_raw col ON c.ID_CONTRATO = col.ID_CONTRATO;
QUIT;

/* ===========================================================================
 * LAYER 4 — PD CALIBRATION + FLOORS
 *   FIX: PD_SUELO by segment (0.03% corp/retail-other, 0.05% retail mortgage
 *   per CRR Art. 160.1). PD_FINAL = max(estimada, suelo) is the CORRECTIVE
 *   FLOOR applied to the raw model PD.
 * ===========================================================================*/
DATA work.pd_cal;
    SET work.ciclos_ead;
    LENGTH PD_SUELO PD_FINAL PD_DOWNTURN 8;
    IF SEGMENTO = 'RETAIL_HIP' THEN PD_SUELO = 0.0005;
    ELSE                             PD_SUELO = 0.0003;
    PD_FINAL    = MAX(PD_ESTIMADA, PD_SUELO);
    PD_DOWNTURN = MIN(1, PD_ESTIMADA * 1.5);
RUN;

/* ===========================================================================
 * LAYER 5 — LGD DOWNTURN / FLOORS / MARGIN OF CONSERVATISM
 *   FIX: LGD_SUELO by collateral (HIPOTECA 0.30, CORP 0.45, else 0).
 *   MoC = 5% of LGD_ESTIMADA (EBA GL 2017/16 §50). Missing LGD imputed to
 *   segment central tendency. LGD_FINAL = max(LGD_CON_MOC, LGD_SUELO).
 * ===========================================================================*/
DATA work.lgd_cal;
    SET work.pd_cal;
    LENGTH LGD_SUELO MOC LGD_CON_MOC LGD_FINAL 8;
    IF COLATERAL_TIPO = 'HIPOTECA'            THEN LGD_SUELO = 0.30;
    ELSE IF SEGMENTO = 'CORP'                 THEN LGD_SUELO = 0.45;
    ELSE                                            LGD_SUELO = 0.00;
    IF LGD_ESTIMADA = . THEN LGD_ESTIMADA = 0.45;
    MOC         = 0.05 * LGD_ESTIMADA;
    LGD_CON_MOC = LGD_ESTIMADA + MOC;
    LGD_FINAL   = MAX(LGD_CON_MOC, LGD_SUELO);
RUN;

/* ===========================================================================
 * LAYER 6 — EAD / CCF
 *   FIX: EAD_BALANCE = drawn (OR_DISPTO); EAD_FUERA_BALANCE = CCF * undrawn
 *   (CRR Art. 166). CCF floored at 0. EAD_TOTAL = balance + off-balance.
 * ===========================================================================*/
DATA work.ead_cal;
    SET work.lgd_cal;
    LENGTH CCF_ESTIMADO EAD_BALANCE EAD_FUERA_BALANCE EAD_TOTAL 8;
    CCF_ESTIMADO       = 0.75;
    EAD_BALANCE        = COALESCE(OR_DISPTO, SALDO_PENDIENTE);
    EAD_FUERA_BALANCE  = CCF_ESTIMADO * COALESCE(OR_DISBLE, 0);
    EAD_TOTAL          = EAD_BALANCE + EAD_FUERA_BALANCE;
    EAD = EAD_TOTAL;
RUN;

/* ===========================================================================
 * LAYER 7 — ECL + RWA + IFRS-9 STAGING  (FINAL OUTPUT TABLE)
 *   FIX: ECL = PD_FINAL * LGD_FINAL * EAD_TOTAL (canonical formula).
 *   RWA = EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB (IRB-adv, simplified).
 *   IFRS-9 backstop: DPDS>=30 & STAGE=1 -> STAGE 2 (IFRS 9 B5.5.12).
 * ===========================================================================*/
DATA mylib.ciclos_calibrados;
    SET work.ead_cal;
    LENGTH K_IRB M_VENCIMIENTO RWA ECL PROVISION 8;
    K_IRB         = SQRT(PD_FINAL) * 0.06 + PD_FINAL * 0.5;
    M_VENCIMIENTO = 2.5;
    IF SEGMENTO = 'CORP' THEN M_VENCIMIENTO = MIN(5, MAX(1, 3.0));
    RWA           = EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB;
    ECL           = PD_FINAL * LGD_FINAL * EAD_TOTAL;
    PROVISION     = ECL;
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN STAGE_IFRS9 = 2;
RUN;

/* ===========================================================================
 * DEFECT-INJECTION BLOCK
 * ----------------------------------------------------------------------------
 * The pipeline above is CLEAN (all coherence invariants hold). The block
 * below is NOT executed in production — it documents the GROUND-TRUTH defects
 * planted by generate_db.py / defect_catalog.py that the DQC agent must
 * detect. Each defect is keyed (D01..D20) and cross-referenced from the
 * defect catalog. They are deliberately NOT fixed.
 * ---------------------------------------------------------------------------
 * D01  ECL <> PD_FINAL * LGD_FINAL * EAD_TOTAL        (formula)
 * D02  RWA <> EAD_TOTAL * LGD_FINAL * 12.5 * K_IRB    (formula)
 * D03  PD_FINAL < PD_SUELO                            (floor violated)
 * D04  LGD_FINAL < LGD_SUELO                          (floor violated)
 * D05  PD_FINAL < PD_ESTIMADA                         (max not applied)
 * D06  STAGE_IFRS9=3 AND DPDS < 30                    (default coherence)
 * D07  ADJUDICACION_VALOR > 0 AND ADJUDICACION_FLAG='0'
 * D08  ADJUDICACION_FLAG='1' AND ADJUDICACION_VALOR=0
 * D09  ADJUDICACION_VALOR > 0 AND ADJUDICACION_TIPO=''
 * D10  RECUPERACION + COSTE_TOTAL > 1.5*EAD_TOTAL
 * D11  ESTADO_CICLO='CERRADO' AND TERMINACION=''
 * D12  EAD_TOTAL <> EAD_BALANCE + EAD_FUERA_BALANCE
 * D13  EAD_FUERA_BALANCE <> CCF_ESTIMADO*OR_DISBLE (OR_DISBLE>0)
 * D14  SEGMENTO='RETAIL_HIP' AND COLATERAL_TIPO<>'HIPOTECA'  (cross-table)
 * D15  COLATERAL_TIPO='HIPOTECA' AND LGD_SUELO < 0.30        (referential)
 * D16  K_IRB <> SQRT(PD_FINAL)*0.06 + PD_FINAL*0.5           (formula)
 * D17  SW_FUSION=1 fusion group has >1 OR_EAD row            (cardinality)
 * DA   EAD_TOTAL <= 0                         (single-col decoy, r_coh=0)
 * DB   LGD_REALIZADA NOT BETWEEN 0 AND 1     (single-col decoy, r_coh=0)
 *****************************************************************************/
