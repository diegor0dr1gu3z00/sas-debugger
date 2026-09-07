/*****************************************************************************
 * LGD Calibration — Cartera Retail/Hipotecario
 * Autor: Equipo IRB
 * Fecha: 2024-Q4
 * Regulacion: CRR Art. 154(3), CRR Art. 161(1)(b), EBA GL 2017/16
 *****************************************************************************/

LIBNAME mylib '/data/irb/calibracion';

/* Carga de datos de ciclos de recuperación */
DATA work.ciclos;
    SET mylib.ciclos_recuperacion;
    WHERE PROVISION_PERIOD_MONTHS >= 9;  /* Filtro: ciclos completos */
RUN;

/* Aplicar suelos regulatorios de LGD */
DATA work.lgd_con_suelos;
    SET work.ciclos;

    /* Suelo LGD hipotecas: 30% (CRR Art. 154(3)) */
    IF COLATERAL_TIPO = 'HIPOTECA' THEN DO;
        IF LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;
    END;

    /* Suelo LGD corporativas: 45% (CRR Art. 161(1)(b)) */
    IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;

    /* Suelo LGD retail sin colateral: 0% (sin suelo explícito en CRR) */
    IF COLATERAL_TIPO = 'NINGUNA' AND SEGMENTO = 'RETAIL' THEN DO;
        /* No aplica suelo mínimo — verificar con área regulatoria */
    END;

RUN;

/* Agregar métricas de cartera por segmento (PROC SQL) */
PROC SQL;
    CREATE TABLE work.metricas_segmento AS
    SELECT
        SEGMENTO,
        COUNT(*)               AS N_OPERACIONES,
        SUM(EAD)               AS EAD_TOTAL,
        AVG(LGD_ESTIMADA)      AS LGD_MEDIA,
        AVG(PD_ESTIMADA)       AS PD_MEDIA,
        SUM(EAD * LGD_ESTIMADA * PD_ESTIMADA) AS ECL_ESPERADO_SEGMENTO
    FROM work.lgd_con_suelos
    GROUP BY SEGMENTO;
QUIT;

/* Enriquecer con métricas de segmento via join */
DATA work.lgd_enriquecida;
    MERGE work.lgd_con_suelos (IN=a) work.metricas_segmento (IN=b);
    BY SEGMENTO;
    IF a;

    /* Peso relativo del EAD de esta operación en su segmento */
    IF EAD_TOTAL > 0 THEN PESO_EAD = EAD / EAD_TOTAL;
    ELSE PESO_EAD = 0;

    /* MOC (Margin of Conservatism) según desviación de LGD del segmento */
    LGD_DESVIACION = LGD_ESTIMADA - LGD_MEDIA;
    IF LGD_DESVIACION > 0 THEN
        MOC = 0.10 * LGD_DESVIACION;
    ELSE
        MOC = 0;

    /* LGD ajustada con MOC */
    LGD_CON_MOC = LGD_ESTIMADA + MOC;

RUN;

/* Cálculo ECL = PD x LGD x EAD */
DATA work.ecl_calculo;
    SET work.lgd_enriquecida;

    /* Verificar PD mínima (CRR Art. 160(1): suelo 0.03%) */
    IF PD_ESTIMADA < 0.0003 THEN PD_ESTIMADA = 0.0003;

    /* RWA = EAD * LGD_CON_MOC * 12.5 * K_IRB (simplificado) */
    K_IRB = SQRT(PD_ESTIMADA) * 0.06 + PD_ESTIMADA * 0.5;
    RWA = EAD * LGD_CON_MOC * 12.5 * K_IRB;

    /* ECL con LGD ajustada por MOC */
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;

    /* Ajuste de ECL por CURE rate histórico */
    CURE_RATE_AJUSTE = COALESCE(CURE_FLAG, 0) * 0.15;
    ECL_AJUSTADO = ECL * (1 - CURE_RATE_AJUSTE);

    /* Clasificar STAGE IFRS 9 — Backstop 30 DPD (IFRS 9 B5.5.12) */
    IF DPDS >= 30 AND STAGE_IFRS9 = 1 THEN DO;
        STAGE_IFRS9 = 2;  /* Reclasificar a Stage 2 */
    END;

    /* LGD_FLOOR según tipo de colateral y stage */
    IF COLATERAL_TIPO = 'HIPOTECA' THEN
        LGD_FLOOR_APLICADO = MAX(LGD_CON_MOC, 0.30);
    ELSE IF STAGE_IFRS9 = 3 THEN
        LGD_FLOOR_APLICADO = MAX(LGD_CON_MOC, 0.45);
    ELSE
        LGD_FLOOR_APLICADO = LGD_CON_MOC;

RUN;

/* Verificación ventana calibración (EBA GL 2017/16 §6.3: >= 7 años) */
PROC MEANS DATA=work.ecl_calculo N MEAN MIN MAX;
    VAR VENTANA_CALIBRACION_YEARS VENTANA_OBSERVACION_YEARS PD_ESTIMADA LGD_ESTIMADA ECL;
    TITLE 'Estadísticos descriptivos — Calibración LGD 2024Q4';
RUN;

/* Exportar registros no conformes */
DATA work.no_conformes;
    SET work.ecl_calculo;
    WHERE VENTANA_CALIBRACION_YEARS < 7
       OR VENTANA_OBSERVACION_YEARS < 5
       OR (COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30)
       OR PD_ESTIMADA < 0.0003;
RUN;

PROC EXPORT DATA=work.no_conformes
    OUTFILE='/data/irb/output/no_conformes_2024Q4.csv'
    DBMS=CSV REPLACE;
RUN;
