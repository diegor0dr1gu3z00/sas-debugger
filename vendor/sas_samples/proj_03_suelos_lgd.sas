/*****************************************************************************
 * proj_03_suelos_lgd.sas — Suelos regulatorios LGD/PD + MoC + ECL + RWA
 * Regulacion: CRR Art. 154(3), Art. 160(1), Art. 161(1)(b); EBA GL 2017/16
 *****************************************************************************/

/* Aplicación de suelos y cálculo de métricas de riesgo */
DATA work.suelos_lgd;
    SET work.ciclos_ead;

    /* Suelo LGD hipotecas: 30% (CRR Art. 154(3)) */
    IF COLATERAL_TIPO = 'HIPOTECA' AND LGD_ESTIMADA < 0.30 THEN LGD_ESTIMADA = 0.30;

    /* Suelo LGD corporativas: 45% (CRR Art. 161(1)(b)) */
    IF SEGMENTO = 'CORP' AND LGD_ESTIMADA < 0.45 THEN LGD_ESTIMADA = 0.45;

    /* Suelo PD (CRR Art. 160(1)) — 0.05% uniforme para todos los segmentos */
    IF PD_ESTIMADA < 0.0005 THEN PD_ESTIMADA = 0.0005;

    /* MoC (Margin of Conservatism) fijo del 5% sobre la LGD estimada.
       Si LGD_ESTIMADA es missing (contratos SW_FUSION=1), MoC queda missing. */
    MoC = 0.05 * LGD_ESTIMADA;

    /* LGD ajustada con MoC */
    LGD_CON_MOC = LGD_ESTIMADA + MoC;

    /* ECL = PD x LGD x EAD */
    ECL = PD_ESTIMADA * LGD_CON_MOC * EAD;

    /* RWA = EAD * LGD_CON_MOC * 12.5 * K_IRB (formulación simplificada) */
    K_IRB = SQRT(PD_ESTIMADA) * 0.06 + PD_ESTIMADA * 0.5;
    RWA = EAD * LGD_CON_MOC * 12.5 * K_IRB;

    /* LGD realizada ex-post para ciclos con recuperación cerrada */
    LGD_REALIZADA = 1 - RECUPERACION_TOTAL / EAD;
RUN;

/* Revisión no_conformes: ventana de observación mínima (EBA GL 2017/16 §6.3) */
DATA work.no_conformes;
    SET work.suelos_lgd;

    IF VENTANA_OBSERVACION_YEARS < 5 THEN DO;
        FLAG_NC = 1;
        MOTIVO = 'Ventana observación < 5 años';
    END;
RUN;
