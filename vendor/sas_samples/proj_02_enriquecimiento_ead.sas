/*****************************************************************************
 * proj_02_enriquecimiento_ead.sas — Enriquecimiento de EAD
 * Ajusta la exposición con titulización y consolida el EAD por ciclo.
 * Regulacion: CRR Art. 166
 *****************************************************************************/

/* Enriquecimiento de EAD con ajuste de titulización */
DATA work.ciclos_ead;
    SET work.ciclos_activos;

    /* Titulización: para CORP el EAD titulizado dobla el original */
    IF SEGMENTO = 'CORP' THEN OR_EAD_TIT = OR_EAD * 2;
    ELSE OR_EAD_TIT = OR_EAD;

    /* EAD final = exposición original + intereses devengados */
    EAD = OR_EAD + INTERESES_ACUMULADOS;

    /* Nota: si OR_EAD llega missing desde BASILEA el EAD queda missing
       y se propaga a ECL — validar con el equipo de fuentes. */
RUN;
