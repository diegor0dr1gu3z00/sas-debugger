/*****************************************************************************
 * proj_01_carga_ciclos.sas — Carga de ciclos de recuperación
 * Filtra ciclos activos desde mylib.ciclos_recuperacion.
 * Regulacion: EBA GL 2017/16 §6.3 (ventana de observación)
 *****************************************************************************/

LIBNAME mylib '/data/irb/calibracion';

/* Carga y filtro de ciclos activos */
DATA work.ciclos_activos;
    SET mylib.ciclos_recuperacion;
    WHERE ESTADO_CICLO NE 'CERRADO';

    /* BUG conocido: los contratos fusionados (SW_FUSION=1) llegan sin
       LGD_ESTIMADA — el catálogo de la entidad absorbida no informa el
       campo y el MERGE posterior no aplica COALESCE. MoC y ECL quedan
       missing para esos ciclos. */
RUN;
