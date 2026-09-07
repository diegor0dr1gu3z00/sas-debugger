# Worked example — a whole agent debugging pass on a *deep* table

This is a concrete, end-to-end example of the agent doing its job on a pipeline
that is **specially deep, with many dependencies and many fields** — the kind of
table the captain actually has.  Every SQL verdict below is produced by the
deterministic oracle running the query against the real trap and clean
databases, not hand-written.  You can reproduce the whole transcript with:

```bash
.venv/bin/python examples/deep_debug_run.py
```

> The actual production case would feed the agent a **`.egp` project** (the SAS
> programs) plus the **Excel of "cycles"** the captain has doubts about.  Here we
> simulate that with a synthetic 8-table credit-risk pipeline and one flagged
> cycle whose `ECL` is wrong.

---

## 1. The deep pipeline

The final reporting table is `t8_final`.  Its `ECL` is the end of a chain that
passes through **8 tables / 10 transformation layers**:

```
t1_contratos ──► t2_basilea ──► t3_colaterales ──► t4_ciclos
      │                │               │                 │
      ▼                ▼               ▼                 ▼
   (segment)       (exposure)      (collateral)      (cycle state)
      │                │               │                 │
      └────────► t5_pd_cal ◄───────────┘                 │
                  │                                      │
                  └────────► t6_lgd_cal ◄─────────────────┘
                                 │
                                 └────────► t7_ead_cal ◄── (from t2)
                                                │
                                                └────────► t8_final  (ECL, RWA, PROVISION)
```

**"Many fields / many dependencies":** `t8_final.ECL` depends *transitively* on
**15 fields across 7 tables**.  The dependency web:

```
ECL ← PD_FINAL, LGD_FINAL, EAD_TOTAL
  PD_FINAL ← PD_ESTIMADA, PD_SUELO
  PD_SUELO ← SEGMENTO, GARANTIA_TIPO
  LGD_FINAL ← LGD_CON_MOC, LGD_SUELO
  LGD_SUELO ← GARANTIA_TIPO, SEGMENTO
  LGD_CON_MOC ← LGD_ESTIMADA
  EAD_TOTAL ← EAD_BALANCE, EAD_FUERA
  EAD_BALANCE ← OR_DISPTO
  EAD_FUERA ← CCF, OR_DISBLE
```

`ECL` also feeds `PROVISION`, and together with `K_IRB` feeds `RWA`.  So a single
suspected value is a *leaf* of a wide, deep graph — and the bug may live anywhere
upstream.

---

## 2. The discrepancy (the "cycle" the captain doubts)

Cycle `__DIRTY_DEEP_3__` is flagged.  The final table reports:

```
SEGMENTO  = CORP
EAD_TOTAL = 802996.915
PD_FINAL  = 0.019352
LGD_FINAL = 0.657119
ECL       = 10211.370
RWA       = 118874.237
```

The captain only knows `ECL` "looks wrong".  Where did it come from?

---

## 3. The agent loop

### Step 1 — shallow hypothesis: is the final formula broken?

The agent requests a key-table extraction of the ECL inputs and checks the most
obvious invariant — the formula in the final table:

```sql
SELECT ID_CICLO FROM t8_final
WHERE ABS(ECL - PD_FINAL*LGD_FINAL*EAD_TOTAL) > 0.01
```

**Oracle:** `valid=true`, `caught=false`, `false_positive=false` (0 rows on trap,
0 rows on clean).

> The check **runs but catches nothing**.  The formula `ECL = PD*LGD*EAD` is
> internally consistent — the bug is **not** in the final formula.  This is the
> classic trap with deep tables: a shallow check of the leaf formula never finds a
> fault that lives in an *input* transformation.

### Step 2 — trace the lineage and request a key-table extraction

The agent follows the dependency graph up from `ECL` and asks for the upstream
source values, joining the EAD table back to its Basilea source:

```sql
SELECT e.ID_CICLO, e.EAD_BALANCE, e.EAD_FUERA, e.EAD_TOTAL,
       b.OR_DISPTO, b.OR_DISBLE, b.CCF
FROM t7_ead_cal e JOIN t2_basilea b
     ON e.ID_CONTRATO = b.ID_CONTRATO AND e.MES = b.MES
WHERE e.ID_CICLO = '__DIRTY_DEEP_3__'
```

**Extraction returns:**

```
ID_CICLO=__DIRTY_DEEP_3__  EAD_BALANCE=458855.38  EAD_FUERA=344141.535
EAD_TOTAL=802996.915  OR_DISPTO=458855.38  OR_DISBLE=20094.18  CCF=0.75
```

The agent spots the discrepancy immediately:

> `EAD_FUERA` should equal `CCF * OR_DISBLE` = `0.75 * 20094.18` = **15070.64**,
> but it is **344141.535**.  The EAD layer built `EAD_FUERA` from `OR_DISPTO`
> (the drawn amount) instead of `OR_DISBLE` (the undrawn), so `EAD_TOTAL` — and
> therefore `ECL` and `RWA` — are inflated.

### Step 3 — deep, cross-table diagnostic SQL

Now the agent localises the fault to the EAD transformation and emits a
cross-table reconciliation check:

```sql
SELECT e.ID_CICLO
FROM t7_ead_cal e JOIN t2_basilea b
     ON e.ID_CONTRATO = b.ID_CONTRATO AND e.MES = b.MES
WHERE ABS(e.EAD_FUERA - b.CCF*b.OR_DISBLE) > 0.01
```

**Oracle:** `valid=true`, `caught=true`, `false_positive=false` (1 row on trap,
0 rows on clean).

> **Caught the flagged cycle with zero false positives.**  The invariant
> `EAD_FUERA = CCF * OR_DISBLE` is violated only by the deep bug.  A shallow
> check could never have found it; the agent had to trace the lineage and pull
> the source table.

### Step 4 — final localisation

```json
{
  "table": "t7_ead_cal",
  "columns": ["EAD_FUERA", "OR_DISBLE", "CCF"],
  "transformation": "EAD conversion: EAD_FUERA = CCF * OR_DISBLE (CRR Art. 166). "
                    "The implementation used OR_DISPTO instead of OR_DISBLE, "
                    "inflating EAD_TOTAL and hence ECL/RWA.",
  "reason": "Reported EAD_FUERA does not equal CCF*OR_DISBLE; the wrong input "
            "field was used in the EAD transformation, which propagates up to a "
            "wrong ECL in t8_final."
}
```

**Verdict:** `t7_ead_cal` · `EAD_FUERA` · **EAD transformation (wrong input field
`OR_DISPTO` vs `OR_DISBLE`)** — the root cause is **6 hops upstream** of the
flagged `ECL`.

---

## 4. Why this matters for the product

- Deep pipelines hide bugs in **input transformations**, not in the final
  formula.  The agent must *trace lineage* and *request key-table extractions*,
  not just check the leaf formula.
- The oracle's **execution-verified** grading (does the query catch the planted
  row? does it fire on clean data?) is what makes this trainable end-to-end
  (SFT) and then optimisable (GRPO) with a deterministic, non-LLM reward.
- In production the same flow applies: the agent reads the `.egp` lineage, spots
  the suspect cycles, issues extraction requests + diagnostic SQL, and localises
  the table / field / transformation — the diagnostic SQL is handed back to the
  captain to run.

## 5. Files

- `examples/deep_debug_run.py` — the runnable, self-contained generator + agent
  simulation that emits the transcript above (deep 8-table pipeline, real oracle
  verdicts).
- `slm/oracle.py`, `slm/lineage.py` — the general grading oracle and the compact
  schema + lineage context used for the real SFT/GRPO episodes.
- `slm/episodes.py` — the full, verified 2,278-episode dataset generator.
