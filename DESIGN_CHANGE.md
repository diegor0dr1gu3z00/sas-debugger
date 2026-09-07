# sas-debugger — As-Is Architecture & Design-Change Points

> Precise current-state description of the SAS Reconcile SLM, written as the
> baseline for a design change. "As-is" is what the code in this repo does today;
> each section closes with **Change points** — the seams a design change would
> touch. Repo root: `~/Documents/PwC/sas-debugger`.

---

## 1. What the system is

A **small language model (0.5–4B)** post-training pipeline whose artifact is a
debugging agent for a *specific* regulatory-reporting problem: given two SAS
projects (`.egp`), a schema + pipeline lineage, and a suspected set of recovery
cycles, the agent decides what is wrong and emits (a) a diagnostic SQL query and
(b) a root-cause localisation (`table / columns / transformation / reason`).

The task is **fully supervised and execution-verifiable by construction**: each
episode plants *exactly one* known defect into a synthetic DB, so ground truth
needs no real bank data and grading never needs an LLM judge.

Distinct surfaces in this repo:

| Surface | Entry point | Role |
|---|---|---|
| Post-training pipeline | `slm/*`, `vendor/*`, `scripts/*.sh` | Data gen → SFT → eval → (gated) GRPO |
| Worked example | `examples/deep_debug_run.py` | One full agent pass on an 8-table deep pipeline |
| Live-debug demo app | `examples/app/` (FastAPI + SSE), `scripts/app.sh` | Interactive demo on :8010 |

---

## 2. Top-level directory map

```
sas-debugger/
├── AGENTS.md               concise project brief + pointer to TRAINING_REPORT.md
├── README.md               public-facing overview + results table
├── TRAINING_REPORT.md      env/data/training/eval report (the canonical record)
├── requirements.txt        SFT stack; + fastapi/uvicorn/openpyxl for the demo app
├── scripts/                runnable wrappers (gen_data / train / eval / grpo_prep / app)
├── vendor/                 REGLLM prior art, vendored with attribution
│   ├── defect_catalog.py   65-68 Defect entries (target_table, oracle_sql, columns)
│   ├── generate_db.py      synthetic DB builder (clean + trap + planted rows)
│   ├── expr.py             expression oracle
│   ├── sas_logic_tree.py   vendored SAS logic-tree parser (reference)
│   └── sas_samples/        sample SAS programs (ciclos_calibrados_pipeline.sas, …)
├── slm/                    the post-training package (import path includes vendor/)
│   ├── oracle.py           deterministic grading oracle
│   ├── lineage.py          schema + 7-layer lineage context builder
│   ├── episodes.py         oracle-verified episode / SFT JSONL generator
│   ├── train_sft.py        QLoRA SFT (HF transformers + peft, no vLLM)
│   ├── evaluate.py         eval vs oracle + baselines, incl. best-of-N
│   ├── reward.py           stage-2 GRPO reward (prepared, gated)
│   └── grpo_config.py      stage-2 GRPO config (prepared, NOT launched)
├── checkpoints/            SFT artifacts (lora/ adapter) — NOT present in this repo
└── examples/
    ├── deep_debug_run.py   worked 8-table agent pass (pure, no model)
    ├── deep_debug_worked_example.md  narrative of that pass
    ├── make_app_assets.py  generates the two .egp + one .xlsx sample assets
    └── app/                FastAPI + SSE demo (server.py / agent.py / index.html)
        └── app_assets/     src_basilea.egp, rep_lgd.egp, ciclos_sospechosos.xlsx
```

---

## 3. Runtime contract (the model's I/O)

Defined in `slm/episodes.py` — this is the contract a design change must preserve
or deliberately break.

**System prompt** (`SYS_PROMPT`): "You are a senior SAS/IRB data-quality
debugger … Respond ONLY with a JSON object containing:
- `diagnostic_sql` — SQL that, run on the whole table, flags every row with the
  same incoherence (reference the suspect columns; do NOT filter by cycle id);
- `root_cause` — `{table, columns, transformation, reason}`;
- `extraction_request` — optional read-only SELECT for key confirmatory fields
  (empty string if not needed)."

**User prompt** (`_USER_TEMPLATE`):
```
{schema}

{lineage}

SUSPECTED CYCLE
  cycle_id = {cycle_id}
  reported values (table {table}):
  {row}

TASK
  Determine what is wrong with this cycle and where the discrepancy comes from.
  Print the diagnostic SQL and the root-cause localisation (table, columns,
  transformation, reason) as JSON.
```

**Gold answer** = the oracle's `gold_trace`: `{diagnostic_sql, root_cause,
extraction_request}`, corroborated by the oracle verdict (episode is kept only if
`success` is true).

> **Change points:** the prompt format is the primary behavioral lever. Any
> change to the action space (e.g. multi-step tool use, adding a second query,
> returning a query plan) means editing `SYS_PROMPT` / `_USER_TEMPLATE` in
> `slm/episodes.py`, then regenerating the dataset (`scripts/gen_data.sh`), since
> the ray-contract lives in the dataset and the collator byte-masks the prompt.

---

## 4. Data-generation path (vendored prior art → episodes)

Flow, top to bottom:

1. `vendor/defect_catalog.py` — `DEFECTS` (≈65–68 `Defect` records). Each records
   `defect_id`, `dimension` (DAMA / ISO 8000 / BCBS 239), `category`
   (`formula | consistencia | referencial | cross_table | rango`), `severity`,
   `description`, `columns` tuple, `oracle_sql` (reference check), `target_table`
   (default `ciclos_calibrados`, some `evolucion_mensual`), and a panel mutate fn.
   Exposed via `DEFECTS_BY_ID` and `defects_by_dimension()`.
2. `vendor/generate_db.py` — `build_clean_conn(seed, n_rows)` and
   `build_trap(defect, seed, n_rows, k=1)` produce the two synthetic SQLite DBs;
   `build_trap` returns `(conn, planted_pks)`. `PANEL_TABLE`/`PANEL_COLS` define
   the monthly-panel variant.
3. `slm/episodes.py:build_episode` — for a (defect, seed, split, id) it rebuilds
   both DBs, fetches the planted row as the "suspected cycle", builds the gold
   trace, and **drops the episode unless the oracle verifies `success`**. Emits a
   JSONL record carrying `schema_version`, `system`, `user`, `assistant` (gold
   JSON), `gold`, `oracle`, plus metadata (`split`, `seed`, `n_clean_rows`,
   `dimension/category/severity`).
4. `generate_episodes` — stratified: for each defect, first
   `max(1, ceil(n_per_defect × test_frac))` go to `test`; the rest `train`.
   Default `n_per_defect=34`, `n_rows=1000` → ~2,240 raw, ~2,278 verified.
5. `write_jsonl` → `data/generated/sft.jsonl`. `split_stats` reports coverage.

> **Change points:** all *task content* is generated here. Real `.egp` ingestion
> or a real DB (rather than `vendor.generate_db`) would replace step 1–2 (the
> vendored catalog + synthetic generator) — the largest structural change. The
> 7-layer lineage narrative lives in `slm/lineage.py` (`_LAYERS`,
> `pipeline_lineage_text`, `schema_text`), separate from the DB generator, so it
> can be swapped independently.

---

## 5. Grading oracle (`slm/oracle.py`)

Two independent surfaces:

- `grade_sql(defect, sql, clean, trap, planted_pks)` — execution-verified:
  runs the query on the **trap** DB (catches ≥1 planted row → `catches`) and on
  the **clean** DB (returns 0 rows → no `false_positives`). `precise` =
  `catches and not false_positives`. Also checks the query actually *references a
  suspect column* (`references_suspect_cols`) to distinguish an invariant query
  from a degenerate `WHERE pk = …` lookup. Returns `valid / catches/
  false_positives / precise / n_caught / n_clean / error`.
- `grade_localization(localization, defect)` — checks table equality and that the
  model named the **primary suspect** column (the first element of
  `defect.columns`), not merely any shared input column. Returns
  `correct / table_ok / columns_correct / primary_identified / overlap`.
- `grade_response(...)` — combines both into the episode-level `success`
  (`valid && catches && !false_positives && localization.correct`).

Helper constants: `MAIN_TABLE=ciclos_calibrados`, `AUX_TABLES`,
`PANEL_TABLE`. Regexes for table refs (`_TABLE_RE`) and column refs
(`_COLREF_RE` / `_column_refs`) are heuristics.

> **Change points:** the oracle *is* the reward/ground truth. Any change to what
> "correct" means (new defect class, new column, different grading, an LLM judge)
> touches `oracle.py`. The SQL-keyword set (`_SQL_KEYWORDS`) and the "primary
> suspect = columns[0]" rule are the two most brittle assumptions.

---

## 6. Training (`slm/train_sft.py`)

QLoRA SFT, HF transformers, **no vLLM** (Blackwell compute capability 12.0):

- `load_4bit_model` — NF4 4-bit quant, bf16 compute, `attn_implementation="sdpa"`,
  `prepare_model_for_kbit_training`.
- `make_lora` — `r=16`, `alpha=32`, `dropout=0.05`, targets
  `q/k/v/o/gate/up/down_proj`.
- `build_dataset` — filters `split == "train"`, applies the chat template to
  system+user, appends the assistant gold + EOS, and **masks prompt tokens with
  -100** (prompt-suffix loss masking) via the custom `CausalLMCollator`
  (pads input_ids/attention_mask/labels together; labels padded with -100).
- `TrainingArguments` — `bf16`, `lr=2e-4`, cosine, `grad_accum=16`,
  `per_device_batch=1`, `gradient_checkpointing` (non-reentrant), 3 epochs,
  `save_total_limit=3`, tensorboard optional.
- Output → `checkpoints/sft-0.5b/{run/, lora/}` (adapter + tokenizer).

> **Change points:** the collator, the chat-template call, and the hardcoded
> per-device batch / grad-accum (1 / 16, tuned for a 16 GB card) are the change
> seams. Switching to a different tokenizer family (non-ChatML) or to a larger
> base (0.5→4B) requires re-checking the template in `build_dataset` and the
> memory/`attn_implementation` assumptions.

---

## 7. Evaluation (`slm/evaluate.py`)

Compares three rows against the stratified held-out split, all graded by the
deterministic oracle:

- `oracle_baseline` — ground-truth SQL + gold localisation = the ceiling (1.0).
- `base_model` — the untrained instruct model, greedy decode.
- `sft_model` — QLoRA-SFT adapter, greedy decode.

Metrics: `valid_query_rate`, `catch_rate`, `false_positive_rate`,
`precision_closed`, `localization_accuracy`, `extraction_useful`,
`overall_success`. `run_best_of_n` samples N responses and keeps the
highest-reward one (deterministic, via `slm.reward.compute_reward` — no LLM
judge). Output JSON → `eval/sft-report.json` etc.

**Current numbers (held-out, n=268):**
| | valid | catch | FP | loc | success |
|---|---:|---:|---:|---:|---:|
| Oracle ceiling | 1.00 | 1.00 | 0.00 | 1.00 | **1.00** |
| Base model | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| QLoRA SFT greedy | 0.86 | 0.72 | 0.02 | 0.70 | **0.70** |
| SFT + best-of-N (6) | 1.00 | 0.90 | 0.00 | 0.90 | **0.90** |

> **Change points:** these numbers are the acceptance bar. The `best-of-N` gate
> is what currently defers stage-2 GRPO. Any design change should restate these
> as the before/after baseline and re-run `scripts/eval.sh`.

---

## 8. Stage-2 GRPO (`slm/reward.py`, `slm/grpo_config.py`) — PREPARED, GATED

`reward.py:compute_reward(env, output)` returns a bounded scalar in `[0,1]`:

```
0.40 * sql_catch        (SQL on trap catches ≥1 planted)
0.20 * sql_precision    (same SQL returns 0 rows on clean)
0.20 * localization     (table + suspect columns match)
0.20 * valid_output     (parses as JSON with the required keys)
```

`RolloutEnv` builds the clean+trap DBs per defect. `verify_reward_sanity()`
checks gold ≈ 1.0, junk ≈ 0.0.

`grpo_config.py` is **explicitly not a runner** — it documents a would-be launch
and the gate. Any GRPO run is disabled while `gate["enabled"]` stays `False`
(unlaunchable). Gate conditions: SFT plateaued below target, reward ≥ 0.9
verifiable on gold, and best-of-N does **not** already close the gap.

> **Change points:** `REWARD_WEIGHTS` and the `validate_gate` conditions. The
> hard rule to respect: do not flip `gate["enabled"]` to `True` without the gate
> clearing, and do not adopt vLLM on this Blackwell GPU (keep transformers/trl).

---

## 9. Demo / worked-example surfaces

Two separate, model-free demos (the deep 8-table pipeline is *the* story they
tell):

- `examples/deep_debug_run.py` — builds `t1_contratos … t8_final` (8 tables, 10
  layers, >7 hops). `FIELD_DEPENDENCIES` encodes the web; `build_trap`'s planted
  bug is `EAD_FUERA = CCF*OR_DISPTO` instead of `CCF*OR_DISBLE`. `grade()` runs
  each candidate SQL on clean+trap. Verdict: shallow `ECL=PD*LGD*EAD` catches
  nothing; deep `EAD_FUERA = CCF*OR_DISBLE` catches the cycle, 0 FP; localised to
  `t7_ead_cal`.
- `examples/app/` — FastAPI. `POST /api/session` accepts/loads the two `.egp` +
  `.xlsx` (or bundled samples). `GET /api/debug/{sid}` streams the agent trace as
  SSE (`header / reason / extract / sql / localize / done`). `agent.py:run_agent`
  is a fixed script that reproduces the deep pass and yields live events; every SQL
  verdict is a real oracle result. `index.html` renders the trace.
- `scripts/app.sh` — regenerates the sample assets and runs the app on `:8010`.

> **Change points:** `agent.py` is a hard-coded narrative, not an LLM call. To
> make the demo actually drive the trained model, `run_agent`'s event stream
> becomes the seam (replace the fixed steps with a real
> `generate(…)` decode + parse, reusing `slm.evaluate.generate`). The `.egp`
> assets are ZIPs of a `manifest.json` + SAS files (`make_app_assets.py`), so a
> real `.egp` parser would slot in at `_egp_lineage`.

---

## 10. Cross-cutting invariants (design-change safety rails)

1. **Synthetic only** — no real customer/bank data anywhere. `vendor/` is
   attributed prior art; the repo has no runtime dependency on RegLLM.
2. **Execution-verifiable everywhere** — both the oracle and the GRPO reward are
   computed by running emitted SQL on trap/clean DBs; there is no LLM judge.
3. **No vLLM** on the RTX 5060 Ti (Blackwell compute 12.0). SFT uses SDPA +
   bf16 NF4; GRPO must use transformers/trl, eager, generation in-rollout.
4. **GRPO is gated** — never launched while `gate["enabled"]` is `False`.
5. **Dataset is stratified** — ≥1 test episode per defect class and ≥ the
   configured test fraction.
6. **Episodes are only kept when oracle-verified** — `build_episode` drops any
   episode whose gold trace does not grade `success=True`.

---

## 11. Change-point register (quick index)

| Module | Change seam | Typical design-change lever |
|---|---|---|
| `slm/episodes.py` | `SYS_PROMPT`, `_USER_TEMPLATE`, action space | new behaviour / multi-turn |
| `vendor/*` | `defect_catalog`, `generate_db` | swap synthetic DB / real `.egp`+DB |
| `slm/lineage.py` | `_LAYERS`, `schema_text` | different pipeline / real lineage parser |
| `slm/oracle.py` | `grade_sql`, `grade_localization`, keyword sets | redefine "correct" |
| `slm/train_sft.py` | collator, chat template, batch/accum, attn | base-model or stacking change |
| `slm/evaluate.py` | metric definitions, best-of-N gate | acceptance bar / new metrics |
| `slm/reward.py` | `REWARD_WEIGHTS` | GRPO objective |
| `slm/grpo_config.py` | `gate["enabled"]`, conditions | stage-2 launch decision |
| `examples/app/agent.py` | event stream | drive demo with the real model |
| `examples/make_app_assets.py` | `.egp` ZIP manifest | real SAS parser |

A design change should first pick the register rows it touches, then restate the
Section 7 baseline as its before/after gate.