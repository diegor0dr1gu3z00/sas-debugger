# db-debug-rl — RL environment: debugging real production databases

Multi-turn RL environment that teaches a small LM to localise the root cause
of a table discrepancy **inside large, real, join-heavy databases**, by
forming hypotheses, probing with read-only SQL, and finishing with a
localisation + diagnostic query. Ground truth is correct **by construction**
(we authored the pipeline and the defect), so the reward is fully
execution-verifiable — the same mutation-testing contract as
`vendor/defect_catalog` + `slm/oracle`, generalised from the synthetic bank
schema to real databases.

## What an episode looks like

1. A **pipeline** (staging steps -> final reporting table) is materialized in
   memory over one of the external databases in `data/external/sql`
   (chinook, northwind, sakila, classicmodels, world, f1db, openflights,
   imdb, tpch, tpcds).
2. The final materialization is replaced with a **mutated** version
   (`db_debug_rl/defects.py`): wrong join key, missing dedup, double-counting
   fan-out, filter drift, time-boundary drift, formula drift, inner-vs-left.
3. The trap DB contains the corrupted reported table plus a hidden `zz_truth`
   (the clean recomputation, blocked from the agent). One suspect row —
   sampled from the gold diagnostic's own result stream — is shown to the
   agent (for missing-row defects, the reference-side row).
4. The agent acts in a loop:
   `{"action": "sql"}` (read-only probe) → `{"action": "hypothesis"}` →
   `{"action": "localize", root_cause, diagnostic_sql}`.
5. Terminal reward (execution-verifiable, same weights as `slm.reward`):
   `0.40 catch + 0.20 precision + 0.20 localisation + 0.20 valid output`,
   plus a <=0.10 hypothesis-shaping bonus.

## Walkthroughs

`db_debug_rl/walkthrough.py` adds the walkthrough capability: explain HOW the
suspect value was produced, layer by layer (final table -> staging -> raw
sources), with one evidence SQL per layer. Verification is execution-based:
evidence must run on the working DB, the first step must return the suspect
row, and layers must match the pipeline backwards. Gold walkthroughs are
constructed from the pipeline definition (score 1.0 by construction) and are
stored inside every episode.

## Modules

| File | Role |
|------|------|
| `db.py` | SQLite helpers, schema text, read-only guarded `exec_sql` (timeouts, LIMIT) |
| `pipelines.py` | 10 materialized pipelines over the external DBs (schemas, lineage, indexes) |
| `defects.py` | 21 join-focused defects + the reconciliation-diff oracle generator |
| `oracle.py` | clean/trap construction (shared `PipelineCache`), planting, grading |
| `env.py` | `DbgRLEnv` multi-turn loop (sql / hypothesis / localize) |
| `reward.py` | terminal + shaping reward (execution-verifiable) |
| `walkthrough.py` | walkthrough gold + verifier |
| `episodes.py` | oracle-verified SFT/eval episodes (`db-debug-rl-episode/v2`) |
| `rollout.py` | batch rollouts for a gated GRPO stage (see `slm/grpo_config.py`) |
| `selftest.py` | full contract selftest (mutation, oracle, env, walkthrough) |

## Run

```bash
bash scripts/build_validation_dbs.py   # download + verify the external DBs (once)
bash scripts/db_debug_rl_smoke.sh      # SELFTEST must PASS
bash scripts/db_debug_rl_data.sh 4     # 84 episodes -> data/generated/external_episodes.jsonl
```

Pipelines are tagged `weight="light"|"heavy"` — restrict GRPO/eval sampling to
light pipelines for fast rollouts; heavy ones (northwind/tpch/tpcds/imdb/f1)
are for periodic generalization checks.
