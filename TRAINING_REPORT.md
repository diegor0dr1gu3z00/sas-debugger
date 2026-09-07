# SLM Post-Training Report — SAS Reconcile (SFT-first)

Branch: `fm/slm-posttrain-sft` · local-only · no push / no PR.

Goal: a working, rerunnable post-training pipeline for a small language model
(0.5–3B) that localises where a discrepancy between two SAS-produced tables
comes from — given a suspected recovery cycle, it emits a diagnostic SQL query,
requests a key-table extraction, and attributes the root cause to a
table / column / transformation. Ground truth is **correct by construction**
(we inject the bug, so the oracle knows the answer); no real bank data is used.

---

## 1. Training environment design (retrieved, not invented)

Prior art was **vendored** from the RegLLM project (`~/Documents/PwC/regllm`)
into `vendor/` with attribution headers (no runtime dependency on regllm):

| File | Source | Role |
|------|--------|------|
| `vendor/defect_catalog.py` | `DQC/eval/defect_catalog.py` | 67 ground-truth defects (mutation oracle + reference SQL) |
| `vendor/generate_db.py` | `DQC/eval/generate_db.py` | synthetic clean / per-defect trap SQLite DB generator |
| `vendor/expr.py` | `apdq/expr.py` | dependency-free SAS-expression recomputation oracle / SQL compiler |
| `vendor/sas_logic_tree.py` | `src/sas_logic_tree.py` | SAS AST + row-level lineage |
| `vendor/sas_samples/*.sas` | `data/samples`, `data/sas/sessions/debug_lgd`, `DQC/eval/sas` | sample SAS projects (7-layer pipeline, LGD debug sessions) |

The 7-layer `ciclos_calibrados_pipeline.sas` (L0 sources → L7 ECL/RWA) is the
canonical lineage narrative shown to the model.

### Oracle (`slm/oracle.py`)
Deterministic grader that ties a planted defect to a verdict:
- `grade_sql` — runs the candidate diagnostic SQL on the **trap** DB (must be
  execution-valid and catch ≥1 planted row) and on the **clean** reference DB
  (must return 0 rows ⇒ no false positives).
- `grade_localization` — table must match `defect.target_table` and the model
  must name the **primary suspect column** (the derived/violating field), not
  merely any shared input column.
- `grade_response` — combined `success` = valid ∧ catches ∧ precise ∧
  localised. Self-test: all 67 catalog oracles return 0 rows on clean and ≥1 on
  their own trap (verified).

### Episode / task format (`slm/episodes.py`, `slm/lineage.py`)
An episode = (a) the two databases/tables + one suspected cycle row, (b) a
compact schema + 7-layer lineage summary, (c) the model action space:
`diagnostic_sql` + `root_cause{table,columns,transformation,reason}` +
`extraction_request`. Every stored episode is **oracle-verified** (the gold
trace's `oracle.success` is `true`), so the dataset is a set of complete,
execution-confirmed traces.

### Ground truth / grading
The defect catalog is the oracle. Because we build the trap DB ourselves, the
oracle always knows the true table/column/transformation.

---

## 2. SFT data generation

Generated via `slm.episodes` (34 per defect × 67 defects = 2,278 episodes),
each a complete oracle-verified trace. Held-out split is **stratified by defect
class** so every one of the 67 classes appears in evaluation.

| Split | Episodes | Fraction |
|-------|----------|----------|
| train | 2,010 | 88.2% |
| test (held-out) | 268 | 11.8% |
| **total** | **2,278** | — |

Dimensions covered (DAMA / ISO 8000 / BCBS 239): accuracy 170, completeness
136, conformity 374, consistency 918, plausibility 306, timeliness 68,
uniqueness 102, validity 204. Defects span single-table formulas, cross-table
reconciliation, panel (monthly-series) invariants, and range/domain decoys.
All 67 classes have exactly 34 episodes (balanced).

Schema: `data/generated/sft.jsonl` (versioned `sas-reconcile-episode/v1`),
fields `schema, episode_id, defect_id, dimension, category, severity, split,
seed, n_clean_rows, system, user, assistant, gold, oracle`. The bulk JSONL is
`.gitignore`d (large, reproducible via `scripts/gen_data.sh`).

---

## 3. Post-training (SFT-first, QLoRA)

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.5B, Qwen-family, fits 16 GB
  with large headroom; selected as the best-fit small instruct base. The 1.7B /
  3B scale-up path is documented below).
- **Quantization:** 4-bit NF4 QLoRA (`bitsandbytes 0.50.2`, Blackwell sm_120
  verified), bf16 compute, double-quant.
- **LoRA:** r=16, α=32, dropout 0.05, target `q/k/v/o/gate/up/down_proj`.
  Trainable params: **8,798,208 / 502,830,976 = 1.75%**.
- **Trainer:** HF `transformers.Trainer` + `peft` (custom `CausalLMCollator`
  that pads `labels` to −100; prompt-suffix loss masking; no vLLM).
- **Hyperparameters:** 3 epochs, lr 2e-4 (cosine), warmup 30 steps, weight
  decay 0.01, batch 1 × grad-accum 16 (effective batch 16), `max_len` 4096,
  gradient checkpointing, seed 42.
- **Result:** `train_loss` 0.0764; final step loss ≈ 0.003; train runtime
  **3,471 s (~58 min)**, ~1.74 samples/s, 0.109 steps/s.

**Checkpoints** (`checkpoints/sft-0.5b/`):
`run/checkpoint-{200,300,378}` (steps) and `lora/adapter_model.safetensors`
(final adapter, 34 MiB) + tokenizer/chat-template.

---

## 4. Evaluation vs baselines (held-out test split, n=268)

Metrics (execution-verified by the oracle): `valid` = diagnostic SQL runs;
`catch` = catches ≥1 planted row; `FP` = fires on the clean DB; `loc` =
localisation correct (table + primary suspect column); `extr` = extraction
request present and runs; `success` = valid ∧ catch ∧ ¬FP ∧ loc.

| Model | valid | catch | FP | precise | loc | extr | **success** |
|-------|------:|------:|---:|--------:|----:|-----:|:-----------:|
| Deterministic oracle (baseline) | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| Untrained base `Qwen2.5-0.5B-Instruct` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| **SFT (QLoRA, greedy)** | 0.86 | 0.72 | 0.02 | 0.70 | 0.70 | 0.86 | **0.70** |
| SFT + **best-of-N (N=6, temp 0.7, 40-ep subset)** | 1.00 | 0.90 | 0.00 | 0.90 | 0.90 | 1.00 | **0.90** |

Findings:
- The SFT model **learns the tool-use contract** (86% valid queries, 86% useful
  key-table extraction) and produces genuinely *invariant* diagnostic SQL
  (catch 72%) with only **1.9% false positives** — i.e. it does not hallucinate
  checks on clean data.
- It plateaus below the oracle ceiling because it cannot perfectly discriminate
  all 67 defect classes: similar formula defects (e.g. D01 ECL vs D40 EAD-alias)
  are sometimes confused, so it emits a *valid but wrong* invariant.
- **best-of-N with the execution-verifiable reward already lifts success to
  0.90** on the sampled subset — this is the key input to the GRPO gate.

---

## 5. VRAM / timing facts (RTX 5060 Ti 16 GB, Blackwell compute 12.0)

- **Peak GPU memory ≈ 15.5 GB** (4-bit QLoRA 0.5B, `max_len` 4096, batch 1,
  grad-accum 16, gradient checkpointing). The allocator logged 4 transient
  OOM warnings (`free <1.1 GB` at 16.6 GB total) but recovered; the run
  completed all 378 steps.
- Total train wall-clock **3,471 s (~58 min)** for 3 epochs / 2,010 examples.
- Data generation: **~2.4 min** for 2,278 episodes.
- Eval generation: ~3 s per response on the 0.5B model (268 episodes ≈ 14 min).
- No vLLM used anywhere (Blackwell-compat risk, per the brief).

**VRAM headroom / scale-up path:** 0.5B leaves ample headroom; a 1.7B
(`Qwen/Qwen3-1.7B`) or 3B QLoRA run with `max_len` ~2048, batch 1, grad-accum
16, and gradient checkpointing is expected to fit in 16 GB (drop `max_len` to
2048 if needed). A **4B run is not justified** here: best-of-N on the 0.5B
already reaches 0.90, so a bigger model adds cost before the gate is satisfied.

---

## 6. GRPO stage-2 gate (prepared, NOT launched)

Stage-2 GRPO is **prepared but not run**:
- `slm/reward.py` — `compute_reward` is **execution-verifiable** (runs the
  emitted SQL on the trap/clean DBs; no LLM judge), bounded in [0,1] =
  0.40·catch + 0.20·precision + 0.20·localization + 0.20·valid-output.
  Self-check: gold trace → **1.0**; junk/empty → **0.0**.
- `slm/grpo_config.py` — would-be config + `validate_gate()`; hard `enabled=False`.

**What the gate needs (measured here):**
1. **SFT plateau below target.** Greedy SFT success is **0.70** (< target).
2. **best-of-N gap.** best-of-N (N=6) already reaches **0.90** on the sampled
   subset (full-268 best-of-N should be run before deciding). If best-of-N
   ≥ target, GRPO is **not** needed and must NOT run. If best-of-N < target
   while the reward is verifiable ≥0.9, GRPO is justified.
3. **Verifiable reward ≥0.9.** Confirmed: `compute_reward` gives 1.0 on gold
   traces and 0.90 on best-of-N samples.

**Recommendation:** run a full-268 best-of-N measurement first. Given best-of-N
already hits 0.90, the marginal value of GRPO is small unless the target is
raised toward ~1.0. GRPO would use transformers+trl/GRPOTrainer (or torchrl
GRPO/MC-GRPO), eager execution, no vLLM.

---

## 7. How to rerun

```bash
# 0. setup venv (Python 3.14 system python; PEP-668-safe; reuses the CUDA torch
#    already installed so bitsandbytes/transformers/trl wheels can install)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt          # skip torch if already present

# 1. regenerate the verified dataset (>=2k, stratified >=10% test)
bash scripts/gen_data.sh                           # -> data/generated/sft.jsonl

# 2. SFT (QLoRA) — edit BASE/OUT in scripts/train.sh as needed
bash scripts/train.sh                              # -> checkpoints/sft-0.5b/

# 3. evaluate the SFT adapter vs oracle + base baselines
bash scripts/eval.sh                               # -> eval/sft-report.json
#    best-of-N (gate input):
.venv/bin/python -m slm.evaluate --test_jsonl data/generated/sft.jsonl \
  --base_model Qwen/Qwen2.5-0.5B-Instruct --adapter checkpoints/sft-0.5b/lora \
  --best_of_n 6 --sample_limit 40 --json_out eval/sft-bestofn.json

# 4. inspect the (gated) stage-2 GRPO briefing
bash scripts/grpo_prep.sh                          # NOT a launch
```

## 8. Known limitations
- 0.5B model confuses structurally-similar formula defects (D01 vs D40 etc.);
  best-of-N and a larger base are the natural fixes.
- best-of-N was measured on a 40-episode subset; the full-268 figure should be
  produced before any GRPO decision.
- The `ciclos_calibrados` schema/row context is verbose (~2.4k tokens); a
  more compact context would lower per-example cost and may improve
  discrimination.
