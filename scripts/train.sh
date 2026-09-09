#!/usr/bin/env bash
# QLoRA SFT of the SAS-reconcile SLM. Edit BASE / OUT per run.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_venv.sh

BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT="${OUT:-checkpoints/sft-0.5b}"
DATA="${DATA:-data/generated/sft.jsonl}"

"$(venv_py)" -m slm.train_sft \
  --base_model "$BASE" \
  --out_dir "$OUT" \
  --data_path "$DATA" \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-4 \
  --r 16 \
  --alpha 32 \
  --max_len 4096 \
  --save_steps 500 \
  --logging_steps 10 \
  --seed 42 \
  --tensorboard
