#!/usr/bin/env bash
# Evaluate an adapter (or the base model) vs the deterministic oracle baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/sft-0.5b/lora}"   # empty => base model
TEST="${TEST:-data/generated/sft.jsonl}"
OUT="${OUT:-eval/sft-report.json}"

.venv/bin/python -m slm.evaluate \
  --test_jsonl "$TEST" \
  --base_model "$BASE" \
  --adapter "$ADAPTER" \
  --json_out "$OUT"
echo "wrote $OUT"
