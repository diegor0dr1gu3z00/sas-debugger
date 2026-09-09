#!/usr/bin/env bash
# Regenerate the verified SFT dataset (>=2k episodes, stratified >=10% test).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_venv.sh

N_PER_DEFECT="${N_PER_DEFECT:-34}"
ROWS="${ROWS:-1000}"
OUT="${OUT:-data/generated/sft.jsonl}"

"$(venv_py)" -m slm.episodes "$N_PER_DEFECT" "$ROWS" "$OUT"
echo "wrote $OUT"
