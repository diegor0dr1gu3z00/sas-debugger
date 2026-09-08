#!/usr/bin/env bash
# Generate oracle-verified episodes over the external production databases
# (SFT data + generalization eval set). Usage: bash scripts/db_debug_rl_data.sh [N]
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-4}"
.venv/bin/python -m db_debug_rl.episodes "$N" data/generated/external_episodes.jsonl
