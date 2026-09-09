#!/usr/bin/env bash
# Stage-2 GRPO PREP (not a launch). Prints the would-be config and the gates.
# The council gate must clear BEFORE any GRPO run; this script never launches it.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_venv.sh

"$(venv_py)" -m slm.grpo_config
echo
echo "GRPO stage-2 is NOT launched. See TRAINING_REPORT.md for the gate."
