#!/usr/bin/env bash
# db-debug-rl selftest: mutation/oracle contract on every external pipeline
# x defect + scripted-policy env run + gold walkthrough verification.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_venv.sh
"$(venv_py)" -m db_debug_rl.selftest
