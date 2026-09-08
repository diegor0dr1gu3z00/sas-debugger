#!/usr/bin/env bash
# db-debug-rl selftest: mutation/oracle contract on every external pipeline
# x defect + scripted-policy env run + gold walkthrough verification.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m db_debug_rl.selftest
