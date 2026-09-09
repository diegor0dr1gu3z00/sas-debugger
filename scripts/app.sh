#!/usr/bin/env bash
# Run the live-debug demo web app. Open the printed URL in a browser.
#   PORT=8010 bash scripts/app.sh
#
# Thin wrapper around scripts/run_demo.sh, which (re)creates the venv if broken,
# resolves the platform venv path (Scripts/ vs bin/), installs the demo deps if
# missing and then serves: it always works on a fresh machine.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_demo.sh
