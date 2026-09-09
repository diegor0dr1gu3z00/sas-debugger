#!/usr/bin/env bash
# Run the live-debug demo web app. Open the printed URL in a browser.
#   PORT=8010 bash scripts/app.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_venv.sh

PORT="${PORT:-8010}"

# (Re)create the bundled sample assets (two .egp projects + one .xlsx).
"$(venv_py)" examples/make_app_assets.py

echo "Demo app running — open http://127.0.0.1:${PORT} in your browser."
echo "Press Ctrl-C to stop."
PORT="$PORT" exec "$(venv_py)" examples/app/server.py
