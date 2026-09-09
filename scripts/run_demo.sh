#!/usr/bin/env bash
# Minimal launcher for the live-debug demo web app.
#   bash scripts/run_demo.sh            # http://127.0.0.1:8010
#   PORT=9000 bash scripts/run_demo.sh
#
# Fixes the usual "no se encontro python" failure: it never assumes a fixed
# venv path. It resolves the interpreter from the layout that actually exists
# (Windows venv: .venv/Scripts/python.exe; Unix: .venv/bin/python), (re)creates
# the venv if missing/broken, installs deps if the app cannot import them, then
# runs uvicorn. No npm — index.html is served by FastAPI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. scripts/_venv.sh
PORT="${PORT:-8010}"

# 1. Find a usable interpreter. Never trust a stale venv layout.
need_py() { "$(venv_py)" -c 'import sys' 2>/dev/null; }
if ! need_py; then
  SYS_PY="$(command -v python3 || command -v python)"
  [[ -n "$SYS_PY" ]] || { echo "ERROR: no python3 found. Install Python 3.11+." >&2; exit 1; }
  echo "Recreating .venv from: $SYS_PY"
  rm -rf .venv
  "$SYS_PY" -m venv --system-site-packages .venv
fi

# 2. Ensure the demo deps are installed (fastapi/uvicorn are enough here).
if ! "$(venv_py)" -c 'import fastapi, uvicorn, openpyxl' 2>/dev/null; then
  echo "Installing demo deps..."
  "$(venv_pip)" install -q fastapi 'uvicorn[standard]' openpyxl python-multipart
fi

# 3. (Re)build the bundled sample assets (two .egp + one .xlsx).
"$(venv_py)" examples/make_app_assets.py

# 4. Serve. server.py runs uvicorn bound to 127.0.0.1:PORT; the page is at /.
echo "Demo app running — open http://127.0.0.1:${PORT} in your browser."
echo "Press Ctrl-C to stop."
PORT="$PORT" exec "$(venv_py)" examples/app/server.py
