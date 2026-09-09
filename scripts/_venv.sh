#!/usr/bin/env bash
# Resolve the project venv interpreter/pip regardless of platform.
#
# A Windows venv has NO bin/ directory — the interpreter and pip live under
# Scripts/ with a .exe suffix (e.g. .venv/Scripts/python.exe). A Unix venv puts
# them under bin/ (e.g. .venv/bin/python). These helpers detect whichever is
# present at call time and fall back to the Windows layout when neither exists
# (the layout we assume is being provisioned).
#
# Source from a script that has already cd'd to the repo root:
#   . scripts/_venv.sh
# Then use:
#   "$(venv_py)" examples/app/server.py
#   "$(venv_pip)" install -r requirements.txt

venv_py() {
  local p
  for p in .venv/Scripts/python.exe .venv/bin/python; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  printf '%s' ".venv/Scripts/python.exe"
}

venv_pip() {
  local p
  for p in .venv/Scripts/pip.exe .venv/bin/pip; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  printf '%s' ".venv/Scripts/pip.exe"
}
