@echo off
REM Windows launcher for the live-debug demo app.
REM   scripts\run_demo.bat           -> http://127.0.0.1:8010
REM   set PORT=9000 && scripts\run_demo.bat
REM A Windows venv has no bin\ — the interpreter lives under Scripts\.
setlocal
cd /d "%~dp0.."

if "%PORT%"=="" set PORT=8010
set PY=.venv\Scripts\python.exe
set PIP=.venv\Scripts\pip.exe

REM 1. Create the venv if missing.
if not exist "%PY%" (
  where python >nul 2>nul || (echo ERROR: no python found. Install Python 3.11+ & exit /b 1)
  echo Recreating .venv from: %~dp0
  python -m venv --system-site-packages .venv
  if errorlevel 1 exit /b 1
)

REM 2. Install demo deps if missing.
"%PY%" -c "import fastapi, uvicorn, openpyxl" >nul 2>nul || (
  echo Installing demo deps...
  "%PIP%" install -q fastapi "uvicorn[standard]" openpyxl python-multipart
)

REM 3. (Re)build bundled sample assets, then serve.
"%PY%" examples\make_app_assets.py
echo Demo app running - open http://127.0.0.1:%PORT% in your browser.
set PORT=%PORT%
"%PY%" examples\app\server.py
endlocal
