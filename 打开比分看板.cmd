@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
if exist ".venv\Scripts\python.exe" (
  set "VEX_PYTHON=.venv\Scripts\python.exe"
) else if exist "..\.venv-vex\Scripts\python.exe" (
  set "VEX_PYTHON=..\.venv-vex\Scripts\python.exe"
) else (
  echo Please run setup.cmd first.
  pause
  exit /b 1
)
"%VEX_PYTHON%" score_studio.py
if errorlevel 1 pause
