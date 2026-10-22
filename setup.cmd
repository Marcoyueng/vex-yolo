@echo off
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "TEMP=%~dp0runs\temp"
set "TMP=%~dp0runs\temp"
if not exist "%TEMP%" mkdir "%TEMP%"
python -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation complete. Open the launcher to start.
pause
exit /b 0
:failed
echo Installation failed. Please check Python and your network connection.
pause
exit /b 1
