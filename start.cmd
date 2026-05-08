@echo off
setlocal

cd /d "%~dp0"

call "%~dp0stop.cmd" >nul 2>nul
timeout /t 1 /nobreak >nul

set "PYTHON_EXE=C:\Users\JackYang\Miniconda3\python.exe"
if not exist "%PYTHON_EXE%" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found in PATH.
    echo Please install Python or add it to PATH, then try again.
    exit /b 1
  )
  set "PYTHON_EXE=python"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '%PYTHON_EXE%' -WindowStyle Hidden -WorkingDirectory '%cd%' -ArgumentList 'experiments/run_dashboard_server.py' | Out-Null"

for /l %%I in (1,1,20) do (
  curl.exe -fsS http://127.0.0.1:8765/api/status >nul 2>nul
  if not errorlevel 1 goto open_dashboard
  timeout /t 1 /nobreak >nul
)

:open_dashboard
start "" http://127.0.0.1:8765/

endlocal
