@echo off
setlocal

cd /d "%~dp0"
set "DASHBOARD_URL=http://127.0.0.1:8765/"

call "%~dp0stop.cmd" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 1"

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
  curl.exe -fsS %DASHBOARD_URL%api/status >nul 2>nul
  if not errorlevel 1 goto open_dashboard
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 1"
)

:open_dashboard
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process '%DASHBOARD_URL%' | Out-Null"

endlocal
