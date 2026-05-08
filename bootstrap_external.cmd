@echo off
setlocal

cd /d "%~dp0"
python experiments\bootstrap_external_dependencies.py %*

endlocal
