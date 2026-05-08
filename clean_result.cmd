@echo off
setlocal

cd /d "%~dp0"

echo Cleaning old research results under "%CD%\results"

for %%D in (
  "results\runs"
  "results\leaderboards"
  "results\reports"
  "results\cross_validation"
  "results\datasets"
  "results\models"
  "results\research_memory"
  "results\dashboard\data"
  "results\best_agents"
) do (
  if exist %%~D (
    rmdir /s /q %%~D
    echo Removed %%~D
  )
)

mkdir "results" >nul 2>nul

echo Done.
endlocal
