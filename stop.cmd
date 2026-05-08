@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue'; " ^
  "$repoRoot = [System.IO.Path]::GetFullPath('%REPO_ROOT%'); " ^
  "$controlDir = Join-Path $repoRoot 'results\dashboard_control'; " ^
  "$statusPath = Join-Path $controlDir 'status.json'; " ^
  "$commandPath = Join-Path $controlDir 'command.json'; " ^
  "$timestamp = [DateTimeOffset]::Now.ToString('o'); " ^
  "$serverTargets = Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^pythonw?\.exe$') -and $_.CommandLine -and $_.CommandLine.Contains('run_dashboard_server.py') }; " ^
  "$status = $null; " ^
  "if (Test-Path $statusPath) { try { $status = Get-Content -Path $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $status = $null } }; " ^
  "$researchPid = $null; " ^
  "if ($status -and $status.pid) { try { $researchPid = [int]$status.pid } catch { $researchPid = $null } }; " ^
  "$hadResearch = $false; " ^
  "if ($researchPid) { $hadResearch = [bool](Get-Process -Id $researchPid) }; " ^
  "if (-not $hadResearch) { " ^
  "  $researchTarget = Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^pythonw?\.exe$') -and $_.CommandLine -and (($_.CommandLine.Contains('run_research_loop.py')) -or ($_.CommandLine.Contains('run_batch.py'))) -and $_.CommandLine.Contains('dashboard_control') } | Select-Object -First 1; " ^
  "  if ($researchTarget) { $researchPid = [int]$researchTarget.ProcessId; $hadResearch = $true } " ^
  "}; " ^
  "if ($hadResearch) { " ^
  "  New-Item -ItemType Directory -Path $controlDir -Force | Out-Null; " ^
  "  @{ action = 'stop'; updated_at = $timestamp } | ConvertTo-Json | Set-Content -Path $commandPath -Encoding UTF8; " ^
  "  if ($status) { " ^
  "    $status.status = 'stopping'; " ^
  "    $status.message = 'Stop requested by stop.cmd.'; " ^
  "    $status.updated_at = $timestamp; " ^
  "    $status | ConvertTo-Json -Depth 10 | Set-Content -Path $statusPath -Encoding UTF8; " ^
  "  }; " ^
  "  $deadline = (Get-Date).AddSeconds(12); " ^
  "  while ((Get-Date) -lt $deadline) { " ^
  "    if (-not (Get-Process -Id $researchPid)) { break }; " ^
  "    Start-Sleep -Milliseconds 500; " ^
  "  }; " ^
  "  if (Get-Process -Id $researchPid) { Stop-Process -Id $researchPid -Force }; " ^
  "  if (Test-Path $statusPath) { " ^
  "    try { " ^
  "      $finalStatus = Get-Content -Path $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json; " ^
  "      $finalStatus.status = 'stopped'; " ^
  "      $finalStatus.message = 'Auto research stopped by stop.cmd.'; " ^
  "      $finalStatus.completed_at = [DateTimeOffset]::Now.ToString('o'); " ^
  "      $finalStatus.updated_at = [DateTimeOffset]::Now.ToString('o'); " ^
  "      $finalStatus | ConvertTo-Json -Depth 10 | Set-Content -Path $statusPath -Encoding UTF8; " ^
  "    } catch { } " ^
  "  } " ^
  "}; " ^
  "if ($serverTargets) { $serverTargets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force } }; " ^
  "$messages = @(); " ^
  "if ($hadResearch) { $messages += 'Auto research stopped.' } else { $messages += 'No active auto research process found.' }; " ^
  "if ($serverTargets) { $messages += 'Dashboard server stopped.' } else { $messages += 'Dashboard server was not running.' }; " ^
  "Write-Output ($messages -join ' ')"

endlocal
