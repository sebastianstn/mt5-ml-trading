param(
    [string]$ProjectDir = "",
    [string]$TaskName = "MT5_Check_Logs_Auto",
    [int]$EveryMinutes = 60,
    [switch]$RunNow
)

# =============================================================================
# windows_register_check_logs_task.ps1
#
# Zweck:
#   Registriert einen geplanten Task, der check_logs_now.bat automatisch
#   ausfuehrt und in logs/ops_checks protokolliert.
# =============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Schtasks {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $argLine = ($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
    }) -join ' '

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()

    try {
        $process = Start-Process -FilePath "schtasks.exe" `
            -ArgumentList $argLine `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        $stdout = if (Test-Path $stdoutFile) { Get-Content -Path $stdoutFile -Raw } else { "" }
        $stderr = if (Test-Path $stderrFile) { Get-Content -Path $stderrFile -Raw } else { "" }

        return [PSCustomObject]@{
            ExitCode = $process.ExitCode
            Output = (($stdout + [Environment]::NewLine + $stderr).Trim())
        }
    }
    finally {
        Remove-Item -Path $stdoutFile -ErrorAction SilentlyContinue
        Remove-Item -Path $stderrFile -ErrorAction SilentlyContinue
    }
}

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Split-Path -Parent $PSScriptRoot
}

if ($EveryMinutes -lt 1) {
    throw "EveryMinutes muss >= 1 sein."
}

$runnerScript = Join-Path $PSScriptRoot "windows_run_check_logs_once.ps1"
if (-not (Test-Path $runnerScript)) {
    throw "Runner-Skript nicht gefunden: $runnerScript"
}

$taskNameNormalized = if ($TaskName.StartsWith("\\")) { $TaskName } else { "\$TaskName" }
$user = "$env:USERDOMAIN\$env:USERNAME"
$startTime = (Get-Date).AddMinutes(1).ToString("HH:mm")

$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runnerScript`" -ProjectDir `"$ProjectDir`""

Write-Host "Task-Registrierung startet..." -ForegroundColor Cyan
Write-Host "Task: $taskNameNormalized"
Write-Host "Intervall: alle $EveryMinutes Minuten"
Write-Host "Runner: $runnerScript"

# Bestehenden Task loeschen (wenn vorhanden)
$deleteResult = Invoke-Schtasks -Arguments @("/Delete", "/TN", $taskNameNormalized, "/F")
if ($deleteResult.ExitCode -eq 0) {
    Write-Host "[OK] Vorhandener Task geloescht" -ForegroundColor Green
}
elseif (($deleteResult.Output -match "angegebene Datei nicht finden") -or ($deleteResult.Output -match "cannot find the file")) {
    Write-Host "[OK] Kein vorhandener Task zum Loeschen (normal beim Erstlauf)" -ForegroundColor Green
}
else {
    throw "Task-Loeschung fehlgeschlagen: $($deleteResult.Output)"
}

# Task neu anlegen
$createResult = Invoke-Schtasks -Arguments @(
    "/Create", "/TN", $taskNameNormalized,
    "/SC", "MINUTE", "/MO", "$EveryMinutes", "/ST", $startTime,
    "/TR", $taskCommand, "/RU", $user, "/F"
)
if ($createResult.ExitCode -ne 0) {
    throw "Task-Erstellung fehlgeschlagen: $($createResult.Output)"
}
Write-Host "[OK] Task erstellt" -ForegroundColor Green

# Verifizieren
$queryResult = Invoke-Schtasks -Arguments @("/Query", "/TN", $taskNameNormalized, "/FO", "LIST", "/V")
if ($queryResult.ExitCode -ne 0) {
    throw "Task-Abfrage fehlgeschlagen: $($queryResult.Output)"
}
Write-Host "[OK] Task verifiziert" -ForegroundColor Green
Write-Host $queryResult.Output -ForegroundColor DarkGray

if ($RunNow) {
    $runResult = Invoke-Schtasks -Arguments @("/Run", "/TN", $taskNameNormalized)
    if ($runResult.ExitCode -ne 0) {
        throw "Task-Start fehlgeschlagen: $($runResult.Output)"
    }
    Write-Host "[OK] Task direkt gestartet" -ForegroundColor Green
}

Write-Host ""
Write-Host "Logs auf dem Laptop:" -ForegroundColor Yellow
Write-Host "  $ProjectDir\\logs\\ops_checks\\check_logs_last_run.log"
Write-Host "  $ProjectDir\\logs\\ops_checks\\check_logs_history.log"
