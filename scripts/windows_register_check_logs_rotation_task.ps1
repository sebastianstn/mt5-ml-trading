param(
    [string]$ProjectDir = "",
    [string]$TaskName = "MT5_Check_Logs_Rotate_Daily",
    [string]$AtTime = "23:59",
    [switch]$RunNow
)

# =============================================================================
# windows_register_check_logs_rotation_task.ps1
#
# Zweck:
#   Registriert einen taeglichen Task zur Rotation von check_logs_history.log.
# =============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Schtasks {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $argLine = ($Arguments | ForEach-Object {
        if ($_ -match '\\s') { '"{0}"' -f $_ } else { $_ }
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

$runnerScript = Join-Path $PSScriptRoot "windows_rotate_check_logs_daily.ps1"
if (-not (Test-Path $runnerScript)) {
    throw "Rotation-Skript nicht gefunden: $runnerScript"
}

$taskNameNormalized = if ($TaskName.StartsWith("\\")) { $TaskName } else { "\$TaskName" }
$user = "$env:USERDOMAIN\$env:USERNAME"
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runnerScript`" -ProjectDir `"$ProjectDir`""

Write-Host "Rotation-Task Registrierung startet..." -ForegroundColor Cyan
Write-Host "Task: $taskNameNormalized"
Write-Host "Zeit: $AtTime"
Write-Host "Runner: $runnerScript"

$deleteResult = Invoke-Schtasks -Arguments @("/Delete", "/TN", $taskNameNormalized, "/F")
if ($deleteResult.ExitCode -eq 0) {
    Write-Host "[OK] Vorhandener Rotation-Task geloescht" -ForegroundColor Green
}
elseif (($deleteResult.Output -match "angegebene Datei nicht finden") -or ($deleteResult.Output -match "cannot find the file")) {
    Write-Host "[OK] Kein vorhandener Rotation-Task zum Loeschen (normal beim Erstlauf)" -ForegroundColor Green
}
else {
    throw "Rotation-Task Loeschung fehlgeschlagen: $($deleteResult.Output)"
}

$createResult = Invoke-Schtasks -Arguments @(
    "/Create", "/TN", $taskNameNormalized,
    "/SC", "DAILY", "/ST", $AtTime,
    "/TR", $taskCommand, "/RU", $user, "/F"
)
if ($createResult.ExitCode -ne 0) {
    throw "Rotation-Task Erstellung fehlgeschlagen: $($createResult.Output)"
}
Write-Host "[OK] Rotation-Task erstellt" -ForegroundColor Green

$queryResult = Invoke-Schtasks -Arguments @("/Query", "/TN", $taskNameNormalized, "/FO", "LIST", "/V")
if ($queryResult.ExitCode -ne 0) {
    throw "Rotation-Task Abfrage fehlgeschlagen: $($queryResult.Output)"
}
Write-Host "[OK] Rotation-Task verifiziert" -ForegroundColor Green
Write-Host $queryResult.Output -ForegroundColor DarkGray

if ($RunNow) {
    $runResult = Invoke-Schtasks -Arguments @("/Run", "/TN", $taskNameNormalized)
    if ($runResult.ExitCode -ne 0) {
        throw "Rotation-Task Start fehlgeschlagen: $($runResult.Output)"
    }
    Write-Host "[OK] Rotation-Task direkt gestartet" -ForegroundColor Green
}

Write-Host ""
Write-Host "Archive auf dem Laptop:" -ForegroundColor Yellow
Write-Host "  $ProjectDir\\logs\\ops_checks\\archive\\check_logs_YYYY-MM-DD.log"
