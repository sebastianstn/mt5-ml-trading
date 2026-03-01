param(
    [string]$TaskName = "MT5_Sync_Live_Logs",
    [string]$SourceDir = "C:\Users\Sebastian Setnescu\mt5_trading\logs",
    [int]$IntervalSec = 5,
    [switch]$RunHidden,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ScriptPath {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $syncPath = Join-Path $scriptDir "sync_live_logs_to_mt5_common.ps1"
    if (-not (Test-Path $syncPath)) {
        throw "Sync-Skript nicht gefunden: $syncPath"
    }
    return $syncPath
}

function Ensure-SourceDir([string]$PathValue) {
    if (-not (Test-Path $PathValue)) {
        throw "SourceDir nicht gefunden: $PathValue"
    }
}

try {
    Ensure-SourceDir -PathValue $SourceDir
    $syncScript = Resolve-ScriptPath

    if ($IntervalSec -lt 1) {
        $IntervalSec = 1
    }

    $windowArg = ""
    if ($RunHidden) {
        $windowArg = "-WindowStyle Hidden "
    }

    $args = "-NoProfile {0}-ExecutionPolicy Bypass -File `"{1}`" -SourceDir `"{2}`" -Continuous -IntervalSec {3}" -f $windowArg, $syncScript, $SourceDir, $IntervalSec

    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $args
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0)

    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

    if ($Force) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    }

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description "Synchronisiert MT5 Live CSV Logs nach Common Files (USDCAD/USDJPY)" `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null

    Write-Host "[TASK] Aufgabe registriert: $TaskName" -ForegroundColor Green
    Write-Host "[TASK] Quelle: $SourceDir"
    Write-Host "[TASK] Sync-Skript: $syncScript"
    Write-Host "[TASK] Start: Beim Login"
    Write-Host "[TASK] Teststart jetzt mit: Start-ScheduledTask -TaskName '$TaskName'"
}
catch {
    Write-Host ("[TASK] FEHLER: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
