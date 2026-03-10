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
    $scriptDir = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptDir)) {
        $scriptDir = Split-Path -Parent $PSCommandPath
    }
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

    # PowerShell-Argumente zusammenbauen
    $psArgs = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$syncScript`" -SourceDir `"$SourceDir`" -Continuous -IntervalSec $IntervalSec"

    if ($RunHidden) {
        # VBS-Wrapper generieren: startet PowerShell komplett unsichtbar (kein Fenster)
        $vbsDir = Split-Path $syncScript
        $vbsPath = Join-Path $vbsDir "mt5_sync_launcher.vbs"
        $escapedArgs = $psArgs.Replace('"', '""')
        $vbsContent = @"
' Auto-generiert von install_sync_task.ps1 – nicht manuell bearbeiten
' Startet PowerShell komplett unsichtbar (Run ..., 0 = SW_HIDE)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "powershell.exe $escapedArgs", 0, False
"@
        Set-Content -Path $vbsPath -Value $vbsContent -Encoding ASCII
        Write-Host "[TASK] VBS-Launcher erstellt: $vbsPath" -ForegroundColor DarkGray

        # Task-Action: wscript.exe fuehrt VBS aus (kein sichtbares Fenster)
        $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbsPath`""
    }
    else {
        # Ohne Hidden: PowerShell direkt starten (Fenster sichtbar)
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $psArgs
    }
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
        -Force -ErrorAction Stop | Out-Null

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
