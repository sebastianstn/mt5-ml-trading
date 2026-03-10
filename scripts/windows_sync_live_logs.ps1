param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$LocalLogsDir = "",
    [string]$LinuxUser = "sebastian",
    [string]$LinuxHost = "192.168.1.35",
    [string]$LinuxLogsDir = "/mnt/1Tb-Data/XGBoost-LightGBM/logs",
    [string]$Symbols = "USDCAD,USDJPY",
    [switch]$SyncCloses,
    [string]$WatchdogTimeframe = "M5_TWO_STAGE",
    [double]$WatchdogStaleFactor = 1.5,
    [double]$WatchdogMaxLagMinutes = 0
)

# ============================================================================
# windows_sync_live_logs.ps1
#
# Zweck:
#   Synchronisiert die aktuellen Live-Trader CSV-Logs vom Windows-Laptop
#   (MT5-Host) auf den Linux-Server für Monitoring & Weekly KPI-Gates.
#
# Läuft auf:
#   Windows 11 Laptop (per geplanter Aufgabe, z. B. alle 5 Minuten)
#
# Uploads:
#   <LocalLogsDir>/SYMBOL_signals.csv  -> LinuxLogsDir/
#   optional <LocalLogsDir>/SYMBOL_closes.csv -> LinuxLogsDir/ (mit -SyncCloses)
#   <LocalLogsDir>/live_trader.log -> LinuxLogsDir/
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$signalsSuffix = "_signals.csv"
$closesSuffix = "_closes.csv"
$runtimeLogName = "live_trader.log"
$watchdogJsonName = "live_log_watchdog_latest.json"
$watchdogCsvName = "live_log_watchdog_latest.csv"
$watchdogScriptPath = Join-Path $PSScriptRoot "windows_live_log_watchdog.ps1"
if ([string]::IsNullOrWhiteSpace($LocalLogsDir)) {
    $localLogsDir = Join-Path $ProjectDir "logs"
}
else {
    $localLogsDir = $LocalLogsDir
}

if (-not (Test-Path $localLogsDir)) {
    throw "Lokaler Log-Ordner nicht gefunden: $localLogsDir"
}

$symbolList = $Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" }
if ($symbolList.Count -eq 0) {
    throw "Keine Symbole angegeben. Beispiel: -Symbols 'USDCAD,USDJPY'"
}

function Write-FallbackWatchdogArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason,
        [Parameter(Mandatory = $true)]
        [string]$OverallStatus
    )

    $fallbackRows = @()
    foreach ($sym in $symbolList) {
        $signalPath = Join-Path $localLogsDir ("{0}{1}" -f $sym, $signalsSuffix)
        $closePath = Join-Path $localLogsDir ("{0}{1}" -f $sym, $closesSuffix)
        $fallbackRows += [PSCustomObject]@{
            symbol = $sym
            status = $OverallStatus
            reason = $Reason
            signal_file = $signalPath
            signal_exists = (Test-Path $signalPath)
            signal_last_event_utc = $null
            signal_age_min = $null
            signal_file_age_min = $null
            runtime_log = (Join-Path $localLogsDir $runtimeLogName)
            runtime_exists = (Test-Path (Join-Path $localLogsDir $runtimeLogName))
            runtime_last_heartbeat_utc = $null
            runtime_age_min = $null
            csv_runtime_lag_min = $null
            closes_exists = (Test-Path $closePath)
        }
    }

    $fallbackPayload = [PSCustomObject]@{
        generated_at_utc = [datetime]::UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
        local_logs_dir = $localLogsDir
        timeframe = $WatchdogTimeframe
        stale_limit_minutes = $null
        lag_limit_minutes = $(if ($WatchdogMaxLagMinutes -gt 0) { $WatchdogMaxLagMinutes } else { $null })
        overall_status = $OverallStatus
        symbols = $fallbackRows
    }

    $fallbackJsonPath = Join-Path $localLogsDir $watchdogJsonName
    $fallbackCsvPath = Join-Path $localLogsDir $watchdogCsvName

    $fallbackPayload | ConvertTo-Json -Depth 6 | Set-Content -Path $fallbackJsonPath -Encoding UTF8
    $fallbackRows | Export-Csv -Path $fallbackCsvPath -NoTypeInformation -Encoding UTF8
}

# SSH/SCP Optionen fuer robusten, nicht-interaktiven Task-Betrieb
$sshOpts = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new"
)

# Zielpfad vorbereiten (einmalig auf Linux)
$target = "{0}@{1}" -f $LinuxUser, $LinuxHost
& ssh @sshOpts $target "mkdir -p '$LinuxLogsDir'"
if ($LASTEXITCODE -ne 0) {
    throw "SSH-Verbindung fehlgeschlagen (User/Host/Key prüfen): $target"
}

# Lokalen Watchdog ausführen (best effort): schreibt JSON/CSV für Sync + spätere Diagnose.
$watchdogExitCode = $null
if (Test-Path $watchdogScriptPath) {
    Write-Host "[INFO] Starte lokalen Live-Log-Watchdog ..." -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $watchdogScriptPath `
        -ProjectDir $ProjectDir `
        -LocalLogsDir $localLogsDir `
        -Symbols $Symbols `
        -Timeframe $WatchdogTimeframe `
        -StaleFactor $WatchdogStaleFactor `
        -MaxCsvRuntimeLagMinutes $WatchdogMaxLagMinutes

    $watchdogExitCode = $LASTEXITCODE
    if ($watchdogExitCode -ne 0) {
        Write-Warning "Watchdog meldet WATCH/INCIDENT. Artefakte werden trotzdem synchronisiert."
    }
}
else {
    Write-Warning "Watchdog-Skript nicht gefunden: $watchdogScriptPath"
}

$watchdogJsonPath = Join-Path $localLogsDir $watchdogJsonName
$watchdogCsvPath = Join-Path $localLogsDir $watchdogCsvName
if ((-not (Test-Path $watchdogJsonPath)) -or (-not (Test-Path $watchdogCsvPath))) {
    $fallbackReason = "Watchdog wurde nicht gestartet"
    if (-not (Test-Path $watchdogScriptPath)) {
        $fallbackReason = "Watchdog-Skript fehlt auf dem Laptop"
    }
    elseif ($null -ne $watchdogExitCode) {
        $fallbackReason = "Watchdog-Artefakte fehlen nach Lauf (ExitCode=$watchdogExitCode)"
    }

    Write-Warning ($fallbackReason + " - schreibe Ersatz-Artefakte fuer den Linux-Sync.")
    Write-FallbackWatchdogArtifacts -Reason $fallbackReason -OverallStatus "INCIDENT"
}

# Zu transferierende Dateien sammeln
$filesToSync = @()
foreach ($sym in $symbolList) {
    $sig = Join-Path $localLogsDir ("{0}{1}" -f $sym, $signalsSuffix)
    if (Test-Path $sig) {
        $filesToSync += $sig
    }

    if ($SyncCloses) {
        $cls = Join-Path $localLogsDir ("{0}{1}" -f $sym, $closesSuffix)
        if (Test-Path $cls) {
            $filesToSync += $cls
        }
    }
}

if ($filesToSync.Count -eq 0) {
    $runtimeLogPath = Join-Path $localLogsDir $runtimeLogName
    if (Test-Path $runtimeLogPath) {
        $filesToSync += $runtimeLogPath
    }
}

if ($filesToSync.Count -eq 0) {
    Write-Warning "Keine synchronisierbaren Log-Dateien gefunden (signals/closes/live_trader.log)."
    exit 0
}

# Laufzeit-Log immer zusätzlich mitnehmen, wenn vorhanden.
$runtimeLogPath = Join-Path $localLogsDir $runtimeLogName
if ((Test-Path $runtimeLogPath) -and (-not ($filesToSync -contains $runtimeLogPath))) {
    $filesToSync += $runtimeLogPath
}

# Watchdog-Artefakte immer mitsenden, wenn vorhanden.
if ((Test-Path $watchdogJsonPath) -and (-not ($filesToSync -contains $watchdogJsonPath))) {
    $filesToSync += $watchdogJsonPath
}

if ((Test-Path $watchdogCsvPath) -and (-not ($filesToSync -contains $watchdogCsvPath))) {
    $filesToSync += $watchdogCsvPath
}

# Upload ausführen
$destination = "{0}@{1}:{2}" -f $LinuxUser, $LinuxHost, $LinuxLogsDir
& scp @sshOpts @filesToSync $destination
if ($LASTEXITCODE -ne 0) {
    throw "SCP-Upload fehlgeschlagen nach: $destination"
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[$ts] Live-Logs synchronisiert: $($filesToSync.Count) Datei(en) -> $destination" -ForegroundColor Green
if ($null -ne $watchdogExitCode) {
    Write-Host "[$ts] Watchdog-ExitCode: $watchdogExitCode" -ForegroundColor Yellow
}
