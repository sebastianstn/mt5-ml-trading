param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$LocalLogsDir = "",
    [string]$LinuxUser = "sebastian",
    [string]$LinuxHost = "192.168.1.35",
    [string]$LinuxLogsDir = "/mnt/1Tb-Data/XGBoost-LightGBM/logs",
    [string]$Symbols = "USDCAD,USDJPY",
    [switch]$SyncCloses
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

# Upload ausführen
$destination = "{0}@{1}:{2}" -f $LinuxUser, $LinuxHost, $LinuxLogsDir
& scp @sshOpts @filesToSync $destination
if ($LASTEXITCODE -ne 0) {
    throw "SCP-Upload fehlgeschlagen nach: $destination"
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[$ts] Live-Logs synchronisiert: $($filesToSync.Count) Datei(en) -> $destination" -ForegroundColor Green
