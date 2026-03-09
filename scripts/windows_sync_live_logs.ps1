param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$LinuxUser = "stnsebi",
    [string]$LinuxHost = "192.168.1.4",
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
#   logs/SYMBOL_signals.csv  -> Linux logs/
#   optional logs/SYMBOL_closes.csv -> Linux logs/ (mit -SyncCloses)
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$signalsSuffix = "_signals.csv"
$closesSuffix = "_closes.csv"
$localLogsDir = Join-Path $ProjectDir "logs"

if (-not (Test-Path $localLogsDir)) {
    throw "Lokaler Log-Ordner nicht gefunden: $localLogsDir"
}

$symbolList = $Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" }
if ($symbolList.Count -eq 0) {
    throw "Keine Symbole angegeben. Beispiel: -Symbols 'USDCAD,USDJPY'"
}

# Zielpfad vorbereiten (einmalig auf Linux)
$target = "{0}@{1}" -f $LinuxUser, $LinuxHost
ssh $target "mkdir -p '$LinuxLogsDir'"

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
    Write-Warning "Keine synchronisierbaren Log-Dateien gefunden (signals/closes)."
    exit 0
}

# Upload ausführen
$destination = "{0}@{1}:{2}" -f $LinuxUser, $LinuxHost, $LinuxLogsDir
scp @filesToSync $destination

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[$ts] Live-Logs synchronisiert: $($filesToSync.Count) Datei(en) -> $destination" -ForegroundColor Green
