param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$LinuxUser = "stnsebi",
    [string]$LinuxHost = "192.168.1.4",
    [string]$LinuxDataDir = "/mnt/1T-Data/XGBoost-LightGBM/data",
    [int]$Bars = 50000
)

# Export + Upload für M15, M30, M60 (M60 = H1 in MT5, aber getrennte CSV-Namen)
# Ausführung: Windows 11 Laptop (MT5-Host)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$pythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$dataLoader = Join-Path $ProjectDir "data_loader.py"
$envFile = Join-Path $ProjectDir ".env"

if (-not (Test-Path $pythonExe)) {
    throw "Python nicht gefunden: $pythonExe"
}
if (-not (Test-Path $dataLoader)) {
    throw "data_loader.py nicht gefunden: $dataLoader"
}
if (-not (Test-Path $envFile)) {
    throw ".env nicht gefunden: $envFile`nBitte MT5_LOGIN, MT5_PASSWORD und MT5_SERVER in .env setzen."
}

# .env auf Pflichtschlüssel prüfen (einfacher Textcheck)
$envText = Get-Content $envFile -Raw
foreach ($requiredKey in @("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")) {
    if ($envText -notmatch "(?m)^\s*$requiredKey\s*=") {
        throw "Fehlender Schlüssel in .env: $requiredKey"
    }
}

Write-Host "[1/3] Exportiere Rohdaten (M15, M30, M60) ..." -ForegroundColor Cyan

Push-Location $ProjectDir
try {
    & $pythonExe $dataLoader --timeframe M15 --bars $Bars
    if ($LASTEXITCODE -ne 0) { throw "Export fehlgeschlagen für M15 (ExitCode=$LASTEXITCODE)" }

    & $pythonExe $dataLoader --timeframe M30 --bars $Bars
    if ($LASTEXITCODE -ne 0) { throw "Export fehlgeschlagen für M30 (ExitCode=$LASTEXITCODE)" }

    & $pythonExe $dataLoader --timeframe M60 --bars $Bars
    if ($LASTEXITCODE -ne 0) { throw "Export fehlgeschlagen für M60 (ExitCode=$LASTEXITCODE)" }
}
finally {
    Pop-Location
}

$dataDirWin = Join-Path $ProjectDir "data"
$tfSuffix = @("M15", "M30", "M60")
$symbols = @("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")

$files = @()
foreach ($sym in $symbols) {
    foreach ($tf in $tfSuffix) {
        $file = Join-Path $dataDirWin ("{0}_{1}.csv" -f $sym, $tf)
        if (-not (Test-Path $file)) {
            throw "Fehlende Datei nach Export: $file"
        }
        $files += $file
    }
}

Write-Host "[2/3] Lade CSVs auf Linux-Server hoch ..." -ForegroundColor Cyan
$target = "{0}@{1}:{2}" -f $LinuxUser, $LinuxHost, $LinuxDataDir
& scp @files $target

Write-Host "[3/3] Fertig. 21 Dateien übertragen (7 Symbole × 3 Timeframes)." -ForegroundColor Green
Write-Host "Tipp: Auf Linux prüfen mit:" -ForegroundColor Yellow
Write-Host "  .venv/bin/python scripts/verify_raw_data_copy.py --timeframes M15 M30 M60"
