param(
    [string]$ProjectDir = "",
    [string]$CheckBatName = "check_logs_now.bat",
    [string]$OutputSubDir = "logs\\ops_checks"
)

# =============================================================================
# windows_run_check_logs_once.ps1
#
# Zweck:
#   Fuehrt check_logs_now.bat einmal aus (ohne Pause) und schreibt
#   eine nachvollziehbare Historie inkl. Exit-Code.
#
# Exit-Codes von check_logs_now.bat:
#   0 = OK
#   1 = FEHLER
#   2 = WARNUNG
# =============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Split-Path -Parent $PSScriptRoot
}

$checkBatPath = Join-Path $ProjectDir $CheckBatName
if (-not (Test-Path $checkBatPath)) {
    throw "check_logs_now.bat nicht gefunden: $checkBatPath"
}

$outputDir = Join-Path $ProjectDir $OutputSubDir
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$historyPath = Join-Path $outputDir "check_logs_history.log"
$lastRunPath = Join-Path $outputDir "check_logs_last_run.log"

$timestampUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")

# CMD nutzen, damit Batch inkl. Argumenten robust ausgefuehrt wird.
$argLine = "/c ""$checkBatPath" --no-pause""

$output = & cmd.exe $argLine 2>&1
$exitCode = $LASTEXITCODE

$status = switch ($exitCode) {
    0 { "OK" }
    1 { "FEHLER" }
    2 { "WARNUNG" }
    default { "UNBEKANNT" }
}

$header = @(
    "=======================================================================",
    "CHECK_LOGS AUTO RUN | $timestampUtc | STATUS=$status | EXIT=$exitCode",
    "======================================================================="
)

$lines = @()
$lines += $header
$lines += ($output | ForEach-Object { "$_" })
$lines += ""

# Letzter Lauf als einzelne Datei (gut fuer schnellen Blick)
$lines | Set-Content -Path $lastRunPath -Encoding UTF8

# Historie fortschreiben
$lines | Add-Content -Path $historyPath -Encoding UTF8

# Optional konsolenfreundliche Kurzzeile
Write-Host "[AUTO-CHECK] STATUS=$status EXIT=$exitCode | $timestampUtc"

exit $exitCode
