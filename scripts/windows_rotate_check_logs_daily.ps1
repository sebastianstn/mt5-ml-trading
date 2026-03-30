param(
    [string]$ProjectDir = "",
    [string]$OutputSubDir = "logs\\ops_checks"
)

# =============================================================================
# windows_rotate_check_logs_daily.ps1
#
# Zweck:
#   Rotiert die Auto-Check-History taeglich in eine Datumsdatei und leert danach
#   die laufende History-Datei fuer den naechsten Tag.
# =============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Split-Path -Parent $PSScriptRoot
}

$outputDir = Join-Path $ProjectDir $OutputSubDir
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$historyPath = Join-Path $outputDir "check_logs_history.log"
$archiveDir = Join-Path $outputDir "archive"
if (-not (Test-Path $archiveDir)) {
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
}

if (-not (Test-Path $historyPath)) {
    "" | Set-Content -Path $historyPath -Encoding UTF8
    Write-Host "[OK] History-Datei neu angelegt (war nicht vorhanden)."
    exit 0
}

$historyContent = Get-Content -Path $historyPath -Raw -ErrorAction Stop
if ([string]::IsNullOrWhiteSpace($historyContent)) {
    Write-Host "[OK] History-Datei ist leer, nichts zu rotieren."
    exit 0
}

$stamp = (Get-Date).ToString("yyyy-MM-dd")
$archivePath = Join-Path $archiveDir ("check_logs_{0}.log" -f $stamp)

if (Test-Path $archivePath) {
    Add-Content -Path $archivePath -Value "" -Encoding UTF8
    Add-Content -Path $archivePath -Value "--- APPEND FROM DAILY ROTATION ---" -Encoding UTF8
    Add-Content -Path $archivePath -Value $historyContent -Encoding UTF8
}
else {
    Set-Content -Path $archivePath -Value $historyContent -Encoding UTF8
}

"" | Set-Content -Path $historyPath -Encoding UTF8
Write-Host "[OK] Rotation abgeschlossen: $archivePath"
exit 0
