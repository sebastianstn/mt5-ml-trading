param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$TaskName = "MT5_Sync_Live_Logs_To_Linux",
    [string]$LinuxUser = "stnsebi",
    [string]$LinuxHost = "192.168.1.4",
    [string]$LinuxLogsDir = "/mnt/1Tb-Data/XGBoost-LightGBM/logs",
    [string]$Symbols = "USDCAD,USDJPY",
    [bool]$RunHidden = $true,
    [switch]$SyncCloses,
    [switch]$RunNow
)

# ============================================================================
# windows_register_live_log_sync_task.ps1
#
# Zweck:
#   Erzeugt aus XML-Template eine konkrete Task-Definition und importiert sie
#   im Windows Task Scheduler (5-Minuten-Intervall).
#
# Läuft auf:
#   Windows 11 Laptop (PowerShell als Administrator empfohlen)
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$templatePath = Join-Path $PSScriptRoot "windows_task_live_log_sync.xml.template"
$syncScriptPath = Join-Path $PSScriptRoot "windows_sync_live_logs.ps1"
$generatedXmlPath = Join-Path $PSScriptRoot "windows_task_live_log_sync.generated.xml"

if (-not (Test-Path $templatePath)) {
    throw "Template nicht gefunden: $templatePath"
}
if (-not (Test-Path $syncScriptPath)) {
    throw "Sync-Skript nicht gefunden: $syncScriptPath"
}

$taskUser = "{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME
$startBoundary = (Get-Date).AddMinutes(1).ToString("yyyy-MM-ddTHH:mm:ss")

$syncArgs = @(
    "-NoProfile",
    "-ExecutionPolicy Bypass",
    "-File `\"$syncScriptPath`\"",
    "-ProjectDir `\"$ProjectDir`\"",
    "-LinuxUser $LinuxUser",
    "-LinuxHost $LinuxHost",
    "-LinuxLogsDir `\"$LinuxLogsDir`\"",
    "-Symbols `\"$Symbols`\""
)
if ($RunHidden) {
    $syncArgs += "-WindowStyle Hidden"
}
if ($SyncCloses) {
    $syncArgs += "-SyncCloses"
}
$xmlHidden = if ($RunHidden) { "true" } else { "false" }
$argString = $syncArgs -join " "

# Taskname für schtasks normalisieren (Root-Task mit führendem Backslash).
$taskNameNormalized = if ($TaskName.StartsWith("\")) { $TaskName } else { "\$TaskName" }

# XML-Template rendern
$xml = Get-Content $templatePath -Raw
$xml = $xml.Replace("{{TASK_NAME}}", $TaskName)
$xml = $xml.Replace("{{TASK_USER}}", $taskUser)
$xml = $xml.Replace("{{START_BOUNDARY}}", $startBoundary)
$xml = $xml.Replace("{{PROJECT_DIR}}", $ProjectDir)
$xml = $xml.Replace("{{POWERSHELL_ARGS}}", $argString)
$xml = $xml.Replace("{{TASK_HIDDEN}}", $xmlHidden)

# UTF-16 schreiben (Task Scheduler XML erwartet meist UTF-16 problemlos)
[System.IO.File]::WriteAllText($generatedXmlPath, $xml, [System.Text.Encoding]::Unicode)

# Bestehende Aufgabe ggf. löschen und neu importieren (mit robuster Fehlerprüfung)
$deleteOutput = schtasks /Delete /TN $taskNameNormalized /F 2>&1
$deleteExit = $LASTEXITCODE
if ($deleteExit -ne 0) {
    $deleteText = ($deleteOutput | Out-String)
    # "Datei nicht gefunden" ist hier ok (Task existierte noch nicht).
    if (($deleteText -notmatch "angegebene Datei nicht finden") -and ($deleteText -notmatch "cannot find the file")) {
        Write-Warning "Vorhandene Task konnte nicht gelöscht werden: $deleteText"
        Write-Warning "Wenn Zugriff verweigert wird: PowerShell als Administrator starten."
    }
}

$createOutput = schtasks /Create /TN $taskNameNormalized /XML $generatedXmlPath /F 2>&1
$createExit = $LASTEXITCODE
if ($createExit -ne 0) {
    $createText = ($createOutput | Out-String)
    throw (
        "Task-Erstellung fehlgeschlagen. Ausgabe: $createText`n" +
        "Hinweis: PowerShell als Administrator starten und erneut ausführen."
    )
}

# Direkt verifizieren, dass der Task wirklich existiert.
$queryOutput = schtasks /Query /TN $taskNameNormalized 2>&1
$queryExit = $LASTEXITCODE
if ($queryExit -ne 0) {
    $queryText = ($queryOutput | Out-String)
    throw "Task wurde erstellt, ist aber nicht auffindbar: $queryText"
}

Write-Host "Task erfolgreich registriert: $taskNameNormalized" -ForegroundColor Green
Write-Host "XML geschrieben: $generatedXmlPath"
Write-Host "User: $taskUser"
Write-Host "Intervall: alle 5 Minuten"

if ($RunNow) {
    $runOutput = schtasks /Run /TN $taskNameNormalized 2>&1
    $runExit = $LASTEXITCODE
    if ($runExit -ne 0) {
        $runText = ($runOutput | Out-String)
        throw "Task konnte nicht gestartet werden: $runText"
    }
    Write-Host "Task wurde direkt gestartet." -ForegroundColor Cyan
}

Write-Host "\nNächster Check auf Linux:" -ForegroundColor Yellow
Write-Host "python scripts/monitor_live_kpis.py --log_dir logs --file_suffix _signals.csv --hours 24 --timeframe M5_TWO_STAGE --export_csv reports/live_kpis_latest.csv"
