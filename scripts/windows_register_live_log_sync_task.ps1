param(
    [string]$ProjectDir = "C:\Users\Sebastian Setnescu\mt5_trading",
    [string]$LocalLogsDir = "",
    [string]$TaskName = "MT5_Sync_Live_Logs_To_Linux",
    [string]$LinuxUser = "sebastian",
    [string]$LinuxHost = "192.168.1.35",
    [string]$LinuxLogsDir = "/mnt/1Tb-Data/XGBoost-LightGBM/logs",
    [string]$Symbols = "USDCAD,USDJPY",
    [string]$WatchdogTimeframe = "M5_TWO_STAGE",
    [double]$WatchdogStaleFactor = 1.5,
    [double]$WatchdogMaxLagMinutes = 0,
    [switch]$RunHidden,
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

function Invoke-Schtasks {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    # Argumente mit Leerzeichen muessen in Anfuehrungszeichen stehen,
    # weil Start-Process -ArgumentList [string[]] sie nur mit Leerzeichen verbindet.
    $argLine = ($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
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

function Write-StepResult {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode,
        [string]$Output = ""
    )

    if ($ExitCode -eq 0) {
        Write-Host ("[OK] {0}" -f $Step) -ForegroundColor Green
    }
    else {
        Write-Host ("[FEHLER] {0} (ExitCode={1})" -f $Step, $ExitCode) -ForegroundColor Red
    }

    if (-not [string]::IsNullOrWhiteSpace($Output)) {
        Write-Host $Output -ForegroundColor DarkGray
    }
}

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
    "-WindowStyle Hidden",
    "-ExecutionPolicy Bypass",
    ('-File "{0}"' -f $syncScriptPath),
    ('-ProjectDir "{0}"' -f $ProjectDir),
    "-LinuxUser $LinuxUser",
    "-LinuxHost $LinuxHost",
    ('-LinuxLogsDir "{0}"' -f $LinuxLogsDir),
    ('-Symbols "{0}"' -f $Symbols),
    ('-WatchdogTimeframe "{0}"' -f $WatchdogTimeframe),
    ('-WatchdogStaleFactor {0}' -f $WatchdogStaleFactor)
)
if ($WatchdogMaxLagMinutes -gt 0) {
    $syncArgs += ('-WatchdogMaxLagMinutes {0}' -f $WatchdogMaxLagMinutes)
}
if (-not [string]::IsNullOrWhiteSpace($LocalLogsDir)) {
    $syncArgs += ('-LocalLogsDir "{0}"' -f $LocalLogsDir)
}
if ($SyncCloses) {
    $syncArgs += "-SyncCloses"
}
$xmlHidden = if ($RunHidden) { "true" } else { "false" }
$argString = $syncArgs -join " "

# VBS-Wrapper generieren fuer komplett unsichtbare Ausfuehrung
if ($RunHidden) {
    $vbsPath = Join-Path $PSScriptRoot "linux_sync_launcher.vbs"
    $escapedArgs = $argString.Replace('"', '""')
    $vbsLines = @(
        "' Auto-generiert von windows_register_live_log_sync_task.ps1",
        "' Startet PowerShell komplett unsichtbar (Run ..., 0 = SW_HIDE)",
        'Set WshShell = CreateObject("WScript.Shell")',
        ('WshShell.Run "powershell.exe {0}", 0, True' -f $escapedArgs)
    )
    Set-Content -Path $vbsPath -Value $vbsLines -Encoding ASCII
    Write-Host "VBS-Launcher erstellt: $vbsPath" -ForegroundColor DarkGray
}

# Taskname für schtasks normalisieren (Root-Task mit führendem Backslash).
$taskNameNormalized = if ($TaskName.StartsWith("\")) { $TaskName } else { "\$TaskName" }

# XML-Template rendern
$xml = Get-Content $templatePath -Raw
$xml = $xml.Replace("{{TASK_NAME}}", $TaskName)
$xml = $xml.Replace("{{TASK_USER}}", $taskUser)
$xml = $xml.Replace("{{START_BOUNDARY}}", $startBoundary)
$xml = $xml.Replace("{{PROJECT_DIR}}", $ProjectDir)
$xml = $xml.Replace("{{TASK_HIDDEN}}", $xmlHidden)

# Exec-Action: bei RunHidden wscript.exe + VBS, sonst powershell.exe direkt
if ($RunHidden) {
    $vbsPath = Join-Path $PSScriptRoot "linux_sync_launcher.vbs"
    $xml = $xml.Replace("{{EXEC_COMMAND}}", "wscript.exe")
    $xml = $xml.Replace("{{EXEC_ARGS}}", "`"$vbsPath`"")
}
else {
    $xml = $xml.Replace("{{EXEC_COMMAND}}", "powershell.exe")
    $xml = $xml.Replace("{{EXEC_ARGS}}", $argString)
}

# UTF-16 schreiben (Task Scheduler XML erwartet meist UTF-16 problemlos)
[System.IO.File]::WriteAllText($generatedXmlPath, $xml, [System.Text.Encoding]::Unicode)

Write-Host "Task-Diagnose gestartet..." -ForegroundColor Cyan
Write-Host "TaskName: $taskNameNormalized"
Write-Host "XML: $generatedXmlPath"

# Bestehende Aufgabe ggf. löschen und neu importieren (mit robuster Fehlerprüfung)
$deleteResult = Invoke-Schtasks -Arguments @("/Delete", "/TN", $taskNameNormalized, "/F")
Write-StepResult -Step "Task löschen (falls vorhanden)" -ExitCode $deleteResult.ExitCode -Output $deleteResult.Output
if ($deleteResult.ExitCode -ne 0) {
    $deleteText = $deleteResult.Output
    # "Datei nicht gefunden" ist hier ok (Task existierte noch nicht).
    if (($deleteText -notmatch "angegebene Datei nicht finden") -and ($deleteText -notmatch "cannot find the file")) {
        Write-Warning "Vorhandene Task konnte nicht gelöscht werden: $deleteText"
        Write-Warning "Wenn Zugriff verweigert wird: PowerShell als Administrator starten."
    }
}

$createResult = Invoke-Schtasks -Arguments @("/Create", "/TN", $taskNameNormalized, "/XML", $generatedXmlPath, "/F")
Write-StepResult -Step "Task erstellen" -ExitCode $createResult.ExitCode -Output $createResult.Output
if ($createResult.ExitCode -ne 0) {
    $createText = $createResult.Output
    throw (
        "Task-Erstellung fehlgeschlagen. Ausgabe: $createText`n" +
        "Hinweis: PowerShell als Administrator starten und erneut ausführen."
    )
}

# Direkt verifizieren, dass der Task wirklich existiert.
$queryResult = Invoke-Schtasks -Arguments @("/Query", "/TN", $taskNameNormalized)
Write-StepResult -Step "Task abfragen" -ExitCode $queryResult.ExitCode -Output $queryResult.Output
if ($queryResult.ExitCode -ne 0) {
    $queryText = $queryResult.Output
    throw "Task wurde erstellt, ist aber nicht auffindbar: $queryText"
}

Write-Host "Task erfolgreich registriert: $taskNameNormalized" -ForegroundColor Green
Write-Host "XML geschrieben: $generatedXmlPath"
Write-Host "User: $taskUser"
Write-Host "Intervall: alle 5 Minuten"
if (-not [string]::IsNullOrWhiteSpace($LocalLogsDir)) {
    Write-Host "Lokaler Log-Ordner: $LocalLogsDir"
}
Write-Host "Watchdog-Timeframe: $WatchdogTimeframe"
Write-Host "Watchdog-StaleFactor: $WatchdogStaleFactor"
if ($WatchdogMaxLagMinutes -gt 0) {
    Write-Host "Watchdog-MaxLagMinutes: $WatchdogMaxLagMinutes"
}
Write-Host "Linux-Zielordner: $LinuxLogsDir"

if ($RunNow) {
    $runResult = Invoke-Schtasks -Arguments @("/Run", "/TN", $taskNameNormalized)
    Write-StepResult -Step "Task direkt starten" -ExitCode $runResult.ExitCode -Output $runResult.Output
    if ($runResult.ExitCode -ne 0) {
        $runText = $runResult.Output
        throw "Task konnte nicht gestartet werden: $runText"
    }
    Write-Host "Task wurde direkt gestartet." -ForegroundColor Cyan
}

Write-Host "\nNächster Check auf Linux:" -ForegroundColor Yellow
Write-Host ("python scripts/monitor_live_kpis.py --log_dir {0} --file_suffix _signals.csv --hours 24 --timeframe M5_TWO_STAGE --export_csv reports/live_kpis_latest.csv" -f $LinuxLogsDir)
