param(
	[string]$ProjectDir = "",
	[string]$LocalLogsDir = "",
	[string]$Symbols = "USDCAD,USDJPY",
	[string]$Timeframe = "M5_TWO_STAGE",
	[double]$StaleFactor = 1.5,
	[double]$MaxAgeMinutes = 0,
	[double]$MaxCsvRuntimeLagMinutes = 0,
	[string]$OutputJsonName = "live_log_watchdog_latest.json",
	[string]$OutputCsvName = "live_log_watchdog_latest.csv"
)

# ============================================================================
# windows_live_log_watchdog.ps1
#
# Zweck:
#   Prüft auf dem Windows-Laptop die lokale Trader-Telemetrie auf Frische,
#   Runtime-Heartbeat und Drift zwischen live_trader.log und *_signals.csv.
#
# Ergebnis:
#   - schreibt JSON + CSV in den lokalen Log-Ordner
#   - Exit-Code 0 bei Gesamtstatus OK, sonst 1
#
# Läuft auf:
#   Windows 11 Laptop (MT5-Host)
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
	$ProjectDir = Split-Path -Parent $PSScriptRoot
}

function Get-TimeframeMinutes {
	param([Parameter(Mandatory = $true)][string]$Name)

	switch ($Name.ToUpper()) {
		"H1" { return 60.0 }
		"M30" { return 30.0 }
		"M15" { return 15.0 }
			# Two-Stage läuft im M15-Loop (HTF auf H1 gecached), deshalb
			# muss die Watchdog-Frische auf 15 Minuten basieren, nicht auf 5.
			"M5_TWO_STAGE" { return 15.0 }
		default { return 60.0 }
	}
}

function ConvertTo-UtcTimestamp {
	param([Parameter(Mandatory = $true)][string]$RawValue)

	if ([string]::IsNullOrWhiteSpace($RawValue)) {
		return $null
	}

	try {
		$parsed = [datetime]::ParseExact(
			$RawValue.Trim(),
			"yyyy-MM-dd HH:mm:ss",
			[System.Globalization.CultureInfo]::InvariantCulture,
			(
				[System.Globalization.DateTimeStyles]::AssumeUniversal -bor
				[System.Globalization.DateTimeStyles]::AdjustToUniversal
			)
		)
		return $parsed
	}
	catch {
		return $null
	}
}

function Get-LastCsvTimestampUtc {
	param([Parameter(Mandatory = $true)][string]$FilePath)

	if (-not (Test-Path $FilePath)) {
		return $null
	}

	$lastTimestamp = $null
	foreach ($row in Import-Csv -Path $FilePath) {
		$rawTime = [string]$row.time
		if ([string]::IsNullOrWhiteSpace($rawTime)) {
			continue
		}

		$parsed = ConvertTo-UtcTimestamp -RawValue $rawTime
		if ($null -ne $parsed) {
			$lastTimestamp = $parsed
		}
	}

	return $lastTimestamp
}

function Get-LastRuntimeHeartbeatUtc {
	param(
		[Parameter(Mandatory = $true)][string]$LogPath,
		[Parameter(Mandatory = $true)][string]$Symbol
	)

	if (-not (Test-Path $LogPath)) {
		return $null
	}

	$pattern = "\[(?<symbol>[A-Z]{6})\]\s+Neue\s+[A-Z0-9_]+-Kerze\s+\|\s+(?<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+UTC"
	$lastHeartbeat = $null

	$fileStream = $null
	$streamReader = $null
	try {
		$fileStream = [System.IO.File]::Open(
			$LogPath,
			[System.IO.FileMode]::Open,
			[System.IO.FileAccess]::Read,
			[System.IO.FileShare]::ReadWrite
		)
		$streamReader = [System.IO.StreamReader]::new($fileStream)

		while (($line = $streamReader.ReadLine()) -ne $null) {
			$match = [regex]::Match($line, $pattern)
			if (-not $match.Success) {
				continue
			}
			if ($match.Groups["symbol"].Value -ne $Symbol.ToUpper()) {
				continue
			}

			$parsed = ConvertTo-UtcTimestamp -RawValue $match.Groups["ts"].Value
			if ($null -ne $parsed) {
				$lastHeartbeat = $parsed
			}
		}
	}
	finally {
		if ($null -ne $streamReader) {
			$streamReader.Dispose()
		}
		elseif ($null -ne $fileStream) {
			$fileStream.Dispose()
		}
	}

	return $lastHeartbeat
}

function Get-AgeMinutes {
	param([Parameter(Mandatory = $false)][datetime]$Timestamp)

	if ($null -eq $Timestamp) {
		return $null
	}

	$nowUtc = [datetime]::UtcNow
	return [math]::Round(($nowUtc - $Timestamp).TotalMinutes, 1)
}

function Get-FileAgeMinutes {
	param([Parameter(Mandatory = $true)][string]$FilePath)

	if (-not (Test-Path $FilePath)) {
		return $null
	}

	$item = Get-Item -Path $FilePath
	$age = [datetime]::UtcNow - $item.LastWriteTimeUtc
	return [math]::Round($age.TotalMinutes, 1)
}

function Normalize-RuntimeTimestampUtc {
	param([Parameter(Mandatory = $false)][datetime]$Timestamp)

	if ($null -eq $Timestamp) {
		return $null
	}

	$normalized = $Timestamp
	$futureLimit = [datetime]::UtcNow.AddMinutes(2)
	$shiftHours = 0

	while ($normalized -gt $futureLimit -and $shiftHours -lt 14) {
		$normalized = $normalized.AddHours(-1)
		$shiftHours += 1
	}

	return $normalized
}

if ([string]::IsNullOrWhiteSpace($LocalLogsDir)) {
	$localLogsDir = Join-Path $ProjectDir "logs"
}
else {
	$localLogsDir = $LocalLogsDir
}

if (-not (Test-Path $localLogsDir)) {
	throw "Lokaler Log-Ordner nicht gefunden: $localLogsDir"
}

$symbolList = @($Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" })
if ($symbolList.Count -eq 0) {
	throw "Keine Symbole angegeben. Beispiel: -Symbols 'USDCAD,USDJPY'"
}

$timeframeMinutes = Get-TimeframeMinutes -Name $Timeframe
$staleLimitMinutes = if ($MaxAgeMinutes -gt 0) { $MaxAgeMinutes } else { [math]::Round($timeframeMinutes * $StaleFactor, 1) }
$lagLimitMinutes = if ($MaxCsvRuntimeLagMinutes -gt 0) { $MaxCsvRuntimeLagMinutes } else { [math]::Max([math]::Round($timeframeMinutes * 3.0, 1), 10.0) }
$runtimeLogPath = Join-Path $localLogsDir "live_trader.log"

$results = @()
foreach ($symbol in $symbolList) {
	$signalPath = Join-Path $localLogsDir ("{0}_signals.csv" -f $symbol)
	$closePath = Join-Path $localLogsDir ("{0}_closes.csv" -f $symbol)

	$signalTs = Get-LastCsvTimestampUtc -FilePath $signalPath
	$runtimeTs = Normalize-RuntimeTimestampUtc -Timestamp (
		Get-LastRuntimeHeartbeatUtc -LogPath $runtimeLogPath -Symbol $symbol
	)

	$signalAge = Get-AgeMinutes -Timestamp $signalTs
	$runtimeAge = Get-AgeMinutes -Timestamp $runtimeTs
	$signalFileAge = Get-FileAgeMinutes -FilePath $signalPath

	$csvRuntimeLag = $null
	if ($null -ne $signalTs -and $null -ne $runtimeTs -and $runtimeTs -gt $signalTs) {
		$csvRuntimeLag = [math]::Round(($runtimeTs - $signalTs).TotalMinutes, 1)
	}

	$status = "OK"
	$reason = "Signal-CSV und Runtime-Heartbeat sind frisch"

	if ($null -eq $signalTs -and $null -eq $runtimeTs) {
		$status = "INCIDENT"
		$reason = "Weder Signal-CSV noch Runtime-Heartbeat gefunden"
	}
	elseif ($null -eq $signalTs) {
		$status = "INCIDENT"
		$reason = "Signal-CSV fehlt oder enthält keine gültigen Zeitstempel"
	}
	elseif ($null -eq $runtimeTs) {
		$status = "INCIDENT"
		$reason = "Runtime-Heartbeat fehlt im live_trader.log"
	}
	elseif ($runtimeAge -gt $staleLimitMinutes) {
		$status = "INCIDENT"
		$reason = "Runtime-Heartbeat stale"
	}
	elseif (
		($signalAge -gt $staleLimitMinutes) -and
		($signalFileAge -gt $staleLimitMinutes)
	) {
		$status = "INCIDENT"
		$reason = "Signal-CSV stale"
	}
	elseif ($null -ne $csvRuntimeLag -and $csvRuntimeLag -gt $lagLimitMinutes) {
		$status = "INCIDENT"
		$reason = "Signal-CSV hinkt hinter Runtime-Heartbeat"
	}
	elseif ((Test-Path $closePath) -eq $false) {
		$status = "WATCH"
		$reason = "Signalfluss ok, aber noch keine Close-CSV vorhanden"
	}

	$results += [PSCustomObject]@{
		symbol = $symbol
		status = $status
		reason = $reason
		signal_file = $signalPath
		signal_exists = (Test-Path $signalPath)
		signal_last_event_utc = if ($null -ne $signalTs) { $signalTs.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
		signal_age_min = $signalAge
		signal_file_age_min = $signalFileAge
		runtime_log = $runtimeLogPath
		runtime_exists = (Test-Path $runtimeLogPath)
		runtime_last_heartbeat_utc = if ($null -ne $runtimeTs) { $runtimeTs.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
		runtime_age_min = $runtimeAge
		csv_runtime_lag_min = $csvRuntimeLag
		closes_exists = (Test-Path $closePath)
	}
}

$overallStatus = "OK"
if (@($results | Where-Object { $_.status -eq "INCIDENT" }).Count -gt 0) {
	$overallStatus = "INCIDENT"
}
elseif (@($results | Where-Object { $_.status -eq "WATCH" }).Count -gt 0) {
	$overallStatus = "WATCH"
}

$jsonPath = Join-Path $localLogsDir $OutputJsonName
$csvPath = Join-Path $localLogsDir $OutputCsvName

$payload = [PSCustomObject]@{
	generated_at_utc = [datetime]::UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
	local_logs_dir = $localLogsDir
	timeframe = $Timeframe
	stale_limit_minutes = $staleLimitMinutes
	lag_limit_minutes = $lagLimitMinutes
	overall_status = $overallStatus
	symbols = $results
}

$payload | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding UTF8
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "WINDOWS LIVE LOG WATCHDOG" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ("Logs: {0}" -f $localLogsDir)
Write-Host ("Timeframe: {0} | Stale-Limit: {1} Min | Lag-Limit: {2} Min" -f $Timeframe, $staleLimitMinutes, $lagLimitMinutes)
$results | Format-Table symbol, status, signal_age_min, runtime_age_min, csv_runtime_lag_min, reason -AutoSize
Write-Host ("JSON: {0}" -f $jsonPath)
Write-Host ("CSV : {0}" -f $csvPath)

if ($overallStatus -eq "OK") {
	exit 0
}

exit 1
