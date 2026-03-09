# Runbook – Auto Log-Sync (Windows → Linux)

**Ziel:** Frische Live-Daten (`*_signals.csv`, optional `*_closes.csv`) automatisiert auf den Linux-Server übertragen, damit Weekly KPI-Gates valide sind.

## 1) Voraussetzung (Windows-Laptop)

- OpenSSH-Client installiert (`ssh`, `scp` verfügbar)
- Zugriff vom Laptop auf Linux-Server funktioniert
- Projekt liegt lokal unter `C:\Users\Sebastian Setnescu\mt5_trading`
- Trader schreibt Logs in `logs/USDCAD_signals.csv` und `logs/USDJPY_signals.csv`

## 2) Manuell testen

PowerShell (Laptop):

- `powershell -ExecutionPolicy Bypass -File .\scripts\windows_sync_live_logs.ps1 -ProjectDir "C:\Users\Sebastian Setnescu\mt5_trading" -LinuxUser stnsebi -LinuxHost 192.168.1.4 -LinuxLogsDir "/mnt/1Tb-Data/XGBoost-LightGBM/logs" -Symbols "USDCAD,USDJPY" -SyncCloses`

Erwartung:

- Meldung `Live-Logs synchronisiert ...`
- Auf Linux werden `logs/USDCAD_signals.csv` und `logs/USDJPY_signals.csv` aktualisiert

## 3) Geplante Aufgabe anlegen (Task Scheduler)

### 3A) 1-Klick (empfohlen)

PowerShell als Administrator auf dem Laptop:

- `powershell -ExecutionPolicy Bypass -File .\scripts\windows_register_live_log_sync_task.ps1 -ProjectDir "C:\Users\Sebastian Setnescu\mt5_trading" -LinuxUser stnsebi -LinuxHost 192.168.1.4 -LinuxLogsDir "/mnt/1Tb-Data/XGBoost-LightGBM/logs" -Symbols "USDCAD,USDJPY" -SyncCloses -RunNow`

Erzeugt/aktualisiert Task:

- `MT5_Sync_Live_Logs_To_Linux`
- Intervall: alle 5 Minuten
- Aktion: Aufruf von `windows_sync_live_logs.ps1`

### 3B) Manuell (Fallback)

Empfehlung:

- Trigger: alle **5 Minuten**, unbegrenzt wiederholen
- Aktion: `powershell.exe`
- Argumente:
  - `-ExecutionPolicy Bypass -File "C:\Users\Sebastian Setnescu\mt5_trading\scripts\windows_sync_live_logs.ps1" -ProjectDir "C:\Users\Sebastian Setnescu\mt5_trading" -LinuxUser stnsebi -LinuxHost 192.168.1.4 -LinuxLogsDir "/mnt/1Tb-Data/XGBoost-LightGBM/logs" -Symbols "USDCAD,USDJPY" -SyncCloses`
- Starten in:
  - `C:\Users\Sebastian Setnescu\mt5_trading`
- Option: „Aufgabe so schnell wie möglich nach einem verpassten Start ausführen“ aktivieren

## 4) Linux-Verifikation

- `python scripts/monitor_live_kpis.py --log_dir logs --file_suffix _signals.csv --hours 24 --timeframe M5_TWO_STAGE --export_csv reports/live_kpis_latest.csv`
- `python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10`

Erwartung:

- `status = OK` (nicht `STALE`) für USDCAD/USDJPY
- `minutes_since_last` deutlich unter Stale-Grenze
- Verifikations-Helper meldet am Ende `SYNC_OK`

## 5) Gate-Erzwingung

`reports/weekly_kpi_report.py` bewertet ohne frische Live-Events automatisch als **NO-GO**.

Das bedeutet:

- Kein frischer Datenfluss ⇒ kein GO
- Erst mit laufendem Sync + frischen Events sind Weekly-Gates aussagekräftig
