# Phase 7 – Daily Routine (Top-Config Testphase)

Gültig für:

- Operative Symbole: `USDCAD`, `USDJPY`
- Setup: `start_testphase_topconfig_H1_M15.bat` (Paper, Two-Stage v4)

## Morgen-Check (3 Schritte, 5–10 Minuten)

1. **Windows Laptop**
   - MT5 offen + eingeloggt
   - Trader-Fenster laufen (`USDCAD`, `USDJPY`)
   - Falls nicht: `start_testphase_topconfig_H1_M15.bat` starten

2. **Linux Freshness-Check**
   - `python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10`
   - Erwartung: `SYNC_OK`

3. **KPI-Snapshot aktualisieren**
   - `python scripts/monitor_live_kpis.py --log_dir logs --file_suffix _signals.csv --hours 24 --timeframe M5_TWO_STAGE --export_csv reports/live_kpis_latest.csv`
   - `python reports/daily_phase7_dashboard.py --hours 24 --timeframe M5_TWO_STAGE`
   - Optional Wochenampel aktualisieren: `python reports/weekly_kpi_report.py --tage 7 --timeframe M5_TWO_STAGE`

## Abend-Check (3 Schritte, 10 Minuten)

1. **Sync erneut prüfen**
   - `python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10`

2. **Fehlerprüfung Logfile**
   - `tail -n 120 logs/live_trader.log`
   - Fokus: `ERROR`, `Traceback`, dauerhafte API-/MT5-Fehler

3. **Tagesbewertung kurz notieren**
   - Status je Symbol: `OK / WATCH / INCIDENT`
   - Datei: `reports/daily_ops_checklist_w1.md` (Kurznotiz ergänzen)

## 24h / 48h Bewertungs-Checkpoint

- 24h:
  - `python scripts/evaluate_mt5_testphase.py --hours 24 --symbols USDCAD,USDJPY --timeframe M5`
- 48h:
  - `python scripts/evaluate_mt5_testphase.py --hours 48 --symbols USDCAD,USDJPY --timeframe M5`

Ergebnisdatei:

- `reports/testphase/mt5_testphase_eval_latest.csv`
- Optional Wochen-Dashboard: `reports/weekly_kpi_report_M5_TWO_STAGE.md`
- Tages-Dashboard: `reports/phase7_daily_dashboard_latest.csv` und `reports/phase7_daily_dashboard_latest.md`

## Wenn `verify_live_log_sync.py` = `SYNC_NICHT_OK`

1. **Windows Task prüfen**
   - `schtasks /Query /TN "\MT5_Sync_Live_Logs_To_Linux" /V /FO LIST`

2. **Task manuell triggern**
   - `schtasks /Run /TN "\MT5_Sync_Live_Logs_To_Linux"`

3. **Direkt-Sync manuell testen (Windows PowerShell)**
   - `& "C:\Users\Sebastian Setnescu\mt5_trading\scripts\windows_sync_live_logs.ps1" -ProjectDir "C:\Users\Sebastian Setnescu\mt5_trading" -LinuxUser stnsebi -LinuxHost 192.168.1.4 -LinuxLogsDir "/mnt/1Tb-Data/XGBoost-LightGBM/logs" -Symbols "USDCAD,USDJPY" -SyncCloses`

4. **Linux erneut prüfen**
   - `python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10`
