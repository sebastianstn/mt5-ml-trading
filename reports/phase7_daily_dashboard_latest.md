# Phase 7 – Daily Dashboard

**Erstellt:** 2026-03-10 20:18
**Fenster:** letzte 24h
**Timeframe:** `M5_TWO_STAGE`
**Log-Quelle:** `/mnt/1Tb-Data/XGBoost-LightGBM/logs/paper_test128`
**Watchdog-Datei:** `/mnt/1Tb-Data/XGBoost-LightGBM/logs/paper_test128/live_log_watchdog_latest.json`
**Gesamtampel:** **WATCH**

## Watchdog-Überblick

- **Datei vorhanden:** Ja
- **Watchdog-Daten geladen:** Ja
- **Watchdog Gesamtstatus:** WATCH
- **Watchdog erstellt (UTC):** 2026-03-10 19:15:23
- **Watchdog Stale-Limit Min:** 75.0
- **Watchdog Lag-Limit Min:** 15.0


| Symbol | Ampel | Watchdog | Sig Fresh | RT Fresh | Letztes Signal (UTC) | Letzter RT-Heartbeat (UTC) | Lag Sig→RT Min | Signale | Closes | TP | SL | Net PnL | PF | DD% | WR% | Ø Dauer Min | Begründung | Watchdog-Reason |
|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| USDCAD | WATCH | WATCH | OK | OK | 2026-03-10 19:15:06+00:00 | 2026-03-10 19:15:00+00:00 | - | 61 | 0 | 0 | 0 | - | - | - | - | - | Signalfluss ok, aber noch keine Close-CSV vorhanden | Signalfluss ok, aber noch keine Close-CSV vorhanden |
| USDJPY | WATCH | OK | OK | OK | 2026-03-10 19:15:09+00:00 | 2026-03-10 19:15:00+00:00 | - | 113 | 1 | 0 | 1 | -363.07 | 0.000 | 0.00 | 0.0 | 0.0 | Erste Closes vorhanden, aber noch geringe Stichprobe | Signal-CSV und Runtime-Heartbeat sind frisch |

## Interpretation

- **OK**: Frische Daten und keine kritischen operativen Auffälligkeiten.
- **WATCH**: Daten sind frisch, aber Aktivität oder Close-KPIs sollten beobachtet werden.
- **INCIDENT**: Stale Logs oder kritischer Drawdown – operativ sofort prüfen.

