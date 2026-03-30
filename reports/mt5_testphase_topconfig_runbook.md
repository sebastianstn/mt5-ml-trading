# Runbook – MT5 Testphase mit neuer Top-Konfiguration (2026-03-08)

## 📖 Inhaltsverzeichnis

- [Ziel](#ziel)
- [1. Vor dem Start](#1-vor-dem-start-windows-laptop)
- [2. Testphase starten](#2-testphase-starten-windows-laptop)
- [3. Während der Laufzeit überwachen](#3-während-der-laufzeit-überwachen)
- [4. Log-Sync sicherstellen](#4-log-sync-richtung-linux-sicherstellen)
- [5. Nach 24–48h auswerten](#5-nach-2448h-auswerten-linux-server)
- [6. Entscheidungslogik GO/WATCH/NO_GO](#6-entscheidungslogik-go--watch--no_go)
- [7. Nächste Roadmap-Aufgabe](#7-nächste-roadmap-aufgabe)

---

## Ziel

Die im Live-kompatiblen Sweep beste Konfiguration kontrolliert auf dem **Windows 11 MT5-Host** im **Paper-Modus** laufen lassen und danach auf dem **Linux-Server** objektiv auswerten.

Top-Konfiguration aus Sweep:

- `schwelle = 0.54`
- `regime_filter = 0,1,2`
- `two_stage_enable = 1`
- `two_stage_ltf_timeframe = M5`
- `two_stage_version = v4`
- `atr_sl = 1`, `atr_faktor = 1.5`

## 1) Vor dem Start (Windows Laptop)

- MT5 Terminal ist geöffnet und mit Demo/Paper-Konto verbunden.
- AutoTrading im MT5 geprüft.
- Projektpfad vorhanden: `C:\Users\Sebastian Setnescu\mt5_trading`
- Modelle vorhanden:
  - `models/lgbm_htf_bias_usdcad_H1_v4.pkl`
  - `models/lgbm_ltf_entry_usdcad_M5_v4.pkl`
  - `models/lgbm_htf_bias_usdjpy_H1_v4.pkl`
  - `models/lgbm_ltf_entry_usdjpy_M5_v4.pkl`
- Zugangsdaten als Umgebungsvariablen gesetzt:
  - `MT5_SERVER`
  - `MT5_LOGIN`
  - `MT5_PASSWORD`

## 2) Testphase starten (Windows Laptop)

Empfohlenes Startskript:

- `start_testphase_topconfig_H1_M15.bat`

Dieses Skript startet beide Trader parallel (USDCAD + USDJPY) im Paper-Modus mit der Top-Konfiguration.

## 3) Während der Laufzeit überwachen

- Dashboard darf nicht dauerhaft auf `MISSING` oder `STALE` stehen.
- `logs/*_signals.csv` muss laufend neue Zeilen bekommen.
- Optional: `logs/*_closes.csv` auf Trade-Schließungen prüfen.

## 4) Log-Sync Richtung Linux sicherstellen

Windows:

- Task Scheduler Task `MT5_Sync_Live_Logs_To_Linux` muss laufen.

Linux:

- `python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10`

Erwartung: `SYNC_OK`

## 5) Nach 24–48h auswerten (Linux Server)

Primäre Auswertung:

- `python scripts/evaluate_mt5_testphase.py --hours 48 --symbols USDCAD,USDJPY --timeframe M5`

Artefakte:

- `reports/testphase/mt5_testphase_eval_<timestamp>.csv`
- `reports/testphase/mt5_testphase_eval_latest.csv`

## 6) Entscheidungslogik (GO / WATCH / NO_GO)

Das Skript bewertet pro Symbol:

- **Freshness** der Signale (stale = NO_GO)
- Mindest-Aktivität bei Signalen
- Bei genügend Close-Events zusätzlich:
  - Gewinnfaktor (PF)
  - Win-Rate
  - Max Drawdown (%)

Statusbedeutung:

- `GO`: alle KPI-Gates erfüllt
- `WATCH`: noch zu wenige Close-Events (weiter laufen lassen)
- `NO_GO`: Freshness-/Aktivitäts-/KPI-Problem

## 7) Nächste Roadmap-Aufgabe

Nach der ersten 48h-Auswertung:

1. Ergebnisse in Wochenlog festhalten.
2. Bei `GO/WATCH`: weitere 5 Tage laufen lassen und erneut bewerten.
3. Bei `NO_GO`: nur einen Parameter ändern (z. B. Schwelle 0.55) und erneut 48h testen.
