# Test 128 – Paper-Trade bereit

## 📖 Inhaltsverzeichnis

- [Empfehlung](#empfehlung)
- [Originale Backtest-Parameter](#originale-backtest-parameter)
- [Log-Speicherorte für Test 128](#log-speicherorte-für-test-128)
- [Live-/Paper-Mapping](#live-paper-mapping)
- [Wichtige Einschränkung](#wichtige-einschränkung)
- [Startdatei](#startdatei)
- [Sicherheits-Hinweise](#sicherheits-hinweise)
- [Erwartung](#erwartung)

---

## Empfehlung

**Test 128 (`ZoneC s0.59 h18`)** ist die beste ausgewogene Konfiguration aus dem 50er-Feintuning.

## Originale Backtest-Parameter

## Log-Speicherorte für Test 128

Die Test-128-Startdatei schreibt bewusst in einen separaten Log-Unterordner:

- Windows-Laptop: `C:\Users\Sebastian Setnescu\mt5_trading\logs\paper_test128\`

Wichtige Dateien:

- `USDCAD_signals.csv`
- `USDJPY_signals.csv`
- optional `USDCAD_closes.csv`
- optional `USDJPY_closes.csv`
- `live_trader.log`

Für den Upload auf den Linux-Server wurde zusätzlich vorbereitet:

- `register_test128_log_sync_to_server.bat`

Linux-Zielordner für die Server-Kopie:

- `/mnt/1Tb-Data/XGBoost-LightGBM/logs/paper_test128/`

- `schwelle = 0.59`
- `tp_pct = 0.006`
- `sl_pct = 0.003`
- `RRR = 2.0`
- `cooldown_bars = 12`
- `regime_filter = 0|1|2`
- `atr_faktor = 1.5`
- `horizon = 18`
- Two-Stage: `H1 -> M5`, Version `v4`

## Live-/Paper-Mapping

Folgende Parameter können im aktuellen `live/live_trader.py` direkt gesetzt werden:

- `--schwelle 0.59`
- `--regime_filter 0,1,2`
- `--atr_sl 1`
- `--atr_faktor 1.5`
- `--tp_pct 0.006`
- `--sl_pct 0.003`
- `--two_stage_cooldown_bars 12`
- `--two_stage_enable 1`
- `--two_stage_ltf_timeframe M5`
- `--two_stage_version v4`
- `--paper_trading 1`

## Wichtige Einschränkung

**`horizon = 18` ist im aktuellen Live-Trader nicht als CLI-Argument verfügbar.**

Das bedeutet:

- die Startdatei bildet Test 128 **so nah wie möglich** ab,
- aber **nicht 1:1** vollständig.

## Startdatei

Für Windows 11 Laptop wurde vorbereitet:

- `start_paper_trading_test128.bat`

## Sicherheits-Hinweise

- Nur auf dem **Windows 11 Laptop** ausführen
- MT5 muss laufen und eingeloggt sein
- Erst **Paper-Trading**, kein Echtgeld
- Start-Lot bleibt `0.01`
- Kill-Switch bleibt auf `15%`

## Erwartung

Wenn Test 128 auch im Paper-Betrieb stabil bleibt, ist er aktuell der beste Kandidat für den nächsten kontrollierten Shadow-/Paper-Vergleich gegen die bisherige Top-Konfiguration.
