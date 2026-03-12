# Archiv: Two-Stage v5 (H1/M5 Two-Stage Experiment)

**Archiviert:** 2026-03-12
**Grund:** Stress-Test (spread_faktor 2.0) fehlgeschlagen – NO-GO

## Ergebnis des Stress-Tests (2026-03-05)

| Symbol | Sharpe | Profit Factor | Status |
|--------|--------|---------------|--------|
| USDCAD | -23.002 | 0.011 | ❌ NO-GO |
| USDJPY | -2.140 | 0.692 | ❌ NO-GO |

## Inhalt

- `scripts/train_two_stage.py` – Training HTF-Bias + LTF-Entry Modelle
- `scripts/two_stage_backtest.py` – Backtest-Engine für Two-Stage
- `models/` – Trainierte Two-Stage Modelle (HTF H1 + LTF M5) für USDCAD und USDJPY
- `data/` – M5 Labeled Data (v5, ATR-Barrieren)
- `backtest/` – Backtest-Ergebnisse und v4 vs v5 Vergleich

## Reaktivierung

Falls das Two-Stage Konzept erneut evaluiert werden soll:
1. Dateien zurück in die Hauptverzeichnisse verschieben
2. Stress-Test mit mindestens spread_faktor 1.5 bestehen
3. Walk-Forward auf neuen Daten durchführen
4. Minimum 4 Wochen Shadow-Compare vs. v4 Baseline

**Aktueller Produktionsstand:** v4 (Single-Stage H1) bleibt operativ.
