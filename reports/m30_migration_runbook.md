# M30/M15-Migration Runbook (H1 → M30 → M15)

Status: M30 + M15 Code-Integration umgesetzt, operative KPI-Gates in Phase 7.

## Ziel

Das bestehende H1-System wird auf M30 und M15 erweitert, um mehr Signale pro Woche zu
erhalten, ohne Look-Ahead-Bias einzuführen und ohne den laufenden Paper-Betrieb zu gefährden.

## Gerätezuordnung

- **Windows 11 Laptop (MT5-Host):** `data_loader.py`, `live/live_trader.py`
- **Linux-Server:** Feature Engineering, Labeling, Training, Walk-Forward, Backtest

## Migration in 2 Stufen

1. **Stufe A (erledigt):** M30 parallel aufgebaut und integriert.
2. **Stufe B (erledigt):** M15 technisch integriert; Freigabe nur via KPI-Gate.

## Umgesetzte Code-Bausteine

- `data_loader.py`: Zeitrahmen-Parameter `H1 | M30 | M15`
- `features/feature_engineering.py`: Zeitrahmen-Parameter, zeitäquivalente Returns/Volatilität
- `features/labeling.py`: Zeitrahmen-Parameter `H1 | M30 | M15`
- `train_model.py`: Zeitrahmen `H1 | M30 | M15 | H4`, timeframe-spezifische Modellpfade
- `walk_forward.py`: Zeitrahmen `H1 | M30 | M15`
- `backtest/backtest.py`: Zeitrahmen `H1 | M30 | M15 | H4`, timeframe-spezifische Modell-/Datenpfade
- `live/live_trader.py`: Zeitrahmen `H1 | M30 | M15`, MT5-Mapping und timeframe-Modellpfade
- `reports/weekly_kpi_report.py`: timeframe-fähige KPI-Reports (`--timeframe M15`)

## Wichtige Sicherheitsregeln

- Test-Set bleibt heilig (nur finale Bewertung).
- Keine zufällige Zeitreihen-Aufteilung (`shuffle=True` bleibt tabu).
- Stop-Loss bleibt Pflicht, auch auf M30.
- Operative Policy bleibt aktiv: nur `USDCAD` und `USDJPY` im Paper-Betrieb.

## KPI-Gate für M30/M15-Freigabe

M30/M15 werden nur aktiv gefahren, wenn Backtest + Walk-Forward mindestens zeigen:

- Sharpe > 1.0
- Gewinnfaktor > 1.3
- Max Drawdown > -20%
- stabile Walk-Forward-Fenster ohne starken Einbruch

Zusätzlich in Phase 7 (Paper-Betrieb):

- mindestens 12 konsekutive GO-Wochen im Wochenreport
- operative Policy bleibt: nur `USDCAD` und `USDJPY`

## Nächste konkrete Aufgabe

M15-Kette ausführen und bewerten:

1. Windows Laptop: `data_loader.py --timeframe M15`
2. Linux Server: Feature/Label/Training/Walk-Forward/Backtest mit `--timeframe M15`
3. KPI-Report: `weekly_kpi_report.py --timeframe M15 --tage 7`
4. Nur bei erfüllten Gates in Paper aktivieren (`live_trader.py --timeframe M15`)
