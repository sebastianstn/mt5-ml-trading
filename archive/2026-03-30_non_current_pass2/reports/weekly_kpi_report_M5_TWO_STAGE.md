# Wöchentlicher KPI-Report (USDCAD / USDJPY)

**Erstellt:** 2026-03-08 23:14
**Zeitraum (Live-Aktivität):** letzte 7 Tage
**Gesamtstatus:** **NO-GO**
**Timeframe:** `M5_TWO_STAGE`
**Paper-Gate (12 Wochen):** **PAPER_ONLY**
**Konsekutive GO-Wochen:** 0 / 12

## KPI-Zielwerte

- Profit Factor > 1.2
- Sharpe > 0.8
- Max Drawdown > -10.0%
- Win-Rate > 45.0%
- Mindest-Live-Signale/Woche: 5

## Ergebnis je Symbol

| Symbol | Status | Live Fresh | Letztes Event (UTC) | Min seit Event | Events | Live-Signale | Long% | Short% | Ø Prob | Return% (BT) | PF (BT) | Sharpe (BT) | MaxDD% (BT) | WinRate% (BT) | Hinweis |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| USDCAD | NO-GO | STALE | - | - | 0 | 0 | 0.0 | 0.0 | 0.000 | +3.32 | 1.762 | 3.736 | -0.70 | 58.5 | Keine frischen Live-Daten (stale/fehlend) |
| USDJPY | NO-GO | STALE | - | - | 0 | 0 | 0.0 | 0.0 | 0.000 | +16.44 | 4.273 | 10.324 | -0.75 | 76.4 | Keine frischen Live-Daten (stale/fehlend) |

## 3-Monats Paper-Gate (stabile Werte)

- Letzte 12 Wochen insgesamt erfasst: 6
- Konsekutive GO-Wochen aktuell: 0
- Entscheidungsstatus: **PAPER_ONLY**
- Begründung: Noch 12 GO-Woche(n) ohne Unterbrechung nötig für kontrollierte Eskalation.

## Interpretation

- **Live-Freshness** ist ein hartes Gate: ohne frische Events wird der Status auf NO-GO gesetzt.
- **Live-Signale** zeigen operative Aktivität; bei zu wenigen Signalen ist die Bewertung statistisch schwach.
- **Profitabilitäts-KPIs** stammen aus den letzten Backtest-Trades (`backtest/SYMBOL_M5_two_stage_trades.csv`).
- Sobald ein echter Live-PnL-Export verfügbar ist, sollte die Profitabilitätssektion auf Live-Daten umgestellt werden.

## Nächste Schritte

1. Bei **NO-GO**: Regime-Filter/Schwelle prüfen und nur im Paper-Modus weiterlaufen lassen.
2. Bei **UNKLAR**: Zeitraum verlängern (z.B. 14/30 Tage) oder Signalzahl erhöhen.
3. Eskalation Richtung Live nur bei **12 konsekutiven GO-Wochen**.

