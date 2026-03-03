# Wöchentlicher KPI-Report (USDCAD / USDJPY)

**Erstellt:** 2026-03-03 12:07
**Zeitraum (Live-Aktivität):** letzte 7 Tage
**Gesamtstatus:** **NO-GO**
**Timeframe:** `H1`
**Paper-Gate (12 Wochen):** **PAPER_ONLY**
**Konsekutive GO-Wochen:** 0 / 12

## KPI-Zielwerte

- Profit Factor > 1.2
- Sharpe > 0.8
- Max Drawdown > -10.0%
- Win-Rate > 45.0%
- Mindest-Live-Signale/Woche: 5

## Ergebnis je Symbol

| Symbol | Status | Live-Signale | Long% | Short% | Ø Prob | Return% (BT) | PF (BT) | Sharpe (BT) | MaxDD% (BT) | WinRate% (BT) | Hinweis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| USDCAD | GO | 6 | 16.7 | 0.0 | 0.118 | +2.95 | 1.355 | 2.111 | -1.18 | 58.4 | Alle Ziel-KPIs erfüllt |
| USDJPY | NO-GO | 5 | 0.0 | 20.0 | 0.306 | +1.33 | 1.130 | 0.820 | -1.85 | 55.5 | KPI unter Ziel: PF |

## 3-Monats Paper-Gate (stabile Werte)

- Letzte 12 Wochen insgesamt erfasst: 3
- Konsekutive GO-Wochen aktuell: 0
- Entscheidungsstatus: **PAPER_ONLY**
- Begründung: Noch 12 GO-Woche(n) ohne Unterbrechung nötig für kontrollierte Eskalation.

## Interpretation

- **Live-Signale** zeigen operative Aktivität; bei zu wenigen Signalen ist die Bewertung statistisch schwach.
- **Profitabilitäts-KPIs** stammen aus den letzten Backtest-Trades (`backtest/SYMBOL_trades.csv`).
- Sobald ein echter Live-PnL-Export verfügbar ist, sollte die Profitabilitätssektion auf Live-Daten umgestellt werden.

## Nächste Schritte

1. Bei **NO-GO**: Regime-Filter/Schwelle prüfen und nur im Paper-Modus weiterlaufen lassen.
2. Bei **UNKLAR**: Zeitraum verlängern (z.B. 14/30 Tage) oder Signalzahl erhöhen.
3. Eskalation Richtung Live nur bei **12 konsekutiven GO-Wochen**.

