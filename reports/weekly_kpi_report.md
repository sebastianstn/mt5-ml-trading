# Wöchentlicher KPI-Report (USDCAD / USDJPY)

**Erstellt:** 2026-03-12 16:17
**Zeitraum (Live-Aktivität):** letzte 7 Tage
**Gesamtstatus:** **UNKLAR**
**Timeframe:** `H1`
**Paper-Gate (12 Wochen):** **PAPER_ONLY**
**Konsekutive GO-Wochen:** 0 / 12

## KPI-Zielwerte

- Profit Factor > 1.2
- Sharpe > 0.8
- Max Drawdown > -10.0%
- Win-Rate > 45.0%
- Mindest-Live-Signale/Woche: 5
- Mindest-Live-Closes/Woche: 5
- **Statistik-Minimum:** 30 abgeschlossene Trades für belastbare KPIs

## Ergebnis je Symbol

| Symbol | Status | Stat.Sign. | WR-CI 95% | Quelle | Live Fresh | Letztes Event (UTC) | Min seit Event | Events | Signale | Closes | Ø Prob | Live PnL | Live PF | Live DD% | Live WR% | Ø Dauer Min | PF (BT) | Sharpe (BT) | Hinweis |
|---|---:|:---:|:---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| USDCAD | DATENBASIS_ZU_KLEIN | ⚠️ (29 fehlen) | 0.0–79.3% | BACKTEST_FALLBACK | OK | 2026-03-12 15:10:16 | 67.3 | 398 | 98 | 1 | 0.465 | -1.76 | 0.000 | 0.00 | 0.0 | 124.0 | 0.000 | -60.756 | Nur 1 Live-Trades (min. 30 für Signifikanz, noch 29 fehlend – weiter Paper-Trading sammeln) |
| USDJPY | DATENBASIS_ZU_KLEIN | ⚠️ (29 fehlen) | 0.0–79.3% | BACKTEST_FALLBACK | OK | 2026-03-12 15:10:08 | 67.4 | 397 | 146 | 1 | 0.464 | -363.07 | 0.000 | 0.00 | 0.0 | 0.0 | 0.000 | 0.000 | Nur 1 Live-Trades (min. 30 für Signifikanz, noch 29 fehlend – weiter Paper-Trading sammeln) |

## 3-Monats Paper-Gate (stabile Werte)

- Letzte 12 Wochen insgesamt erfasst: 7
- Konsekutive GO-Wochen aktuell: 0
- Entscheidungsstatus: **PAPER_ONLY**
- Begründung: Noch 12 GO-Woche(n) ohne Unterbrechung nötig für kontrollierte Eskalation.

## Interpretation

- **Live-Freshness** ist ein hartes Gate: ohne frische Events wird der Status auf NO-GO gesetzt.
- **Live-Signale** messen operative Aktivität und Freshness.
- **Live-Closes** liefern realisierte Paper-/Live-PnL-KPIs und werden bevorzugt, sobald genug Daten vorliegen.
- **Statistische Signifikanz (⚠️/✅):** KPIs sind erst ab 30 abgeschlossenen Trades belastbar. Darunter ist die Win-Rate statistisch zu unsicher für Entscheidungen.
- **WR-CI 95%:** Wilson-Konfidenzintervall für die Win-Rate. Breite Spanne = wenig Daten, enge Spanne = belastbarer Wert.
- **DATENBASIS_ZU_KLEIN:** Status wenn Live-Closes vorhanden aber < 30. Kein GO/NO-GO möglich.
- **Backtest-KPIs** bleiben Fallback, solange noch nicht genug Live-Closes vorhanden sind (`backtest/SYMBOL_trades.csv`).

## Nächste Schritte

1. Bei **NO-GO**: Schwellen, Regime oder Overtrading prüfen und nur im Paper-Modus weiterlaufen lassen.
2. Bei **UNKLAR**: mehr Live-Closes sammeln (Trader weiterlaufen lassen) oder Zeitraum verlängern.
3. Eskalation Richtung Live nur bei **12 konsekutiven GO-Wochen**.

