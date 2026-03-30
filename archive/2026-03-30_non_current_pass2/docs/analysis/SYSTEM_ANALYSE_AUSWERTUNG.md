# 📊 System-Analyse-Auswertung – MT5 ML-Trading-System

**Datum:** 9. März 2026  
**Scope:** aktueller Code- und Startskript-Stand im Repository  
**Operativer Modus:** Demo-Live auf Demo-Konto (`--paper_trading 0`)  
**Operative Symbole:** `USDCAD`, `USDJPY`

---

## 📖 Inhaltsverzeichnis

- [Executive Summary](#executive-summary)
- [1. Verifizierter Ist-Stand](#1-verifizierter-ist-stand-code-basiert)
- [2. Was überholt ist](#2-was-aus-der-alten-auswertung-überholt-ist)
- [3. Was kritisch/offen bleibt](#3-was-weiterhin-kritischoffen-bleibt)
- [4. Kurzbewertung](#4-kurzbewertung-hat-sich-das-system-verbessert)
- [5. Priorisierte nächste Schritte](#5-priorisierte-nächste-schritte-ab-heute)
- [6. Aktueller Gesamtstatus](#6-aktueller-gesamtstatus)
- [7. Die 5 wichtigsten Fragen](#7-die-5-wichtigsten-fragen-ab-jetzt)
- [8. Einordnung der Profitabilität](#8-einordnung-der-profitabilität)
- [9. Konkrete nächste Schritte](#9-konkrete-nächste-schritte)
- [9.2.1 KPI Go-No-Go Matrix (v4, woechentlich)](#921-kpi-go-no-go-matrix-v4-woechentlich)

---

## Executive Summary

Dein System ist gegenüber dem alten Analyse-Stand **klar verbessert**:

- ✅ v5 ist aus dem operativen Startpfad entfernt (v4-only Betrieb)
- ✅ Regime-Filter ist auf Trend-Regime begrenzt (`1,2`)
- ✅ Logging ist erweitert und sauber getrennt (`*_signals.csv`, `*_closes.csv`)
- ✅ PnL-Tracking nutzt echte MT5-Deal/Position-Informationen
- ✅ Dashboard und Deploy-Pfad sind auf das neue CSV-Schema harmonisiert

**Wichtig:** Trotz technischer Stabilisierung ist das System weiterhin **nicht** für Echtgeld freigegeben. Es braucht weiterhin KPI-Nachweis über Zeit (Phase 7 Policy).

---

## 1) Verifizierter Ist-Stand (Code-basiert)

### 1.1 Startmodus / operative Konfiguration

Quelle: `start_paper_trading.bat`

- Beide Fenster starten mit:
  - `--version v4`
  - `--two_stage_version v4`
  - `--regime_filter 1,2`
  - `--paper_trading 0`
  - `--schwelle 0.55 --short_schwelle 0.45`
  - `--two_stage_enable 1 --two_stage_ltf_timeframe M5`

**Interpretation:**

- Kein operativer v5-Startpfad mehr
- Choppy-Regime (0) operativ ausgeschlossen
- Demo-Konto mit echten Demo-Orders (wichtig für echtes PnL-Tracking)

### 1.2 Logging / Monitoring-Daten

Quelle: `live/live_trader.py`

#### Signal-Log

- Dateiname: `*_signals.csv`
- Enthält u. a.:
  - `entry_price`, `sl_price`, `tp_price`
  - `htf_bias`, `ltf_signal`

#### Close-Log

- Dateiname: `*_closes.csv`
- Enthält u. a.:
  - `exit_price`, `pnl_pips`, `pnl_money`
  - `close_grund`, `ticket`

**Interpretation:**

- Keine gemischte CSV-Struktur mehr
- KPI-Auswertung (Signale vs. Exits) ist sauber möglich

### 1.3 Order-/Trade-Lifecycle

Quelle: `live/live_trader.py`

- `order_senden(...)` liefert strukturierte Rückgabe inkl.:
  - `deal_ticket`
  - `position_ticket`
  - `entry_price`, `sl_price`, `tp_price`
- Close-Erkennung nutzt diese IDs (statt späteren Ticket-Ratens)
- `close_grund` wird aus MT5-Deal-Feldern bestimmt
  - Fallback: `UNKNOWN`

**Interpretation:**

- Deutlich robuster als vor den letzten Updates
- Entfernt den früheren Fehlerpfad mit fehleranfälliger nachträglicher Ticketableitung

### 1.4 Dashboard-Kompatibilität

Quelle: `live/mt5/LiveSignalDashboard.mq5`

- `InpFileSuffix` ist auf `"_signals.csv"` gesetzt
- Kommentare/Beispiele sind auf `*_signals.csv` aktualisiert

**Interpretation:**

- Reader/Writer sind konsistent

### 1.5 Deploy-Konsistenz

Quelle: `deploy_to_laptop.sh`

- Überträgt:
  - `live_trader.py`
  - `LiveSignalDashboard.mq5`
  - `two_stage_signal.py`
  - `start_paper_trading.bat`
  - `stop_all_traders.bat`
  - v4-relevante Modellartefakte
- Veraltete, entfernte Startpfade sind bereinigt

**Interpretation:**

- Deployment ist auf aktuellen Betrieb ausgerichtet

---

## 2) Was aus der alten Auswertung überholt ist

Folgende alte Aussagen sind **nicht mehr aktuell**:

1. „Operativ läuft v5/Shadow-Compare“  
   → **Überholt** (operativ v4-only)

2. „Regime 0 wird operativ gehandelt“  
   → **Überholt** (`--regime_filter 1,2` im Startpfad)

3. „Log-Format unvollständig (kein Entry/SL/TP/PnL/HTF/LTF)“  
   → **Überholt** (inkl. getrenntem Close-Log)

4. „Dashboard liest _live_trades.csv“  
   → **Überholt** (`_signals.csv`)

5. „.bat enthält operative Klartext-Zugangsdaten“  
   → Für den aktuellen Startpfad **nicht zutreffend** (ENV-Variablen-Pattern)

---

## 3) Was weiterhin kritisch/offen bleibt

Auch nach den Verbesserungen:

1. **Kein Echtgeld-Go**
   - Policy-konform weiter Demo/Paper

2. **KPI-Nachweis fehlt noch in Laufzeit**
   - Nach den letzten Fixes braucht es neue rollierende KPI-Evidenz

3. **Walk-Forward / OOS-Nachweis weiter Pflicht**
   - Technik ist besser, aber statistischer Nachweis muss folgen

4. **Historische v5-Bewertung bleibt als Kontext relevant**
   - Als Warnsignal für Regressionen, nicht als aktueller Betriebsmodus

---

## 4) Kurzbewertung: Hat sich das System verbessert?

**Ja, eindeutig technisch verbessert.**

Verbessert wurden genau die zuvor kritischen operativen Punkte:

- Betriebsversionen (v4-only)
- Regime-Filter (1,2)
- Logging-Tiefe und Struktur
- PnL-Lifecycle-Integrität
- Dashboard-/Deploy-Konsistenz

**Nicht verbessert (noch offen):** langfristiger Performance-Nachweis und GO-Kriterien über Zeit.

---

## 5) Priorisierte nächste Schritte (ab heute)

### Diese Woche (Phase 7 operativ)

1. **Demo-Live stabil laufen lassen** (ohne Strategie-Wechsel)
2. **Neue Logs sammeln** (`*_signals.csv`, `*_closes.csv`)
3. **Wöchentliche KPI-Auswertung** auf Basis der neuen Struktur:
   - Winrate
   - Profit Factor
   - Drawdown
   - Erwartungswert pro Trade

### Nächste 1–2 Wochen

1. **Walk-Forward / OOS-Check nachziehen**
2. **Threshold-Analyse erneut auf aktuellem Setup**
3. **GO/NO-GO strikt gegen Phase-7-Kriterien**

---

## 6) Aktueller Gesamtstatus

- **Technik-Status:** 🟢 stabilisiert
- **Betriebs-Status:** 🟡 Demo-Live (Kontrollphase)
- **Freigabe Echtgeld:** 🔴 NO-GO (bis KPI-Nachweis stabil erfüllt)

---

## 7) Die 5 wichtigsten Fragen ab jetzt

Die technische Basis ist deutlich besser als zuvor. Ab jetzt ist die zentrale Frage nicht mehr primär, **ob das System läuft**, sondern **ob es robust genug ist, um langfristig profitabel und kontrollierbar zu sein**.

### 7.1 Ist die Performance stabil oder nur phasenweise gut?

Einzelne gute Backtests oder einige starke Tage reichen nicht aus. Entscheidend ist, ob das System:

- über mehrere Zeiträume stabil bleibt,
- in unterschiedlichen Marktphasen tragfähig ist,
- auf `USDCAD` und `USDJPY` konsistent arbeitet,
- und nicht nur auf eine historische Sonderphase passt.

**Kernaussage:** Gesucht wird keine schöne Momentaufnahme, sondern belastbare Wiederholbarkeit.

### 7.2 Sind Backtest, Demo-Live und Monitoring wirklich konsistent?

Die letzten Verbesserungen bei Logging, Dashboard, Deploy und Stop-/Start-Skripten waren wichtig, weil Profitabilität nur dann beurteilbar ist, wenn alle Stufen sauber zusammenpassen:

- Signal-Generierung,
- Order-Ausführung,
- SL/TP-Verhalten,
- Spread/Kosten,
- CSV-/Dashboard-Monitoring,
- Log-Synchronisation.

**Kernaussage:** Wenn Backtest und Demo-Live operativ unterschiedlich sprechen, sind die Ergebnisse nicht entscheidungsreif.

### 7.3 Bringt die zusätzliche System-Komplexität wirklich messbaren Mehrwert?

Das betrifft besonders den Two-Stage-Ansatz. Mehr Komplexität ist nur sinnvoll, wenn sie in der Praxis klar sichtbar hilft, zum Beispiel durch:

- bessere Filterung schlechter Trades,
- geringeren Drawdown,
- stabilere Wochenresultate,
- robustere Regime-Anpassung.

**Kernaussage:** Komplexität ist nur dann gerechtfertigt, wenn sie nicht nur interessanter aussieht, sondern operativ nachweisbar besser ist.

### 7.4 Ist das Risiko robust genug kontrolliert?

Ein technisch funktionierendes System ist noch kein sicheres System. Kritisch bleiben:

- Stop-Loss-Disziplin,
- konservative Positionsgröße,
- realistische Handelskosten,
- saubere Demo-/Paper-Trennung,
- definierte Abbruch- und Eskalationsregeln.

**Kernaussage:** Die relevante Frage ist nicht nur „Kann das System gewinnen?“, sondern auch „Kann es schlechte Phasen kontrolliert überleben?“

### 7.5 Gibt es klare GO/NO-GO-Regeln für die nächsten Entscheidungen?

Ein reifes System braucht feste Regeln für:

- Weiterlauf im Demo-/Paper-Modus,
- Retraining,
- Threshold-Anpassungen,
- Strategie-Stopp,
- eventuelle spätere Echtgeld-Freigabe.

**Kernaussage:** Ohne klare Entscheidungslogik droht operative Bauchsteuerung statt systematischer Weiterentwicklung.

---

## 8) Einordnung der Profitabilität

### 8.1 Aktuelle Bewertung

Die aktuelle Lage spricht dafür, dass das System **ein plausibler Profitabilitäts-Kandidat** ist, aber **noch kein belastbar bewiesenes profitables System**.

Das ist kein Rückschritt, sondern eine saubere professionelle Einordnung:

- Die operative Infrastruktur wurde verbessert.
- Die Datengrundlage für KPI-Auswertung ist besser.
- Die Demo-Live-Bewertung ist aussagekräftiger als zuvor.
- Der Langzeitnachweis fehlt aber noch.

### 8.2 Was hier als echte Profitabilität gelten sollte

Profitabilität bedeutet in diesem Projekt nicht:

- ein paar gute Trades,
- eine einzelne starke Backtest-Periode,
- oder eine kurzfristig schöne Equity-Kurve.

Profitabilität sollte erst dann als belastbar gelten, wenn die Phase-7-Ziele über Zeit nachvollziehbar erfüllt werden, insbesondere:

- **Profit Factor > 1.3**
- **Sharpe Ratio > 0.8**
- **Max Drawdown < 10%**
- **Win-Rate stabil > 50%**
- konsistente Ergebnisse über mehrere Wochen
- möglichst ähnliche Signalqualität auf `USDCAD` und `USDJPY`

### 8.3 Zusammenfassung Profitabilität

**Kurzurteil:**

> Das System wirkt aktuell wie ein ernstzunehmender Kandidat für Profitabilität, aber der statistisch und operativ belastbare Beweis steht noch aus.

Damit bleibt die richtige Haltung aktuell:

- weiter kontrolliert Demo-/Paper-Betrieb,
- keine Echtgeld-Freigabe,
- Entscheidungen nur auf Basis rollierender KPI-Evidenz.

---

## 9) Konkrete nächste Schritte

### 9.1 Sofort / laufend

1. **Demo-/Paper-Betrieb stabil weiterlaufen lassen**
   - keine hektischen Strategie-Wechsel,
   - keine parallelen Experiment-Wechsel im operativen Pfad,
   - Fokus auf saubere Evidenz.

2. **Neue Logs konsequent sammeln und prüfen**
   - `*_signals.csv`
   - `*_closes.csv`
   - Sync-Status zwischen Laptop/MT5/Common Files/Server

3. **Dashboard und Ausführung weiter beobachten**
   - Signal sichtbar?
   - HTF/LTF plausibel?
   - Entries/Closes korrekt?
   - Stale-/Timing-Probleme?

### 9.2 Diese Woche

1. **Wöchentliche KPI-Auswertung standardisieren**
   - Winrate
   - Profit Factor
   - Drawdown
   - Erwartungswert pro Trade
   - Anzahl Trades
   - Regime-spezifische Performance

2. **Backtest-vs-Demo-Live-Abweichungen prüfen**
   - gleiche Signale?
   - ähnliche Entry-Preise?
   - SL/TP konsistent?
   - Schließungsgründe nachvollziehbar?

### 9.2.1 KPI Go-No-Go Matrix (v4, woechentlich)

Die folgende Matrix gilt als operative Entscheidungsbasis für den aktuellen v4-Betrieb im Demo-/Paper-Modus.

| KPI | GO (grün) | WATCH (gelb) | NO-GO (rot) |
| --- | --- | --- | --- |
| Sharpe (7d rollierend) | >= 0.8 | 0.5 bis < 0.8 | < 0.5 |
| Profit Factor | >= 1.3 | 1.1 bis < 1.3 | < 1.1 |
| Max Drawdown (Wochenpeak→Tal) | <= 10% | 10% bis 12% | > 12% |
| Win-Rate | >= 50% | 45% bis < 50% | < 45% |
| Trades pro Woche (pro Symbol) | >= 8 | 5 bis 7 | < 5 |
| Spread-Blockquote | 10% bis 40% | 41% bis 55% | > 55% |
| Kongruenz-Blocks bei HTF neutral | sichtbar + plausibel | unklar | fehlt/inkonsistent |
| Technische Stabilität (Crashs/Exceptions) | 0 | 1 | >= 2 |

**Entscheidungsregel (wöchentlich):**

- **GO-Woche:** kein roter KPI und mindestens 5 grüne KPI.
- **WATCH-Woche:** max. 1 roter KPI oder weniger als 5 grüne KPI.
- **NO-GO-Woche:** 2+ rote KPI oder Drawdown > 12%.

**Promotionsregel (Phase 7):**

- v4 bleibt Paper/Demo, bis 12 GO-Wochen in Folge erreicht sind.
- Bei 2 WATCH-Wochen in Folge: keine Promotion, nur kontrollierte Feinjustierung.
- Bei NO-GO-Woche: Ursachenanalyse + Rollback auf letzte stabile Konfiguration.

**Mindest-Dokumentation pro Woche (pro Symbol):**

1. Trades gesamt, Wins/Losses
2. PnL %, Profit Factor, Sharpe, Max Drawdown
3. Anzahl Log-Events:
   - `SPREAD-FILTER: Trade blockiert`
   - `KONGRUENZ-FILTER ... BLOCKIERT`
   - `Regime=Seitwärts → Schwelle erhöht`
4. Anzahl Exceptions/Neustarts

### 9.3 Nächste 1–2 Wochen

1. **Threshold- und Regime-Analyse auf dem aktuellen Setup nachziehen**
2. **Walk-Forward / OOS-Nachweis weiter schärfen**
3. **GO/NO-GO-Regeln strikt an KPI-Gates koppeln**
4. **Erst danach über Retraining oder neue Varianten entscheiden**

### 9.4 Operative Leitlinie

Die beste nächste Entscheidung ist aktuell nicht „mehr Komplexität“, sondern:

> **mehr belastbare Evidenz bei unverändert kontrolliertem Betrieb.**

---

## Änderungsvermerk

Diese Fassung ersetzt die vorherige, historisch gewachsene Mischversion der Analyse und bildet den aktuellen Repository-Stand zum 09.03.2026 konsistent ab.
