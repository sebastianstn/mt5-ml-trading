# 📊 System-Analyse-Auswertung – MT5 ML-Trading-System

**Datum:** 8. März 2026  
**Scope:** aktueller Code- und Startskript-Stand im Repository  
**Operativer Modus:** Demo-Live auf Demo-Konto (`--paper_trading 0`)  
**Operative Symbole:** `USDCAD`, `USDJPY`

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

## Änderungsvermerk

Diese Fassung ersetzt die vorherige, historisch gewachsene Mischversion der Analyse und bildet den aktuellen Repository-Stand zum 08.03.2026 konsistent ab.
