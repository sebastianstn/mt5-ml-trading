# Phase 6 → 7: 90-Tage Paper-Trading Plan

**Stand:** 2026-03-01  
**Roadmap-Bezug:** `Roadmap.md` (Phase 6 abgeschlossen, Phase 7 aktiv: Monitoring & KPI-Gates)

---

## Ziel dieses Plans

Dieser Plan stellt sicher, dass das System **operativ stabil** läuft, bevor Echtgeld aktiviert wird.

- **Gerät Windows 11 Laptop:** `live/live_trader.py` ausführen (MT5 + Paper-Trading)
- **Gerät Linux-Server:** Auswertung, Reports, Modellpflege

**Operativer Einstieg:** Für die ersten 4 Wochen bitte zwingend
`reports/paper_trading_4w_execution_plan.md` verwenden und wöchentlich abhaken.

**Bereits umgesetzt (KW09 Start):**

- Sync-Task + Dashboard-Datenfluss stabilisiert
- Incident-Tracking gestartet (`reports/incidents_2026w09.md`)
- Daily Ops Checklist angelegt (`reports/daily_ops_checklist_w1.md`)

---

## Aktive Setups (Betriebs-Policy)

1. **USDCAD H1**: `--schwelle 0.60 --regime_filter 1,2`
2. **USDJPY H1**: `--schwelle 0.60 --regime_filter 1`

> Hinweis: Alle anderen Paare bleiben **Research-only** und werden nicht live/paper operativ eskaliert,
> bis eine explizite Freigabe nach separatem Nachweis erfolgt.

---

## Harte Schutzregeln (No-Exception)

1. **Nur Paper-Modus** (kein Echtgeld)
2. **Stop-Loss immer aktiv**
3. **Kill-Switch aktiv**: `--kill_switch_dd 0.15`
4. Bei Prozessabsturz: Neustart + Incident-Log

---

## KPI-Definition (Go/No-Go)

### Primäre KPIs (über 90 Tage)

- Profit Factor $> 1.20$
- Sharpe Ratio $> 0.80$
- Max Drawdown $< 10\%$ (Paper)
- Technische Uptime $\ge 98\%$

### Sekundäre KPIs

- Keine Serie von > 3 Tagen ohne Log-Updates
- Keine ungeklärten Exceptions im Live-Log
- Signal-/Trade-Frequenz im erwartbaren Rahmen (kein „flatline bot“)

---

## Zeitplan

## Woche 1 (Stabilität)

- [ ] Prozesse täglich prüfen (läuft der Bot durch?)
- [ ] Log-Rotation / Dateigröße prüfen
- [ ] Erste Incident-Liste erstellen (falls Fehler)
- [ ] Baseline-Export: Anzahl Signale, Trades, DD

**Gate Woche 1:** Keine kritischen technischen Fehler.

## Woche 2–4 (Verlässlichkeit)

- [ ] Wöchentliche KPI-Auswertung
- [ ] Uptime messen
- [ ] Incident-Rate reduzieren
- [ ] Konsistenz zwischen Symbolen prüfen

**Gate Monat 1:** Stabiler Betrieb + keine Kill-Switch-Events.

## Monat 2 (Robustheit)

- [ ] KPI-Trends wöchentlich dokumentieren
- [ ] Auffällige Marktphasen markieren (News, Volatilität)
- [ ] Reality-Check auf Trade-Qualität (Stichproben)

**Gate Monat 2:** KPIs bleiben in Zielnähe, keine strukturellen Ausfälle.

## Monat 3 (Go/No-Go)

- [ ] Finale 90-Tage-Auswertung erstellen
- [ ] KPI gegen Schwellwerte vergleichen
- [ ] Entscheidung dokumentieren: Go / No-Go

**Gate Monat 3:** Alle primären KPIs erfüllt.

---

## Tägliche Kurz-Checkliste (Windows Laptop)

- [ ] MT5 offen und verbunden
- [ ] Trader-Prozess aktiv
- [ ] Neue Einträge in `logs/*_live_trades.csv`
- [ ] Keine kritischen Errors in `logs/live_trader.log`
- [ ] Drawdown-Status unauffällig

Praktische Umsetzung siehe `reports/daily_ops_checklist_w1.md`.

---

## Wöchentliche Auswertung (Linux Server)

- [ ] Trades der Woche pro Symbol zusammenfassen
- [ ] Win-Rate, PF, Sharpe (rolling) berechnen
- [ ] Max DD aktualisieren
- [ ] Incident-Report pflegen (Root-Cause + Fix)

---

## Go/No-Go-Entscheidungsvorlage (nach 90 Tagen)

### GO (nur wenn alles erfüllt)

- [ ] PF $> 1.20$
- [ ] Sharpe $> 0.80$
- [ ] Max DD $< 10\%$
- [ ] Uptime $\ge 98\%$
- [ ] Keine kritischen offenen Incidents
- [ ] Mindestens **12 konsekutive Wochen** mit Gesamtstatus `GO`

### NO-GO (wenn eins verletzt)

- [ ] Paper-Trading verlängern (weitere 30 Tage)
- [ ] Ursachenanalyse dokumentieren
- [ ] Konfiguration/Regime-Filter nachschärfen
- [ ] GO-Serienzähler zurücksetzen (Stabilität neu aufbauen)

---

## Nächster Roadmap-Schritt nach diesem Plan

Wenn die 90 Tage erfolgreich sind, folgt direkt:

1. Phase 7 Monitoring automatisieren (Alerts + Tagesreport)
2. Monatliches Retraining weiter betreiben
3. Erst dann kontrolliert Echtgeld-Freigabe diskutieren
