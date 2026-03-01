# 4-Wochen Ausführungsplan (Start: 2026-03-01)

**Ziel:** Die Phase-6/7-Übergangsarbeit in den ersten 4 Wochen klar strukturieren,
objektiv abhaken und ohne Aktionismus auswerten.

**Scope:** Nur aktive Paare `USDCAD`, `USDJPY` (Paper-Modus).

**Wichtig:** In diesen 4 Wochen primär **operative Stabilität** absichern.
Strategisches Feintuning nur bei klar dokumentierter Ursache.

---

## Woche 1 — Stabilität & Datenfluss (2026-03-01 bis 2026-03-07)

### Status-Update (Stand: 2026-03-01)

- ✅ USDCAD und USDJPY Trader wurden erfolgreich im Paper-Modus gestartet
- ✅ Task Scheduler Sync läuft automatisch mit `LastTaskResult = 0`
- ✅ CSV-Dateien werden in `logs/` erzeugt und in MT5 `Common\Files` gespiegelt
- ✅ Heartbeat-Logging aktiv (`--heartbeat_log 1`) für beide Symbole
- ✅ `Common\Files` aktualisiert sich mit frischen Zeitstempeln (USDCAD/USDJPY)
- ✅ Erster Daily-Check durchgeführt (MT5, Logs, Task, Dateifluss)
- ✅ Incident-Log initialisiert (`reports/incidents_2026w09.md`)
- ⏳ Wochenkriterien (6/7 Tage, Daily Checks, Incident-Doku) laufen noch

### Aufgaben Woche 1

- [x] Beide Trader-Prozesse stabil gestartet (USDCAD, USDJPY)
- [x] MT5 Dashboard zeigt Daten für beide Symbole (kein permanentes `MISSING`)
- [x] Sync-Task läuft automatisch (Task Scheduler `LastTaskResult = 0`)
- [x] Daily Check durchgeführt (Logs, Prozesse, MT5-Verbindung)
- [x] Incident-Log initialisiert (`reports/incidents_2026w09.md`)

### Done-Definition (Woche 1)

- [ ] An mindestens **6/7 Tagen** neue CSV-Updates in `logs/*_live_trades.csv`
- [ ] Keine kritischen Prozessabbrüche ohne dokumentierten Fix
- [x] Datenkette funktioniert durchgehend: `live_trader.py -> logs -> Common Files -> MT5 EA`

---

## Woche 2 — Monitoring-Routine (2026-03-08 bis 2026-03-14)

### Aufgaben Woche 2

- [ ] Wöchentlichen KPI-Report erzeugen (`reports/weekly_kpi_report.py`)
- [ ] KPI-Historie fortschreiben (`reports/weekly_kpi_history.csv`)
- [ ] Daily Betriebs-Checklist konsequent durchführen
- [ ] Regime-Verteilung in Live-Signalen stichprobenartig prüfen
- [ ] 1x Recovery-Test (Trader-Neustart + Sync-Neustart) dokumentieren

### Done-Definition (Woche 2)

- [ ] KPI-Report für Woche 2 liegt vor
- [ ] Recovery-Test erfolgreich innerhalb < 10 Minuten
- [ ] Kein unbemerkter Ausfall > 1 Stunde

---

## Woche 3 — Qualitätskontrolle & Risiko (2026-03-15 bis 2026-03-21)

### Aufgaben Woche 3

- [ ] Incident-Root-Cause Analyse für alle offenen Punkte
- [ ] Logqualität prüfen (keine stillen Fehler, sinnvolle Warnungen)
- [ ] Drawdown-/Kill-Switch Verhalten gegenchecken (simulativ/Log-basiert)
- [ ] Datenfrische prüfen (kein dauerhafter `STALE` im Dashboard)
- [ ] Prozesshärtung: Startreihenfolge und Notfall-Runbook finalisieren

### Done-Definition (Woche 3)

- [ ] Alle Incidents haben Status + Ursache + Fix/Workaround
- [ ] Keine wiederkehrende Störung > 2x ohne Gegenmaßnahme
- [ ] Notfallablauf schriftlich finalisiert

---

## Woche 4 — Erste belastbare Zwischenbewertung (2026-03-22 bis 2026-03-28)

### Aufgaben Woche 4

- [ ] 4-Wochen Zwischenreport erstellen (`reports/paper_4w_review_2026-03-28.md`)
- [ ] KPI-Trend bewerten (nicht nur Einzelwoche)
- [ ] Feintuning-Entscheidung treffen (Ja/Nein + Begründung)
- [ ] Falls Feintuning: nur 1-2 vorab definierte Parameteränderungen
- [ ] Nächste 8 Wochen (Monat 2+3) konkret planen

### Done-Definition (Woche 4)

- [ ] Zwischenreport dokumentiert Technikstabilität + KPI-Tendenz
- [ ] Entscheidung festgehalten: **Stabil halten** oder **kontrolliertes Feintuning**
- [ ] Keine ungeplanten Strategieänderungen außerhalb des Beschlusses

---

## Entscheidungsregeln für Feintuning (in den ersten 4 Wochen)

### Feintuning **erlaubt**, wenn

- klarer technischer Fehler oder Reproduzierbarkeitsproblem vorliegt
- KPI-Ausreißer auf Betriebsproblem (nicht Marktphase) zurückzuführen sind
- Änderung klein, reversibel und dokumentiert ist

### Feintuning **nicht erlaubt**, wenn

- nur kurzfristige Performance-Schwankung ohne technische Ursache vorliegt
- mehrere Parameter gleichzeitig geändert würden
- keine Vorher/Nachher-Definition existiert

---

## Abhak-Board (kompakt)

- [ ] W1 abgeschlossen (Stabilität)
- [ ] W2 abgeschlossen (Monitoring-Routine)
- [ ] W3 abgeschlossen (Qualitätskontrolle)
- [ ] W4 abgeschlossen (Zwischenbewertung)

**Nächster Meilenstein nach W4:** Übergang in Monat 2/3 gemäß `paper_trading_90d_plan.md`.
