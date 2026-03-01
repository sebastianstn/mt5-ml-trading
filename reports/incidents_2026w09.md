# Incident-Log – KW09 (2026-03-01 bis 2026-03-07)

**Zweck:** Alle technischen Störungen, Ursachen und Gegenmaßnahmen während Woche 1 dokumentieren.

---

## Kurzstatus

- **Systemmodus:** Paper-Trading (USDCAD, USDJPY)
- **Dashboard:** aktiv
- **Sync-Task:** aktiv
- **Owner:** Sebastian

---

## Incident-Tabelle

| Datum/Zeit | Symbol/System | Schweregrad | Symptom | Root Cause | Sofortmaßnahme | Dauer | Status | Prävention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-01 16:10 | Dashboard/MT5 | Mittel | `Datei fehlt` im EA, wiederholte Meldungen | Sync/Dateipfad initial nicht stabil eingerichtet | Sync-Task eingerichtet, manuelles Sync getestet | ~60 min | Behoben | Task dauerhaft auf 1-Minuten-Trigger + Daily Check |
| 2026-03-01 16:53 | Python/Trader | Mittel | `ModuleNotFoundError: numpy` auf Laptop | venv ohne vollständige Paketinstallation gestartet | `requirements-laptop.txt` installiert, Import-Test durchgeführt | ~20 min | Behoben | Standard-Start mit `./.venv/Scripts/python.exe` |
| 2026-03-01 17:08 | Strategie | Info | Kein Trade (Regime Seitwärts) | Regime-Filter blockiert korrekt Seitwärtsmarkt | Kein Eingriff notwendig | laufend | Erwartetes Verhalten | Monitoring fortsetzen |
| 2026-03-01 17:39 | Pipeline | Niedrig | CSVs wurden als `STALE` erkannt | Alte `live_trader.py` ohne Heartbeat auf Laptop | Heartbeat-Version deployed, `--heartbeat_log 1` aktiv | ~30 min | Behoben | Heartbeat als Default beibehalten |
| 2026-03-01 18:05 | Ops/Windows | Niedrig | PowerShell-Kommandos wurden versehentlich zusammenkopiert | Mehrere Befehle in einer Zeile ohne Trennung ausgeführt | Kommandos einzeln erneut ausgeführt, Task danach erfolgreich registriert | ~15 min | Behoben | Runbook: pro Schritt genau ein Kommando + Ergebnis prüfen |

---

## Schweregrad-Definition

- **Kritisch:** Trading/Überwachung vollständig ausgefallen, sofortiger Eingriff nötig
- **Mittel:** Teilfunktion gestört, Betrieb eingeschränkt
- **Niedrig:** Komfort-/Monitoringproblem, Kernbetrieb läuft
- **Info:** erwartetes Verhalten, kein Fehler

---

## Offene Punkte (Woche 1)

- [x] Tägliche Incident-Prüfung für 2026-03-01 durchgeführt
- [ ] Tägliche Incident-Prüfung für Folgetage durchführen und Tabelle ergänzen
- [ ] Wiederkehrende Muster identifizieren (>= 2x gleiches Symptom)
- [ ] Am Wochenende W1-Zusammenfassung ergänzen

---

## W1-Zusammenfassung (am 2026-03-07 ausfüllen)

- Gesamtanzahl Incidents:
- Davon kritisch:
- Durchschnittliche Behebungszeit:
- Top-Root-Cause:
- Maßnahmen für Woche 2:
