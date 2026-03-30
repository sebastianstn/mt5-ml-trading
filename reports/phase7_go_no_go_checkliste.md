# Phase 7 Go/No-Go Checkliste (60 Sekunden)

<!-- cspell:ignore No-Go Go/No-Go Runtime Heartbeat Watchdog Check-Zyklen USDCAD USDJPY INCIDENT overall_status TWO-STAGE Prob -->

Zweck: Schneller Gesundheitscheck nach Start von `start_testphase_topconfig_H1_M15.bat`.

## 1) Prozessstatus
- [ ] USDCAD-Fenster läuft ohne Crash
- [ ] USDJPY-Fenster läuft ohne Crash
- [ ] Kein permanenter Exception-Loop im Terminal

## 2) Runtime-Heartbeat (Pflicht)
Prüfen in `logs/live_trader.log`:
- [ ] Für USDCAD erscheint: `Neue M15-Kerze | ... UTC`
- [ ] Für USDJPY erscheint: `Neue M15-Kerze | ... UTC`
- [ ] Beide Symbole bekommen spätestens alle 15 Minuten neue Kerzen

No-Go wenn:
- Ein Symbol > 30 Minuten keinen neuen Kerzen-Heartbeat schreibt.

## 3) Signal-CSV-Aktualität (Pflicht)
Prüfen in:
- `logs/USDCAD_signals.csv`
- `logs/USDJPY_signals.csv`

- [ ] Beide Dateien haben neue Zeilen nach dem Startzeitpunkt
- [ ] Die Zeitstempel laufen weiter (keine feste, alte Uhrzeit)

No-Go wenn:
- Eine Symbol-CSV > 30 Minuten nicht fortgeschrieben wird.

## 4) Watchdog-Status
Prüfen in:
- `logs/live_log_watchdog_latest.json`
- `logs/live_log_watchdog_latest.csv`

- [ ] `overall_status` ist `OK` oder `WATCH`
- [ ] Kein dauerhafter `INCIDENT` über 2 Check-Zyklen

No-Go wenn:
- `INCIDENT` in 2 aufeinanderfolgenden Zyklen bleibt.

## 5) Signalqualität (Kurzcheck)
Prüfen in `logs/live_trader.log`:
- [ ] `TWO-STAGE DEBUG` erscheint für beide Symbole
- [ ] `Signal=... | Prob=...` erscheint für beide Symbole
- [ ] Beobachtungsphase zählt erwartungsgemäß hoch (`Kerze 1/5`, `2/5`, ...)

## 6) Sofortmaßnahmen bei No-Go
1. Nur betroffenes Symbol-Fenster neu starten (nicht beide blind stoppen).
2. 1 Kerzenzyklus warten (M15).
3. Wenn weiter No-Go: kompletten Start via `start_testphase_topconfig_H1_M15.bat` neu ausführen.
4. Danach Watchdog erneut prüfen.

## Entscheidungsregel
- GO: Alle Pflichtpunkte 2, 3 und 4 sind erfüllt.
- NO-GO: Mindestens ein Pflichtpunkt verletzt.
