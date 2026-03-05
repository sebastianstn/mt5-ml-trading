# Operator-Checkliste (MT5 Dashboard + Shadow-Engine)

Ziel: In 3–5 Minuten prüfen, ob das System stabil läuft und ob Eingriff nötig ist.

---

## Ampel-Logik (Sofortbewertung)

- 🟢 **Grün (normal)**
  - `STATUS=CONNECTED`
  - `State` pro Symbol = `LIVE_SIGNAL` oder `LIVE_NO_SIGNAL`
  - `STALE>70min (SignalTF)` sichtbar (für H1-Signalbetrieb)
  - `Rows` steigen über die Zeit

- 🟡 **Gelb (beobachten)**
  - Kurzzeitig `PARTIAL` oder `WAITING_FOR_CSV` direkt nach Neustart
  - Einzelnes Symbol kurz `STALE`, dann Rückkehr zu `LIVE_*`
  - `History-Entries` ist 0 bei ruhigem Markt (kann normal sein)

- 🔴 **Rot (Eingriff nötig)**
  - `MISSING` > 5 Minuten trotz laufender Engine
  - `PARTIAL_STALE`/`STALE` bleibt über 2 Prüfzyklen bestehen
  - `Rows` steigt über ~2 H1-Zyklen nicht
  - Stop-Skript beendet Prozesse nicht zuverlässig

---

## Start-Check (vor Betriebsbeginn)

1. MT5 ist geöffnet und eingeloggt.
2. `stop_all_traders.bat` ausführen (sauberer Zustand).
3. `start_shadow_compare.bat` starten.
4. Dashboard-EA auf **einem** Chart aktivieren.
5. Im Dashboard prüfen:
   - `STATUS=CONNECTED`
   - `STALE>70min (SignalTF)`
   - beide Symbole werden angezeigt

---

## Laufender Betrieb (alle 30–60 Minuten)

1. **Gesamtstatus** prüfen (`CONNECTED` erwartet).
2. **Frische** prüfen (`State` nicht dauerhaft `STALE`).
3. **Rows** prüfen (müssen mit der Zeit wachsen).
4. **Dir/Prob/Regime** auf Plausibilität prüfen.
5. **Zeichnungen** prüfen (Entry/SL/TP/History vorhanden, wenn Signale da sind).

---

## Eskalation: Wann Copilot sofort einbeziehen?

Bitte sofort melden mit Screenshot + Uhrzeit + Symbol, wenn:

- `MISSING` oder `STALE` dauerhaft bleibt
- Dashboard und Logs widersprüchlich sind
- keine neuen CSV-Updates trotz laufendem Trader kommen
- Entry/SL/TP offensichtlich falsch eingezeichnet werden
- nach Deploy/Compile die erwartete Version nicht sichtbar ist

---

## Standard-Fehlerbilder & Sofortmaßnahmen

### Problem: `WAITING_FOR_CSV` bleibt stehen

- Prüfen, ob `start_shadow_compare.bat` wirklich läuft
- Prüfen, ob CSV im MT5 Common Files Pfad landet
- Prüfen, ob `InpUseCommonFiles=true`

### Problem: `STALE` auf M5-Chart trotz H1-Betrieb

- Prüfen, ob `InpUseSignalTimeframeForStale=true`
- Prüfen, ob `InpSignalTimeframeMinutes=60`

### Problem: Keine Pfeile sichtbar

- Prüfen, ob `InpDrawTrades=true` und `InpDrawEntryHistory=true`
- Prüfen, ob `History-Entries` > 0
- Zoom/Chartbereich anpassen

---

## Betriebsregel (wichtig)

- `start_shadow_compare.bat` = **Engine (arbeitet)**
- `LiveSignalDashboard.mq5` = **Monitoring (zeigt an)**

Single Source of Truth bleibt das Repo auf Linux.
Änderungen: Repo -> Deploy -> MT5 kompilieren/testen.
