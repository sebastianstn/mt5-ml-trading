# MT5 Live Signal Dashboard (USDCAD / USDJPY)

Diese Dateien helfen dir, Python-Signale in MT5 **zeitnah** zu sehen:

- `LiveSignalDashboard.mq5` → MT5 Indikator (Chart-Panel + Alerts)
- `sync_live_logs_to_mt5_common.ps1` → kopiert Python-Logs in MT5 Common Files
- `install_sync_task.ps1` → registriert den Autostart-Task beim Login
- `sync_live_logs_task_template.xml` → importierbare Task-Scheduler XML-Vorlage
- `OPERATOR_CHECKLIST.md` → kompakte Ampel-Checkliste für den laufenden Betrieb
- `CHART_LESELEITFADEN.md` → ausführliche Anleitung „Wie lese ich den Chart?“

---

## 📖 Inhaltsverzeichnis

- [Aktueller Stand](#aktueller-stand-2026-03-09)
- [1. Indikator installieren](#1-indikator-in-mt5-installieren-windows-laptop)
- [2. Logs in MT5 verfügbar machen](#2-logs-aus-python-in-mt5-verfügbar-machen)
  - [2.1 Einmaliger Testlauf](#21-einmaliger-testlauf-windows-powershell)
  - [2.2 Dauerlauf](#22-dauerlauf-empfohlen-während-paper-trading)
  - [2.3 Autostart Task Scheduler](#23-autostart-beim-login-task-scheduler)
  - [2.4 Verifikation in MT5](#24-verifikation-in-mt5)
  - [2.5 Typische Fehler](#25-typische-fehler--schnelle-fixes)
- [3. Was du im Dashboard siehst](#3-was-du-im-mt5-dashboard-siehst)
- [4. Bessere Idee (für später)](#4-bessere-idee-für-später)
- [5. Kompatibilitätshinweis](#5-kompatibilitätshinweis-wichtig)

---

## Aktueller Stand (2026-03-09)

- Paper-Betrieb für `USDCAD` und `USDJPY` läuft.
- Sync-Task ist produktiv nutzbar (`LastTaskResult = 0`).
- Dashboard zeigt Zustände pro Symbol (`MISSING`, `STALE`, `LIVE_*`) und Gesamtzustand (`CONNECTED`, `PARTIAL`, ...).
- `live_trader.py` unterstützt Heartbeat-Logging (`--heartbeat_log 1`), damit Frische auch ohne Trade sichtbar bleibt.

---

## 1) Indikator in MT5 installieren (Windows Laptop)

1. MT5 öffnen
2. `Datei -> Datenordner öffnen`
3. In `MQL5/Indicators/` die Datei `LiveSignalDashboard.mq5` kopieren
4. In MT5 MetaEditor öffnen, Datei kompilieren
5. Indikator auf ein Chart ziehen

Empfohlene Inputs:

- `InpSymbol1 = USDCAD`
- `InpSymbol2 = USDJPY`
- `InpUseCommonFiles = true`
- `InpRefreshSeconds = 5`
- `InpEnableAlerts = true`

---

## 2) Logs aus Python in MT5 verfügbar machen

Der Indikator liest aus (aktuell):

`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv`
`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv`

Optional (für Auswertungen/Close-Events):

`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDCAD_closes.csv`
`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDJPY_closes.csv`

Nutze dazu das Skript:

- `sync_live_logs_to_mt5_common.ps1`

Wichtig:

- Das Sync-Skript durchsucht jetzt standardmäßig auch Unterordner unter `logs/`
- Dadurch werden aktive Läufe wie `logs\paper_test128\USDCAD_signals.csv` bevorzugt,
  selbst wenn im Root-Ordner noch ältere `*_signals.csv` liegen

### 2.1 Einmaliger Testlauf (Windows PowerShell)

1. PowerShell als normaler Benutzer öffnen
2. In den Skriptordner wechseln
3. Erst Dry-Run, dann echter Lauf

Beispiel:

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs" -DryRun`

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs"`

Wenn alles passt, sollten im Zielordner erscheinen:

- `USDCAD_signals.csv`
- `USDJPY_signals.csv`
- optional zusätzlich `*_closes.csv`

### 2.2 Dauerlauf (empfohlen während Paper-Trading)

Starte den Sync mit Dauerlauf (alle 5 Sekunden):

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs" -Continuous -IntervalSec 5`

### 2.3 Autostart beim Login (Task Scheduler)

#### Empfohlen: 1-Klick per PowerShell-Installer

Im Ordner `live/mt5/` auf dem Windows-Laptop ausführen:

`powershell -ExecutionPolicy Bypass -File .\install_sync_task.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs" -IntervalSec 5 -RunHidden -Force`

Danach direkt testen:

`Start-ScheduledTask -TaskName "MT5_Sync_Live_Logs"`

Status prüfen:

`Get-ScheduledTask -TaskName "MT5_Sync_Live_Logs" | Get-ScheduledTaskInfo`

Task wieder entfernen (falls nötig):

`Unregister-ScheduledTask -TaskName "MT5_Sync_Live_Logs" -Confirm:$false`

#### Alternative: XML-Import

Datei: `sync_live_logs_task_template.xml`

1. Vor dem Import Platzhalter ersetzen:
   - `__WINDOWS_USER__` → z. B. `DESKTOP-ABC123\Sebastian`
   - `__SCRIPT_PATH__` → voller Pfad zu `sync_live_logs_to_mt5_common.ps1`
   - `__SOURCE_DIR__` → voller Pfad zu deinem `logs`-Ordner
2. In Task Scheduler: **Aufgabe importieren...**
3. XML auswählen und speichern

Damit läuft der Sync automatisch nach Windows-Login.

### 2.4 Verifikation in MT5

Im Dashboard sollte oben der Gesamtstatus von `WAITING_FOR_CSV` auf `CONNECTED` wechseln,
sobald beide Dateien verfügbar und frisch sind.

---

### 2.5 Typische Fehler & schnelle Fixes

1. **"Datei fehlt" in MT5 bleibt bestehen**
   - Prüfe `-SourceDir` (existiert der Ordner wirklich?)
   - Prüfe, ob `live_trader.py` auf dem Laptop läuft und `logs/*.csv` schreibt
   - Prüfe MT5 Input `InpUseCommonFiles=true`

2. **Dashboard zeigt uralte Werte wie `STALE (1900 min)` obwohl `paper_test128` läuft**
   - Häufige Ursache: Im Root-Ordner `logs\` liegen alte `USDCAD_signals.csv` / `USDJPY_signals.csv`,
     während der aktive Lauf in `logs\paper_test128\` schreibt
   - Der aktualisierte Sync bevorzugt automatisch die frischste Datei je Symbol aus Unterordnern
   - Task nach Update einmal neu starten, damit die neue Skriptversion aktiv ist

3. **PowerShell blockiert Skript**
   - Mit `-ExecutionPolicy Bypass` starten (wie oben)

4. **Doppelte Dashboard-Logs**
   - Indikator nur auf **einem** Chart laufen lassen (liest ohnehin beide Symbole)

5. **Automated trading is disabled**
   - Für dieses Dashboard nicht kritisch (Dateilesen funktioniert trotzdem)
   - Für echte Trade-Ausführung im MT5-Terminal AutoTrading aktivieren

---

Beispiel (minimal):

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1`

---

## 3) Was du im MT5-Dashboard siehst

- Richtung (Long/Short/Kein)
- Wahrscheinlichkeit (`prob`)
- Regime-Name
- Modus (PAPER/LIVE)
- Datenfrische (OK/STALE)
- Neue Signale werden per `Alert()` gemeldet

---

## 5) Kompatibilitätshinweis (wichtig)

- `LiveSignalDashboard.mq5` (v2.24) ist auf das aktuelle Logging von `live_trader.py` ausgerichtet.
- Robustes Parsing von Zeitformaten (`YYYY-MM-DD HH:MM:SS`, ISO-Varianten).
- Robustes Mapping von `richtung` plus Fallback auf numerisches Feld `signal` (`2/-1/0`).

- Wenn das Dashboard trotz vorhandener CSV auf `NO_TS`/`MISSING` bleibt:
   1. Indikator neu kompilieren
   2. neu auf den Chart ziehen
   3. prüfen, dass `InpUseCommonFiles=true` gesetzt ist

---

## 4) Bessere Idee (für später)

Wenn du noch schneller/robuster willst:

1. Python schreibt zusätzlich eine kompakte `dashboard_state.json`
2. EA liest nur diese JSON (statt vollständiger CSV)
3. Optional: Telegram/Push bei NO-GO-Ereignissen

Das kann ich dir als nächsten Schritt auch direkt bauen.
