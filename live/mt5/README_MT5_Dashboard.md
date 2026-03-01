# MT5 Live Signal Dashboard (USDCAD / USDJPY)

Diese Dateien helfen dir, Python-Signale in MT5 **zeitnah** zu sehen:

- `LiveSignalDashboard.mq5` → MT5 Expert Advisor (Chart-Panel + Alerts)
- `sync_live_logs_to_mt5_common.ps1` → kopiert Python-Logs in MT5 Common Files
- `install_sync_task.ps1` → registriert den Autostart-Task beim Login
- `sync_live_logs_task_template.xml` → importierbare Task-Scheduler XML-Vorlage

---

## Aktueller Stand (2026-03-01)

- Paper-Betrieb für `USDCAD` und `USDJPY` läuft.
- Sync-Task ist produktiv nutzbar (`LastTaskResult = 0`).
- Dashboard zeigt Zustände pro Symbol (`MISSING`, `STALE`, `LIVE_*`) und Gesamtzustand (`CONNECTED`, `PARTIAL`, ...).
- `live_trader.py` unterstützt Heartbeat-Logging (`--heartbeat_log 1`), damit Frische auch ohne Trade sichtbar bleibt.

---

## 1) EA in MT5 installieren (Windows Laptop)

1. MT5 öffnen
2. `Datei -> Datenordner öffnen`
3. In `MQL5/Experts/` die Datei `LiveSignalDashboard.mq5` kopieren
4. In MT5 MetaEditor öffnen, Datei kompilieren
5. EA auf ein Chart ziehen

Empfohlene Inputs:

- `InpSymbol1 = USDCAD`
- `InpSymbol2 = USDJPY`
- `InpUseCommonFiles = true`
- `InpRefreshSeconds = 5`
- `InpEnableAlerts = true`

---

## 2) Logs aus Python in MT5 verfügbar machen

Der EA liest aus:

`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDCAD_live_trades.csv`
`%APPDATA%\MetaQuotes\Terminal\Common\Files\USDJPY_live_trades.csv`

Nutze dazu das Skript:

- `sync_live_logs_to_mt5_common.ps1`

### 2.1 Einmaliger Testlauf (Windows PowerShell)

1. PowerShell als normaler Benutzer öffnen
2. In den Skriptordner wechseln
3. Erst Dry-Run, dann echter Lauf

Beispiel:

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs" -DryRun`

`powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\<DEIN_USER>\mt5_trading\logs"`

Wenn alles passt, sollten im Zielordner erscheinen:

- `USDCAD_live_trades.csv`
- `USDJPY_live_trades.csv`

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

Im EA-Dashboard sollte oben der Gesamtstatus von `WAITING_FOR_CSV` auf `CONNECTED` wechseln,
sobald beide Dateien verfügbar und frisch sind.

---

### 2.5 Typische Fehler & schnelle Fixes

1. **"Datei fehlt" in MT5 bleibt bestehen**
   - Prüfe `-SourceDir` (existiert der Ordner wirklich?)
   - Prüfe, ob `live_trader.py` auf dem Laptop läuft und `logs/*.csv` schreibt
   - Prüfe MT5 Input `InpUseCommonFiles=true`

2. **PowerShell blockiert Skript**
   - Mit `-ExecutionPolicy Bypass` starten (wie oben)

3. **Doppelte Dashboard-Logs**
   - EA nur auf **einem** Chart laufen lassen (liest ohnehin beide Symbole)

4. **Automated trading is disabled**
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

## 4) Bessere Idee (für später)

Wenn du noch schneller/robuster willst:

1. Python schreibt zusätzlich eine kompakte `dashboard_state.json`
2. EA liest nur diese JSON (statt vollständiger CSV)
3. Optional: Telegram/Push bei NO-GO-Ereignissen

Das kann ich dir als nächsten Schritt auch direkt bauen.
