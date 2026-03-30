# 🚨 Befehle – STALE-Notfall (MT5 Dashboard)

Kurze Referenz für den Fall, dass das MT5-Dashboard plötzlich `STALE`, `PARTIAL_STALE`
oder sehr alte Minutenwerte zeigt.

---

## 📖 Inhaltsverzeichnis

- [Ziel](#ziel)
- [🪟 Schnellcheck](#-windows-laptop--schnellcheck)
  - [1) Lokale Logs prüfen](#1-aktive-lokale-logs-prüfen)
  - [2) MT5 Common\\Files prüfen](#2-mt5-commonfiles-prüfen)
  - [3) Dateizeit + Größe prüfen](#3-dateizeit--größe-prüfen)
- [🪟 Sync manuell anstoßen](#-windows-laptop--sync-manuell-anstoßen)
  - [4) Skriptordner wechseln](#4-in-den-mt5-skriptordner-wechseln)
  - [5) Dry-Run](#5-dry-run-ausführen)
  - [6) Echter Sync](#6-echten-sync-ausführen)
  - [7) Dauerlauf](#7-dauerlauf-starten)
- [🪟 Task Scheduler](#-windows-laptop--task-scheduler)
  - [8) Task starten](#8-sync-task-starten)
  - [9) Task-Status prüfen](#9-task-status-prüfen)
  - [10) Task neu registrieren](#10-task-neu-registrieren)
- [🪟 Trader neu starten](#-windows-laptop--trader-neu-starten)
  - [11) Alle Trader stoppen](#11-alle-trader-stoppen)
  - [12) Top-Konfiguration neu starten](#12-top-konfiguration-neu-starten)
- [🐧 Linux – Deploy](#-linux-server--frischen-code-deployen)
- [Entscheidungshilfe](#entscheidungshilfe)
  - [Fall A: Sync-Problem](#fall-a-lokale-logs-frisch-commonfiles-alt)
  - [Fall B: Indikator-Problem](#fall-b-lokale-logs-frisch-commonfiles-frisch-chart-trotzdem-stale)
  - [Fall C: Trader hängt](#fall-c-lokale-logs-selbst-alt)
- [⚡ Minimal-Sequenz](#empfohlene-minimal-sequenz)
- [Merksatz](#merksatz)

---

## Ziel

In 1–3 Minuten herausfinden:

1. Sind die aktiven Trader-Logs frisch?
2. Sind die MT5 `Common\Files` frisch?
3. Hängt der Sync?
4. Muss nur der Indikator neu geladen werden?

---

## 🪟 Windows-Laptop – Schnellcheck

### 1) Aktive lokale Logs prüfen

```powershell
# Zeigt die letzten 5 Zeilen der USDCAD-Signal-Datei an.
# Diese Datei wird vom live_trader.py geschrieben – jede Zeile ist ein Signal-Eintrag.
# Wenn die Zeitstempel hier aktuell sind (letzte 15–30 Min), läuft der Trader korrekt.
# Wenn die Zeitstempel alt sind → der Trader-Prozess hängt oder ist abgestürzt (→ Fall C).
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\USDCAD_signals.csv" -Tail 5

# Dasselbe für USDJPY – beide Paare separat prüfen,
# weil ein Trader laufen kann, während der andere hängt.
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\USDJPY_signals.csv" -Tail 5
```

### 2) MT5 `Common\Files` prüfen

```powershell
# Zeigt die letzten 5 Zeilen der USDCAD-Signal-Datei im MT5-Common-Ordner.
# Das ist der Ordner, aus dem der MT5-Indikator (LiveSignalDashboard) die Daten liest.
# Die Datei wird vom Sync-Skript hierhin kopiert (aus dem lokalen logs-Ordner).
# Wenn diese Daten ALT sind, aber die lokalen Logs (Schritt 1) frisch → Sync-Problem (→ Fall A).
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" -Tail 5

# Dasselbe für USDJPY.
# $env:APPDATA löst sich auf zu z.B. C:\Users\<Name>\AppData\Roaming
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" -Tail 5
```

### 3) Dateizeit + Größe prüfen

```powershell
# Zeigt den letzten Schreibzeitpunkt (LastWriteTime) und die Dateigröße (Length in Bytes).
# Damit siehst du auf einen Blick, WANN der Sync zuletzt die Datei aktualisiert hat.
# Wenn LastWriteTime mehrere Stunden zurückliegt → Sync läuft nicht (→ Fall A).
# Wenn Length = 0 → Datei ist leer, d.h. Trader hat noch keine Signale geschrieben.
Get-Item "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" | Select-Object LastWriteTime, Length

# Dasselbe für USDJPY – immer beide Symbole prüfen.
Get-Item "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" | Select-Object LastWriteTime, Length
```

---

## 🪟 Windows-Laptop – Sync manuell anstoßen

### 4) In den MT5-Skriptordner wechseln

```powershell
# Wechselt in den Ordner, in dem das Sync-Skript liegt.
# Muss vor den folgenden Befehlen (5–7) ausgeführt werden,
# weil die Skripte mit relativem Pfad (.\) aufgerufen werden.
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
```

### 5) Dry-Run ausführen

```powershell
# Führt das Sync-Skript im Testmodus aus (kein Kopieren, nur Anzeige).
# -ExecutionPolicy Bypass: Umgeht die Windows-Sicherheitsrichtlinie für Skript-Ausführung.
# -File .\sync_live_logs_to_mt5_common.ps1: Das Sync-Skript, das CSV-Dateien kopiert.
# -SourceDir: Quellordner, aus dem die Signal-CSVs gelesen werden.
# -DryRun: WICHTIG – es wird NICHTS kopiert, nur angezeigt, WAS kopiert WÜRDE.
# Nutze das zuerst, um zu sehen, ob das Skript die richtigen Dateien findet.
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -DryRun
```

### 6) Echten Sync ausführen

```powershell
# Führt den Sync EINMALIG aus – kopiert alle Signal-CSVs aus dem logs-Ordner
# in den MT5 Common\Files-Ordner, damit der Indikator die aktuellen Daten sieht.
# Ohne -DryRun wird tatsächlich kopiert.
# Nach dem Ausführen: in MT5 prüfen, ob das Dashboard aktualisiert.
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
```

### 7) Dauerlauf starten

```powershell
# Startet den Sync als Endlosschleife – kopiert alle 5 Sekunden neu.
# -Continuous: Aktiviert den Dauerlauf-Modus (Endlosschleife).
# -IntervalSec 5: Pause von 5 Sekunden zwischen jedem Sync-Durchlauf.
# Das Fenster bleibt offen, solange der Sync läuft.
# Beenden mit Strg+C. Alternativ: Task Scheduler nutzen (Schritt 8–10).
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -Continuous -IntervalSec 5
```

---

## 🪟 Windows-Laptop – Task Scheduler

### 8) Sync-Task starten

```powershell
# Startet den Windows-Task "MT5_Sync_Live_Logs" manuell.
# Dieser Task wurde mit install_sync_task.ps1 (Schritt 10) im Task Scheduler registriert.
# Er führt das Sync-Skript automatisch im Hintergrund aus.
# Vorteil gegenüber Dauerlauf (Schritt 7): Kein offenes PowerShell-Fenster nötig.
Start-ScheduledTask -TaskName "MT5_Sync_Live_Logs"
```

### 9) Task-Status prüfen

```powershell
# Zeigt den aktuellen Status des Sync-Tasks an:
# - LastRunTime: Wann lief der Task zuletzt?
# - LastTaskResult: 0 = Erfolg, anderer Wert = Fehler
# - NextRunTime: Wann wird der Task als nächstes ausgeführt?
# Wenn LastTaskResult ≠ 0 → Task neu registrieren (Schritt 10).
Get-ScheduledTask -TaskName "MT5_Sync_Live_Logs" | Get-ScheduledTaskInfo
```

### 10) Task neu registrieren

```powershell
# Wechselt in den Ordner mit dem Installations-Skript.
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"

# Registriert den Sync-Task neu im Windows Task Scheduler.
# -SourceDir: Quellordner für die Signal-CSVs.
# -IntervalSec 5: Der Task synchronisiert alle 5 Sekunden.
# -RunHidden: Task läuft unsichtbar im Hintergrund (kein Fenster).
# -Force: Überschreibt einen vorhandenen Task gleichen Namens ohne Rückfrage.
# Nach dem Ausführen: mit Schritt 9 prüfen, ob der Task läuft.
powershell -ExecutionPolicy Bypass -File .\install_sync_task.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -IntervalSec 5 -RunHidden -Force
```

---

## 🪟 Windows-Laptop – Trader neu starten

### 11) Alle Trader stoppen

```powershell
# Wechselt in das Hauptverzeichnis des Trading-Systems auf dem Laptop.
cd "C:\Users\Sebastian Setnescu\mt5_trading"

# Stoppt ALLE laufenden Trader-Prozesse (live_trader.py für jedes Symbol).
# Das Skript beendet alle Python-Prozesse, die zum Trading-System gehören.
# ACHTUNG: Offene Positionen werden NICHT automatisch geschlossen –
# sie bleiben in MT5 offen, bis der Trader neu gestartet wird.
.\stop_all_traders.bat
```

### 12) Top-Konfiguration neu starten

```powershell
# Wechselt in das Hauptverzeichnis (falls nicht schon dort).
cd "C:\Users\Sebastian Setnescu\mt5_trading"

# Startet die Trader mit der besten Konfiguration aus dem Testplan (H1 + M15).
# Dieses Batch-Skript startet für jedes aktive Symbol (USDCAD, USDJPY) einen
# eigenen live_trader.py-Prozess im Paper-Trading-Modus.
# Die Konfiguration (Timeframe, Modell, Threshold etc.) ist im Skript hinterlegt.
# Nach dem Start: Schritt 1 wiederholen, um zu prüfen, ob Signale geschrieben werden.
.\start_testphase_topconfig_H1_M15.bat
```

---

## 🐧 Linux-Server – frischen Code deployen

### 13) Deploy auf den Laptop

```bash
# Wechselt in das Projektverzeichnis auf dem Linux-Server.
cd /mnt/1Tb-Data/XGBoost-LightGBM

# Führt das Deploy-Skript aus, das alle relevanten Dateien per SCP/rsync
# auf den Windows-Laptop kopiert (live_trader.py, Modelle, Konfiguration etc.).
# Voraussetzung: SSH-Zugang zum Laptop muss konfiguriert sein.
# Nach dem Deploy: auf dem Laptop die Trader neu starten (Schritt 11 + 12).
bash deploy_to_laptop.sh
```

---

## Entscheidungshilfe

### Fall A: lokale Logs frisch, `Common\Files` alt

→ Sync-Problem: Der Trader schreibt korrekt Signale, aber das Sync-Skript kopiert sie
nicht in den MT5-Common-Ordner. Der Indikator sieht daher veraltete Daten.

```powershell
# Lösung: Einmalig manuell synchronisieren.
# Wechselt in den Skriptordner und führt einen sofortigen Sync aus.
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
# Danach: Task Scheduler prüfen (Schritt 9), ob der automatische Sync wieder läuft.
```

### Fall B: lokale Logs frisch, `Common\Files` frisch, Chart trotzdem stale

→ MT5-Indikator neu laden: Die Daten sind korrekt vorhanden, aber der Indikator
im MT5-Chart liest sie nicht mehr ein (z.B. nach MT5-Update oder Absturz).

Schritte in MT5:

1. `LiveSignalDashboard` vom Chart entfernen
2. in MetaEditor neu kompilieren
3. neu auf den Chart ziehen
4. Inputs prüfen:
   - `InpUseCommonFiles=true` – Indikator liest aus Common\Files statt lokalem Ordner
   - `InpSymbol1=USDCAD` – Erstes aktives Handelspaar
   - `InpSymbol2=USDJPY` – Zweites aktives Handelspaar
   - `InpUseSignalTimeframeForStale=true` – Stale-Erkennung basiert auf Signal-Timeframe
   - `InpSignalTimeframeMinutes=15` – Signale kommen alle 15 Min (M15/H1-Pipeline)

### Fall C: lokale Logs selbst alt

→ Trader/Prozess hängt: Der live_trader.py-Prozess schreibt keine neuen Signale mehr.
Entweder ist er abgestürzt, hängt in einer Endlosschleife, oder MT5 ist nicht verbunden.

```powershell
# Lösung: Alle Trader stoppen und mit der aktuellen Konfiguration neu starten.
cd "C:\Users\Sebastian Setnescu\mt5_trading"

# Zuerst alle laufenden Prozesse beenden.
.\stop_all_traders.bat

# Dann Trader mit der Top-Konfiguration neu starten.
# Danach Schritt 1 wiederholen und prüfen, ob frische Signale erscheinen.
.\start_testphase_topconfig_H1_M15.bat
```

---

## Empfohlene Minimal-Sequenz

Wenn du wenig Zeit hast, nimm genau diese Befehle:

```powershell
# --- SCHRITT 1: Diagnose – Wo liegt das Problem? ---

# Prüfe: Schreibt der USDCAD-Trader frische Signale? (Zeitstempel anschauen)
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\USDCAD_signals.csv" -Tail 5

# Prüfe: Schreibt der USDJPY-Trader frische Signale?
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\USDJPY_signals.csv" -Tail 5

# Prüfe: Sind die Dateien im MT5-Common-Ordner aktuell? (Vergleich mit lokalen Logs)
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" -Tail 5
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" -Tail 5

# --- SCHRITT 2: Fix – Sync manuell anstoßen ---

# In den Skriptordner wechseln und Sync einmalig ausführen.
# Das kopiert die aktuellen Signal-CSVs in den MT5-Common-Ordner.
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
```

---

## Merksatz

- **Logs frisch + Common Files alt** → Sync prüfen
- **Logs frisch + Common Files frisch + Chart stale** → Indikator neu laden
- **Logs alt** → Trader neu starten

---

Stand: 2026-03-14
