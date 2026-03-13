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
  - [12) Test 128 neu starten](#12-test-128-neu-starten)
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
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\paper_test128\USDCAD_signals.csv" -Tail 5
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\paper_test128\USDJPY_signals.csv" -Tail 5
```

### 2) MT5 `Common\Files` prüfen

```powershell
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" -Tail 5
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" -Tail 5
```

### 3) Dateizeit + Größe prüfen

```powershell
Get-Item "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" | Select-Object LastWriteTime, Length
Get-Item "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" | Select-Object LastWriteTime, Length
```

---

## 🪟 Windows-Laptop – Sync manuell anstoßen

### 4) In den MT5-Skriptordner wechseln

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
```

### 5) Dry-Run ausführen

```powershell
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -DryRun
```

### 6) Echten Sync ausführen

```powershell
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
```

### 7) Dauerlauf starten

```powershell
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -Continuous -IntervalSec 5
```

---

## 🪟 Windows-Laptop – Task Scheduler

### 8) Sync-Task starten

```powershell
Start-ScheduledTask -TaskName "MT5_Sync_Live_Logs"
```

### 9) Task-Status prüfen

```powershell
Get-ScheduledTask -TaskName "MT5_Sync_Live_Logs" | Get-ScheduledTaskInfo
```

### 10) Task neu registrieren

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
powershell -ExecutionPolicy Bypass -File .\install_sync_task.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs" -IntervalSec 5 -RunHidden -Force
```

---

## 🪟 Windows-Laptop – Trader neu starten

### 11) Alle Trader stoppen

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
.\stop_all_traders.bat
```

### 12) Test 128 neu starten

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
.\start_paper_trading_test128.bat
```

---

## 🐧 Linux-Server – frischen Code deployen

### 13) Deploy auf den Laptop

```bash
cd /mnt/1Tb-Data/XGBoost-LightGBM
bash deploy_to_laptop.sh
```

---

## Entscheidungshilfe

### Fall A: lokale Logs frisch, `Common\Files` alt

→ Sync-Problem

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
```

### Fall B: lokale Logs frisch, `Common\Files` frisch, Chart trotzdem stale

→ MT5-Indikator neu laden

Schritte in MT5:

1. `LiveSignalDashboard` vom Chart entfernen
2. in MetaEditor neu kompilieren
3. neu auf den Chart ziehen
4. Inputs prüfen:
   - `InpUseCommonFiles=true`
   - `InpSymbol1=USDCAD`
   - `InpSymbol2=USDJPY`
   - `InpUseSignalTimeframeForStale=true`
   - `InpSignalTimeframeMinutes=60`

### Fall C: lokale Logs selbst alt

→ Trader/Prozess hängt

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
.\stop_all_traders.bat
.\start_paper_trading_test128.bat
```

---

## Empfohlene Minimal-Sequenz

Wenn du wenig Zeit hast, nimm genau diese Befehle:

```powershell
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\paper_test128\USDCAD_signals.csv" -Tail 5
Get-Content "C:\Users\Sebastian Setnescu\mt5_trading\logs\paper_test128\USDJPY_signals.csv" -Tail 5
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDCAD_signals.csv" -Tail 5
Get-Content "$env:APPDATA\MetaQuotes\Terminal\Common\Files\USDJPY_signals.csv" -Tail 5
cd "C:\Users\Sebastian Setnescu\mt5_trading\live\mt5"
powershell -ExecutionPolicy Bypass -File .\sync_live_logs_to_mt5_common.ps1 -SourceDir "C:\Users\Sebastian Setnescu\mt5_trading\logs"
```

---

## Merksatz

- **Logs frisch + Common Files alt** → Sync prüfen
- **Logs frisch + Common Files frisch + Chart stale** → Indikator neu laden
- **Logs alt** → Trader neu starten

---

Stand: 2026-03-10
