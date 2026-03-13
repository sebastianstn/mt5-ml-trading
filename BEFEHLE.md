# 📋 Befehlsreferenz – MT5 ML-Trading-System

> Alle Befehle auf einen Blick. Achte auf die Spalte **Gerät** – manche laufen nur auf Windows!

---

## 📖 Inhaltsverzeichnis

- [📍 Schnellübersicht](#-schnellübersicht)
- [🪟 Windows Laptop – Befehle](#-windows-laptop--befehle)
  - [Virtuelle Umgebung aktivieren](#virtuelle-umgebung-aktivieren)
  - [Daten von MT5 laden](#daten-von-mt5-laden-data_loaderpy)
  - [Paper-Trading starten](#paper-trading-starten-livelive_traderpy)
  - [Paper-Trading – Sofort-Start](#paper-trading--sofort-start-konkrete-werte-2026-03-03)
  - [Shadow-Compare starten](#shadow-compare-starten-usdcad-v4-vs-usdjpy-v5)
  - [Testphase starten](#testphase-starten-neue-top-konfiguration-two-stage-v4)
  - [Alle Trader stoppen](#alle-trader-stoppen-vor-neustart-empfohlen)
  - [Daten zum Server kopieren](#daten-vom-laptop-zum-linux-server-kopieren)
- [🐧 Linux Server – Befehle](#-linux-server--befehle)
  - [Virtuelle Umgebung aktivieren](#virtuelle-umgebung-aktivieren-linux)
  - [Feature Engineering](#feature-engineering-featuresfeature_engineeringpy)
  - [Labeling](#labeling-featureslabelingpy)
  - [Modell trainieren](#modell-trainieren-train_modelpy)
  - [Walk-Forward-Analyse](#walk-forward-analyse-walk_forwardpy)
  - [Backtest](#backtest-backtestbacktestpy)
  - [Retraining](#retraining-retrainingpy)
  - [📅 Sonntagsroutine](#-sonntagsroutine--wöchentlicher-check) → [`scripts/sunday_check.sh`](scripts/sunday_check.sh)
  - [KPI-Report](#kpi-report-reportsweekly_kpi_reportpy)
  - [Testphase-Auswertung](#testphase-auswertung-scriptsevaluate_mt5_testphasepy)
- [🔧 Pipeline-Skripte](#-pipeline-skripte-linux-server)
- [🚀 Deployment (Linux → Windows)](#-deployment-linux--windows)
- [📊 Referenz-Konfiguration](#-referenz-konfiguration-h1-baseline-stand-2026-03-03)
- [📉 Two-Stage Status](#-aktueller-two-stage-status-stand-2026-03-05-stress-re-run)
- [🔑 Regime-Nummern](#-regime-nummern-referenz)
- [📁 Wichtige Pfade](#-wichtige-pfade)

---

## 📍 Schnellübersicht

| Kategorie | Gerät | Skript |
| --------- | ----- | ------ |
| Daten laden | 🪟 Windows | `data_loader.py` |
| Features berechnen | 🐧 Linux | `features/feature_engineering.py` |
| Labels erstellen | 🐧 Linux | `features/labeling.py` |
| Modell trainieren | 🐧 Linux | `train_model.py` |
| Walk-Forward | 🐧 Linux | `walk_forward.py` |
| Backtest | 🐧 Linux | `backtest/backtest.py` |
| Retraining | 🐧 Linux | `retraining.py` |
| KPI-Report | 🐧 Linux | `reports/weekly_kpi_report.py` |
| Paper-Trading | 🪟 Windows | `live/live_trader.py` |
| Deploy auf Laptop | 🐧 Linux | `deploy_to_laptop.sh` |

> 🚨 Bei Dashboard-Problemen mit `STALE` / `PARTIAL_STALE` siehe direkt:
> `BEFEHLE_STALE_NOTFALL.md`
>
> Relevante Verzeichnisse im Workspace:
> [Projektwurzel](./) · [MT5-Dashboard-Skripte](live/mt5/) · [Test128-Logs](logs/paper_test128/)

### Im Notfall zuerst öffnen

1. [`BEFEHLE_STALE_NOTFALL.md`](BEFEHLE_STALE_NOTFALL.md) – wenn das Dashboard `STALE` oder `PARTIAL_STALE` zeigt
2. [`live/mt5/OPERATOR_CHECKLIST.md`](live/mt5/OPERATOR_CHECKLIST.md) – wenn du die Prüf-Reihenfolge Schritt für Schritt brauchst
3. [`logs/paper_test128/`](logs/paper_test128/) – wenn du direkt in die aktiven Signal- und Close-Logs schauen willst

### Wichtige Schnellzugriffe

| Bereich | Direktlink | Zweck |
| ------- | ---------- | ----- |
| Stale-Notfallhilfe | [`BEFEHLE_STALE_NOTFALL.md`](BEFEHLE_STALE_NOTFALL.md) | Sofortbefehle für `STALE` / `PARTIAL_STALE` |
| Operator-Checkliste | [`live/mt5/OPERATOR_CHECKLIST.md`](live/mt5/OPERATOR_CHECKLIST.md) | 1-Minuten-Check für Dashboard, Sync und Trader |
| Dashboard-Doku | [`live/mt5/README_MT5_Dashboard.md`](live/mt5/README_MT5_Dashboard.md) | Installation, Nutzung und Troubleshooting |
| Projektwurzel | [`./`](./) | Schnell zurück zur Repo-Übersicht |
| MT5-Dashboard-Skripte | [`live/mt5/`](live/mt5/) | EA, Sync-Skripte und MT5-Hilfsdateien |
| Test128-Logs | [`logs/paper_test128/`](logs/paper_test128/) | Aktive Signal-/Close-Logs für Paper-Betrieb |

---

## 🪟 Windows Laptop – Befehle

> Schnellzugriff für Dashboard-/Sync-Störungen:
> `BEFEHLE_STALE_NOTFALL.md`
>
> Direkt öffnen:
> [Projektwurzel](./) · [MT5-Dashboard-Skripte](live/mt5/) · [Test128-Logs](logs/paper_test128/)

### Virtuelle Umgebung aktivieren

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
venv\Scripts\activate
```

### Daten von MT5 laden (`data_loader.py`)

> ⚠️ MT5 Terminal muss geöffnet und eingeloggt sein!

```powershell
# Alle Symbole, H1 (Standard)
python data_loader.py --symbol alle

# Einzelnes Symbol, anderer Zeitrahmen
python data_loader.py --symbol USDCAD --timeframe M15 --bars 30000

# Alle Symbole, M30
python data_loader.py --symbol alle --timeframe M30
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `alle` | Einzelnes Symbol oder `alle` |
| `--timeframe` | `H1` | H1, M60, M30, M15 |
| `--bars` | `50000` | Anzahl zu ladender Kerzen |

---

### Paper-Trading starten (`live/live_trader.py`)

> ⚠️ MT5 Terminal muss geöffnet sein! Jedes Symbol in eigenem PowerShell-Fenster!

```powershell
# USDCAD – Regime 2 (Abwärtstrend), ATR-SL aktiv
python live\live_trader.py --symbol USDCAD --schwelle 0.50 --regime_filter 2 --atr_sl 1 --atr_faktor 1.5 --mt5_server "$env:MT5_SERVER" --mt5_login $env:MT5_LOGIN --mt5_password "$env:MT5_PASSWORD"

# USDJPY – Regime 1 (Aufwärtstrend), ATR-SL aktiv, Option B (optimiert)
python live\live_trader.py --symbol USDJPY --schwelle 0.55 --regime_filter 1 --atr_sl 1 --atr_faktor 1.5 --mt5_server "$env:MT5_SERVER" --mt5_login $env:MT5_LOGIN --mt5_password "$env:MT5_PASSWORD"
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `USDCAD` | Handelssymbol (USDCAD, USDJPY aktiv) |
| `--schwelle` | `0.52` | Mindest-Wahrscheinlichkeit für Trade (Test-Phase: 0.52) |
| `--regime_filter` | `0,1,2` | Regime-Nummern: 0=Seitwärts, 1=Aufwärts, 2=Abwärts, 3=Hohe Vola |
| `--atr_sl` | `1` | 1 = ATR-SL dynamisch, 0 = festes SL (0.3%) |
| `--atr_faktor` | `1.5` | ATR-Multiplikator für SL (empfohlen: 1.5) |
| `--lot` | `0.01` | Lot-Größe (NICHT erhöhen ohne Erfahrung!) |
| `--paper_trading` | `1` | 1 = Paper-Modus, 0 = Live (⚠️ echtes Geld!) |
| `--mt5_server` | `""` | Broker-Server |
| `--mt5_login` | `0` | Kontonummer |
| `--mt5_password` | `""` | Passwort |
| `--version` | `v1` | Modell-Version |
| `--timeframe` | `H1` | H1, M30, M15 |
| `--kill_switch_dd` | `0.15` | Max. Drawdown → automatischer Stopp (15%) |
| `--kapital_start` | `10000.0` | Startkapital für Kill-Switch im Paper-Modus |
| `--heartbeat_log` | `1` | 1 = Heartbeat pro Kerze loggen |
| `--allow_research_symbol` | `0` | 1 = andere Symbole erlauben (nur Paper!) |

---

### Paper-Trading – Sofort-Start (konkrete Werte, 2026-03-03)

> Fenster 1 und Fenster 2 gleichzeitig öffnen (beide mt5_trading Verzeichnis, venv aktiviert)

**Fenster 1 – USDJPY (Option B):**

```powershell
python live\live_trader.py --symbol USDJPY --schwelle 0.55 --regime_filter 1 --atr_sl 1 --atr_faktor 1.5 --mt5_server "$env:MT5_SERVER" --mt5_login $env:MT5_LOGIN --mt5_password "$env:MT5_PASSWORD"
```

**Fenster 2 – USDCAD (Regime 2, bewährt):**

```powershell
python live\live_trader.py --symbol USDCAD --schwelle 0.50 --regime_filter 2 --atr_sl 1 --atr_faktor 1.5 --mt5_server "$env:MT5_SERVER" --mt5_login $env:MT5_LOGIN --mt5_password "$env:MT5_PASSWORD"
```

| Symbol | Schwelle | Regime | Grund |
| ------ | -------- | ------ | ----- |
| USDJPY | 0.55 | 1 (Aufwärts) | Option B optimiert (Phase 5 Grid-Search) |
| USDCAD | 0.50 | 2 (Abwärts) | Bewährte Phase-5/6 Konfiguration (PF 1.355) |

> ⚠️ **Wichtig:** MT5 Terminal muss offen sein. Laufen beide Fenster, haben Sie Paper-Trading aktiv.

### Shadow-Compare starten (USDCAD v4 vs. USDJPY v5)

> Kontrollierter Rollout auf **Windows Laptop**: USDCAD bleibt auf stabiler v4, USDJPY testet v5.

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
.\start_shadow_compare.bat
```

**Voraussetzung (einmalig):**

```powershell
setx MT5_SERVER "SwissquoteLtd-Server"
setx MT5_LOGIN "<dein_login>"
setx MT5_PASSWORD "<dein_passwort>"
```

### Testphase starten (neue Top-Konfiguration, Two-Stage v4)

> Empfohlener Start für die aktuelle Testphase auf dem Windows-Laptop.

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
.\start_testphase_topconfig.bat
```

Verwendete Kernparameter:

- `--schwelle 0.54`
- `--regime_filter 0,1,2`
- `--two_stage_enable 1`
- `--two_stage_ltf_timeframe M5`
- `--two_stage_version v4`
- `--paper_trading 1`

### Alle Trader stoppen (vor Neustart empfohlen)

> Beendet alle bekannten Trader-Fenster und laufende `live_trader.py` Prozesse.

```powershell
cd "C:\Users\Sebastian Setnescu\mt5_trading"
stop_all_traders.bat
```

### Rollenklärung: Engine vs. Dashboard (wichtig)

| Komponente | Rolle | Nutzung |
| --- | --- | --- |
| `start_shadow_compare.bat` | **Engine** (Ausführung) | Startet die Python-Trader im Hintergrund, erzeugt Signale/CSV/Logs, führt Paper-Orders aus |
| `LiveSignalDashboard.mq5` | **Dashboard** (Beobachtung) | Liest CSV aus MT5 Common Files, zeigt Status/Freshness und zeichnet Entry/SL/TP/History im Chart |

**Merksatz:** `start_shadow_compare.bat` arbeitet – `LiveSignalDashboard.mq5` zeigt an.

**Empfehlung im Betrieb:**

1. `stop_all_traders.bat`
2. `start_shadow_compare.bat`
3. Dashboard-EA auf einem MT5-Chart aktivieren (doppelte Alerts vermeiden)

**Entwicklung:**
Single Source of Truth bleibt das Repo auf Linux (`live/mt5/LiveSignalDashboard.mq5`).
Änderung im Repo → deployen → in MT5 kompilieren/testen.

---

### Daten vom Laptop zum Linux-Server kopieren

```powershell
# Einzelne Datei
scp "data\USDCAD_H1.csv" "sebastian setnescu@192.168.1.19:/mnt/1T-Data/XGBoost-LightGBM/data/"

# Alle H1-Daten
scp data\*_H1.csv "sebastian setnescu@192.168.1.19:/mnt/1T-Data/XGBoost-LightGBM/data/"

# Alle M15-Daten
scp data\*_M15.csv "sebastian setnescu@192.168.1.19:/mnt/1T-Data/XGBoost-LightGBM/data/"
```

---

## 🐧 Linux Server – Befehle

### Virtuelle Umgebung aktivieren (Linux)

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
source .venv/bin/activate
```

### Feature Engineering (`features/feature_engineering.py`)

```bash
# Alle Symbole, H1
python features/feature_engineering.py --symbol alle

# Einzelnes Symbol, M15
python features/feature_engineering.py --symbol USDCAD --timeframe M15
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `alle` | Einzelnes Symbol oder `alle` |
| `--timeframe` | `H1` | H1, M60, M30, M15 |

---

### Labeling (`features/labeling.py`)

```bash
# Standard-Labeling (feste Barrieren 0.3%)
python features/labeling.py --symbol alle --version v1

# Asymmetrisches Labeling (Risk-Reward 2:1)
python features/labeling.py --symbol alle --tp_pct 0.006 --sl_pct 0.003 --version v3 --modus rrr

# ATR-basiertes Labeling (dynamisch)
python features/labeling.py --symbol alle --modus atr --atr_faktor 1.5 --version v4
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `alle` | Einzelnes Symbol oder `alle` |
| `--tp_pct` | `0.003` | Take-Profit (0.3%) |
| `--sl_pct` | `0.003` | Stop-Loss (0.3%) |
| `--horizon` | `5` | Zeitschranke in Kerzen |
| `--version` | `v1` | Versions-Suffix (v1, v2, v3, v4) |
| `--timeframe` | `H1` | H1, M60, M30, M15 |
| `--modus` | `standard` | `standard`, `rrr`, `atr` |
| `--atr_faktor` | `1.5` | ATR-Multiplikator (nur bei `--modus atr`) |

---

### Modell trainieren (`train_model.py`)

```bash
# Einzelnes Symbol trainieren (50 Optuna-Trials)
python train_model.py --symbol USDCAD --version v1 --trials 50

# Alle Symbole trainieren
python train_model.py --symbol alle --trials 50

# M15-Modell trainieren
python train_model.py --symbol USDCAD --version v1 --timeframe M15 --trials 50
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `EURUSD` | Einzelnes Symbol oder `alle` |
| `--trials` | `50` | Anzahl Optuna-Trials (mehr = besser, aber langsamer) |
| `--version` | `v1` | Modell-Versions-Suffix |
| `--timeframe` | `H1` | H1, M60, M30, M15, H4 |

---

### Walk-Forward-Analyse (`walk_forward.py`)

```bash
# Einzelnes Symbol
python walk_forward.py --symbol USDCAD --version v1

# Alle Symbole
python walk_forward.py --symbol alle
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `EURUSD` | Einzelnes Symbol oder `alle` |
| `--version` | `v1` | Versions-Suffix |
| `--timeframe` | `H1` | H1, M60, M30, M15 |

---

### Backtest (`backtest/backtest.py`)

```bash
# Standard-Backtest (festes SL)
python backtest/backtest.py --symbol USDCAD --schwelle 0.50 --regime_filter 2 --version v1

# ATR-SL Backtest (empfohlen!)
python backtest/backtest.py --symbol USDCAD --schwelle 0.50 --regime_filter 2 --atr_sl --atr_faktor 1.5

# Alle Symbole mit Kapital-Simulation
python backtest/backtest.py --symbol alle --atr_sl --kapital 10000 --risiko_pct 0.01

# Stress-Test: doppelte Spreads
python backtest/backtest.py --symbol USDCAD --spread_faktor 2.0 --atr_sl

# Bestimmter Zeitraum
python backtest/backtest.py --symbol USDCAD --zeitraum_von 2024-01-01 --zeitraum_bis 2024-12-31
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `EURUSD` | Ein oder mehrere Symbole, oder `alle` |
| `--schwelle` | `0.55` | Wahrscheinlichkeits-Schwelle |
| `--tp_pct` | `0.003` | Take-Profit (0.3%) |
| `--sl_pct` | `0.003` | Stop-Loss (0.3%) |
| `--regime_filter` | `None` | Regime-Filter (0, 1, 2, 3) |
| `--version` | `v1` | Daten-Version |
| `--model_version` | `None` | Modell-Version (falls abweichend) |
| `--horizon` | `5` | Zeitschranke in Kerzen |
| `--atr_sl` | `False` | ATR-basiertes SL aktivieren (Flag) |
| `--atr_faktor` | `1.5` | ATR-Multiplikator |
| `--kapital` | `0.0` | Startkapital (0 = deaktiviert) |
| `--risiko_pct` | `0.01` | Max. Risiko pro Trade (1%) |
| `--kontrakt_groesse` | `100000` | Kontraktgröße 1 Lot |
| `--zeitraum_von` | `None` | Startdatum (z.B. 2024-01-01) |
| `--zeitraum_bis` | `None` | Enddatum (z.B. 2024-12-31) |
| `--swap_aktiv` | `False` | Swap-Kosten einbeziehen (Flag) |
| `--spread_faktor` | `1.0` | Spread-Multiplikator für Stress-Test |
| `--timeframe` | `H1` | H1, M60, M30, M15, H4 |

---

### Retraining (`retraining.py`)

```bash
# Retraining bei schlechter Performance (automatisch)
python retraining.py --symbol USDCAD --sharpe_limit 0.5

# Retraining erzwingen
python retraining.py --symbol USDCAD --erzwingen --trials 50

# Alle Symbole retrainen
python retraining.py --symbol alle --erzwingen
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--symbol` | `EURUSD` | Einzelnes Symbol oder `alle` |
| `--erzwingen` | `False` | Retraining erzwingen (ignoriert Sharpe-Trigger) |
| `--sharpe_limit` | `0.5` | Sharpe < Limit → Retraining auslösen |
| `--trials` | `30` | Anzahl Optuna-Trials |
| `--toleranz` | `0.01` | F1-Toleranz (1%) – neues Modell nur deployed wenn besser |

---

### 📅 Sonntagsroutine – Wöchentlicher Check

```bash
# Jeden Sonntag ausführen (Linux Server, vom Projekt-Root):
cd /mnt/1Tb-Data/XGBoost-LightGBM
bash scripts/sunday_check.sh
```

> Führt automatisch aus: Server-Sync-Check → KPI-Report → Trades der Woche → Checkliste

---

### KPI-Report (`reports/weekly_kpi_report.py`)

```bash
# Wöchentlicher Report (mit korrektem Log-Pfad für paper_test128)
python reports/weekly_kpi_report.py --tage 7 --log_dir logs/paper_test128

# 14-Tage-Report
python reports/weekly_kpi_report.py --tage 14

# M15-Zeitrahmen
python reports/weekly_kpi_report.py --timeframe M15

# Two-Stage Shadow-Monitoring (M5 Trades)
python reports/weekly_kpi_report.py --tage 7 --timeframe M5_TWO_STAGE
```

| Argument | Standard | Beschreibung |
| -------- | -------- | ----------- |
| `--tage` | `7` | Zeitraum in Tagen |
| `--timeframe` | `H1` | H1, M60, M30, M15, H4, M5_TWO_STAGE |

> Bei `--timeframe M5_TWO_STAGE` liest der Report die Dateien
> `backtest/USDCAD_M5_two_stage_trades.csv` und
> `backtest/USDJPY_M5_two_stage_trades.csv`.

### Testphase-Auswertung (`scripts/evaluate_mt5_testphase.py`)

> Bewertet laufende MT5-Paper-Tests auf Basis von `logs/*_signals.csv` und
> `logs/*_closes.csv` (Freshness + Aktivität + KPI-Gates).

```bash
python scripts/evaluate_mt5_testphase.py --hours 48 --symbols USDCAD,USDJPY --timeframe M5
```

Artefakte:

- `reports/testphase/mt5_testphase_eval_<timestamp>.csv`
- `reports/testphase/mt5_testphase_eval_latest.csv`

---

## 🔧 Pipeline-Skripte (Linux Server)

### Komplette M15-Pipeline

```bash
bash run_m15_pipeline.sh
```

> Führt nacheinander aus: Feature Engineering → Labeling → Training → Walk-Forward → Backtest → KPI-Report (für USDCAD + USDJPY, M15)

### v2/v3-Pipeline

```bash
bash run_pipeline_v2_v3.sh
```

---

## 🚀 Deployment (Linux → Windows)

### Automatisch (alle Modelle + Skript)

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
bash deploy_to_laptop.sh
```

### Manuell (einzelne Dateien)

```bash
# Live-Trader übertragen
scp live/live_trader.py "sebastian setnescu@192.168.1.19:mt5_trading/live/live_trader.py"

# Einzelnes Modell übertragen
scp models/lgbm_usdcad_v1.pkl "sebastian setnescu@192.168.1.19:mt5_trading/models/"

# Alle v1-Modelle übertragen
scp models/lgbm_*_v1.pkl "sebastian setnescu@192.168.1.19:mt5_trading/models/"

# Requirements übertragen
scp requirements-laptop.txt "sebastian setnescu@192.168.1.19:mt5_trading/"
```

### SSH-Verbindung testen

```bash
ssh "sebastian setnescu@192.168.1.19" "echo OK"
```

---

## 📊 Referenz-Konfiguration (H1-Baseline, Stand: 2026-03-03)

| Symbol | Schwelle | Regime | ATR-SL | Sharpe | Rendite |
| ------ | -------- | ------ | ------ | ------ | ------- |
| USDCAD | 0.50 | 2 (Abwärtstrend) | 1.5× ATR | 2.118 | +2.95% |
| USDJPY | 0.50 | 1 (Aufwärtstrend) | 1.5× ATR | 1.263 | +5.94% |

> Hinweis: Diese Tabelle ist die historische H1-Baseline.
> Aktueller Fokus (2026-03-05) ist Shadow-Compare im Two-Stage-Setup (M5).

## 📉 Aktueller Two-Stage Status (Stand: 2026-03-05, Stress-Re-Run)

| Symbol | Version | Sharpe | PF | MaxDD | Status |
| ------ | ------- | ------ | -- | ----- | ------ |
| USDCAD | v5 | -23.002 | 0.011 | -164.30% | ❌ NO-GO |
| USDJPY | v5 | -2.140 | 0.692 | -34.36% | ❌ NO-GO |

> Konsequenz: Weiterhin **PAPER_ONLY** und Shadow-Compare-Laufzeit bewerten.

---

## 🔑 Regime-Nummern Referenz

| Nummer | Regime | Beschreibung |
| ------ | ------ | ----------- |
| 0 | Seitwärts | Kein klarer Trend |
| 1 | Aufwärtstrend | Bullish |
| 2 | Abwärtstrend | Bearish |
| 3 | Hohe Volatilität | Unruhiger Markt |

---

## 📁 Wichtige Pfade

| Was | Linux Server | Windows Laptop |
| --- | ------------ | -------------- |
| Projektordner | `/mnt/1T-Data/XGBoost-LightGBM/` | `C:\Users\Sebastian Setnescu\mt5_trading\` |
| venv | `.venv/bin/activate` | `venv\Scripts\activate` |
| Modelle | `models/` | `models/` |
| Daten (CSV) | `data/` | `data/` |
| Logs | – | `logs/` |
| Backtest-Ergebnisse | `backtest/` | – |
| Reports | `reports/` | – |

---

## Stand

Letzte Aktualisierung: 2026-03-05
