# MT5 ML-Trading-System

Automatisches Trading-System mit **LightGBM/XGBoost** + Regime-Detection für MT5.

Aktueller Betriebsmodus: **Paper-Trading aktiv** (USDCAD + USDJPY).

---

## System-Architektur

| Gerät | Rolle | Was läuft hier? |
| --- | --- | --- |
| **Windows 11 Laptop** | MT5-Host & Paper-Betrieb | MT5 Terminal, `live/live_trader.py`, Dashboard/Sync |
| **Linux Server (1TB SSD)** | Training & Auswertung | Feature-Engineering, Training, Backtests, KPI/Reports |
| **VS Code Remote SSH** | Entwicklung | Codebearbeitung auf dem Linux-Server |

> Wichtig: `MetaTrader5` funktioniert nur auf Windows mit laufendem MT5-Terminal.

---

## Schnellstart

### Linux-Server (Training/Backtest)

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
pytest tests -q
```

### Windows-Laptop (Paper-Trading)

```bash
cd C:\Users\<USER>\mt5_trading
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-laptop.txt
```

Danach Trader starten (Paper-Modus):

- `USDCAD`: Schwelle 0.60, Regime `1,2`
- `USDJPY`: Schwelle 0.60, Regime `1`

Details inkl. Dashboard-Sync: `live/mt5/README_MT5_Dashboard.md`.
Deploy-Ablauf Server → Laptop: `reports/deploy_server_to_laptop.md`.

---

## Projektstruktur

```text
/mnt/1T-Data/XGBoost-LightGBM/
├── data/                    # Rohdaten, Features, Labels
├── features/                # Feature-/Label-/Regime-Pipelines
├── models/                  # Modellartefakte (.pkl)
├── backtest/                # Backtesting-Skripte + Resultate
├── live/                    # Live/Paper-Trading + MT5-Integration
├── reports/                 # KPI-, Plan- und Incident-Dokumente
├── tests/                   # Unit-Tests
├── train_model.py
├── retraining.py
├── walk_forward.py
└── Roadmap.md
```

---

## Phasenstatus (aktuell)

| Phase | Beschreibung | Status |
| --- | --- | --- |
| 0 | Vorbereitung | ✅ |
| 1 | Umgebung & Datenbeschaffung | ✅ |
| 2 | Feature Engineering | ✅ |
| 3 | Regime Detection | ✅ |
| 4 | Labeling & Modelltraining | ✅ |
| 5 | Backtesting | ✅ |
| 6 | Live-Integration (Paper-Betrieb) | ✅ |
| 7 | Überwachung & Wartung | 🔄 |

Details: `Roadmap.md`.

---

## Operative Leitplanken

- **Keine Zukunftsdaten** in Features (Look-Ahead-Bias verhindern, Rolling mit `.shift(1)`).
- **Zeitliche Splits** statt zufälligem Split für Time-Series.
- **Paper first**: Echtgeld erst nach stabilen KPI-Gates.
- **Stop-Loss Pflicht** + Kill-Switch aktiv.
- **Aktive Paare aktuell nur:** `USDCAD`, `USDJPY`.

---

## Nächster Fokus

Phase 7 sauber ausführen:

1. Daily Ops diszipliniert pflegen (`reports/daily_ops_checklist_w1.md`)
2. Wöchentlichen KPI-Report erzeugen (`reports/weekly_kpi_report.py`)
3. 12 konsekutive GO-Wochen für Eskalationsfreigabe aufbauen
4. Dokumente regelmäßig mit Guard prüfen (`reports/doc_drift_guard.py` + `reports/doc_sync_checklist.md`)

---

## Automatischer Doc-Guard vor Commit

Damit der Doc-Drift-Guard bei jedem Commit automatisch läuft:

- Hook-Template: `.githooks/pre-commit`
- Hook-Template (PowerShell): `.githooks/pre-commit.ps1`
- Installer: `scripts/install_pre_commit_hook.sh`
- Installer (PowerShell): `scripts/install_pre_commit_hook.ps1`

Einmalig installieren (Linux-Server):

- `bash scripts/install_pre_commit_hook.sh`

Einmalig installieren (Windows PowerShell):

- `powershell -ExecutionPolicy Bypass -File .\scripts\install_pre_commit_hook.ps1`

Danach prüft Git bei jedem Commit automatisch `reports/doc_drift_guard.py`.
