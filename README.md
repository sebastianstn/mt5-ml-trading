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

- Shadow-Compare starten (Windows): `start_shadow_compare.bat`
- Setup aktuell: `USDCAD` als stabile Kontrolle (`v4`) vs. `USDJPY` als Kandidat (`v5`)
- Wichtig: `MT5_SERVER`, `MT5_LOGIN`, `MT5_PASSWORD` müssen gesetzt sein

Details inkl. Dashboard-Sync: `live/mt5/README_MT5_Dashboard.md`.
Deploy-Ablauf Server → Laptop: `reports/deploy_server_to_laptop.md`.

### Threshold-KPI-Auswertung (Linux-Server)

Vergleich der Signal-Schwellen `0.50`, `0.55`, `0.60` auf Basis von `logs/*_live_trades.csv`:

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
source .venv/bin/activate
python scripts/evaluate_threshold_kpis.py --symbols USDCAD,USDJPY --hours 72 --thresholds 0.50,0.55,0.60 --threshold_operator ge
```

Artefakte:

- `reports/threshold_eval/summary.csv` (Ranking pro Symbol/Threshold)
- `reports/threshold_eval/detail.csv` (Regime-Details pro Symbol/Threshold)

Hinweis zur Vergleichbarkeit:

- `--threshold_operator ge` entspricht `>=` (Live-Logik)
- `--threshold_operator gt` entspricht `>` (Backtest-Logik)

Wichtig für Entry-Logik im Live-Trader:

- Empfohlenes Mapping: `--decision_mapping long_prob --schwelle 0.55 --short_schwelle 0.45`
- Das entspricht: Long nur bei `prob_long >= 0.55`, Short nur bei `prob_long <= 0.45`

### Probability-Kalibrierung (Linux-Server)

Kalibrierung prüfen/erzeugen (Platt oder Isotonic), damit Schwellen wie `0.55` belastbarer sind:

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
source .venv/bin/activate
python scripts/calibrate_probabilities.py \
  --model_path models/lgbm_usdjpy_v4.pkl \
  --data_csv data/processed/USDJPY_H1_regime_labelled.csv \
  --label_col label \
  --method sigmoid \
  --output_model models/lgbm_usdjpy_v4_calibrated_sigmoid.pkl \
  --report_path reports/threshold_eval/calibration_report_usdjpy_sigmoid.json
```

### Demo-Micro-Reactive Modus (Windows-Laptop)

Wenn du im Demo-Betrieb bewusst mehr Bewegung/mehr Trades sehen willst:

- `start_demo_micro_reactive.bat` starten
- Dieser Modus setzt aggressivere Parameter:
  - `--decision_mapping long_prob`
  - `--schwelle 0.52 --short_schwelle 0.48`
  - `--regime_source market_regime_hmm`
  - `--regime_filter 0,1,2,3` (alle Regimes)
  - `--two_stage_kongruenz 0` (Filter aus, mehr Durchsatz)

Stufen für noch mehr Aktivität (nur Demo/Paper):

1. Stufe A: `0.55 / 0.45` (konservativ)
2. Stufe B: `0.52 / 0.48` (reaktiv, aktueller Demo-Micro-Mode)
3. Stufe C: `0.50 / 0.50` (maximal aktiv, viele Trades)

Turbo-Testplan (30-60 Minuten):

1. Starte `start_demo_micro_reactive.bat` (Stufe B) und beobachte:
   - Dashboard `Dir/Prob` Wechsel
   - Anzahl Alerts/Signale
   - Einträge in `logs/*_live_trades.csv`
2. Wenn zu wenig Aktivität: stoppe beide Trader und starte `start_demo_turbo_max.bat` (Stufe C).
3. Nach dem Test wieder auf konservative Settings zurückgehen (`start_shadow_compare.bat`), bevor du KPI-Entscheidungen triffst.

### Betriebsrollen: `start_shadow_compare.bat` vs. `LiveSignalDashboard.mq5`

Beide laufen parallel, haben aber **unterschiedliche Aufgaben**:

| Komponente | Hauptrolle | Was sie konkret macht | Ohne diese Komponente... |
| --- | --- | --- | --- |
| `start_shadow_compare.bat` (Windows/PowerShell) | **Engine / Ausführung** | Startet `live_trader.py` Prozesse, holt Marktdaten, berechnet Features/Regime, erzeugt Signale, schreibt Logs/CSV, führt Paper-Orders aus | entstehen keine neuen Signale/CSV-Updates |
| `LiveSignalDashboard.mq5` (MT5-Chart) | **Monitoring / Visualisierung** | Liest CSV aus MT5-Common-Files, zeigt Status (`CONNECTED/STALE/LIVE_*`), zeichnet Entry/SL/TP/History im Chart | läuft Trading weiter, aber Transparenz im Chart fehlt |

**Merksatz:**
`start_shadow_compare.bat` = *arbeitet* · `LiveSignalDashboard.mq5` = *zeigt an*.

#### Empfohlener Ablauf im Betrieb (Paper)

1. `stop_all_traders.bat` ausführen (sauberer Zustand)
2. `start_shadow_compare.bat` starten (Engine läuft)
3. MT5 offen lassen, `LiveSignalDashboard.mq5` auf **einem** Chart aktivieren
4. Status/Freshness prüfen (`CONNECTED`, `STALE>...`)
5. KPI/Logs regelmäßig auswerten

> Hinweis: Das Dashboard möglichst nur auf einem Chart aktiv lassen, um doppelte Alerts zu vermeiden.

#### Entwicklung: Was ist maßgebend?

**Single Source of Truth ist das Repository auf dem Linux-Server**
(`live/mt5/LiveSignalDashboard.mq5`).
Änderungen dort machen → deployen → in MT5 kompilieren/testen.
Direkte Hotfixes im MetaEditor danach immer zurück ins Repo übernehmen.

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
| 7 | Überwachung & Wartung | 🔄 (Paper-Trading aktiv, KPI-Gates laufen) |
| 7A | Option 1 Migration (H1/M5 Two-Stage) | 🔄 (Shadow vorbereitet, Start auf Laptop ausstehend) |
| 7B | SMC/MTF-Gaps | 🔄 (Schritte 1–4 abgeschlossen, Step 5 Laufzeit offen) |

**Aktueller Sub-Fokus:** Test-Phase Option 1 (Zwei-Stufen-Modell HTF/LTF).  
Details: `Roadmap.md`, Recherche: `reports/HTF_LTF_Strategie_Recherche.md`.

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

1. Shadow-Compare auf dem Windows-Laptop starten (`start_shadow_compare.bat`)
2. Wöchentlichen KPI-Report für Two-Stage erzeugen (`reports/weekly_kpi_report.py --timeframe M5_TWO_STAGE`)
3. 2 Wochen Shadow-Logs sammeln und v4/v5 vergleichen
4. Bis dahin strikt **PAPER_ONLY** (keine Eskalation auf Echtgeld)
5. Dokumente regelmäßig mit Guard prüfen (`reports/doc_drift_guard.py` + `reports/doc_sync_checklist.md`)

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
