# 🗺️ MT5 ML-Trading-System – Roadmap

> **Ziel:** Ein automatisches Handelssystem mit XGBoost/LightGBM + Regime-Detection, das in MetaTrader 5 live handelt.

---

## 🖥️ System-Architektur

| Gerät | Rolle | Was läuft hier? |
| --- | --- | --- |
| **Windows 11 Laptop** | MT5-Host & Live-Trading | MT5 Terminal, `MetaTrader5` Python-Lib, `live_trader.py`, Paper-Trading |
| **Linux Server (1TB SSD)** | Datenspeicher & Training | Rohdaten (CSV), Modelle (.pkl), `train_model.py`, `backtest.py` |
| **VS Code Remote SSH** | Entwicklung | Code wird auf dem Linux-Server bearbeitet und ausgeführt |

> **Hinweis:** `data_loader.py` und `live_trader.py` müssen auf dem **Windows 11 Laptop** laufen, da die `MetaTrader5`-Bibliothek eine laufende MT5-Instanz auf demselben Rechner benötigt. Alle anderen Skripte laufen auf dem Linux-Server.

---

## 📌 Wie du diese Roadmap benutzt

- Hake jede Aufgabe ab, wenn sie erledigt ist: `[ ]` → `[x]`
- Gehe **niemals** zur nächsten Phase, bevor die aktuelle abgeschlossen ist
- Bei jedem Schritt kannst du den **KI-Assistenten** mit dem beigelegten Prompt um Hilfe bitten

---

## ✅ Wichtige Tipps & Checkliste vor Projektstart

### 🌐 Plattform- & Deployment-Checkliste

- [x] MetaTrader 5 auf Windows 11 Laptop installiert ✅
- [x] VS Code Remote SSH zum Linux-Server eingerichtet ✅
- [x] `vectorbt`, `lightgbm`, `xgboost` auf Linux-Server (ARM) installieren und Testimport prüfen ✅
- [x] Python-Umgebung auf Windows 11 Laptop einrichten (für `data_loader.py` und `live_trader.py`) ✅
- [ ] Pfad- und Rechte-Management beachten (absolute Pfade, Dateirechte, Umgebungsvariablen)
- [ ] Deployment-Skripte plattformunabhängig gestalten (z.B. mit Python `os`/`pathlib`)
- [ ] Dokumentation zu Systemvoraussetzungen und Setup für beide Plattformen pflegen

### 📋 Allgemeine Qualitäts-Checkliste

- [ ] Datenqualität prüfen (keine NaN-Werte, richtige Zeitzonen, vollständige Historie)
- [ ] Feature-Engineering ohne Look-Ahead-Bias (z.B. `.shift(1)` bei Rolling-Features)
- [ ] Zeitliche Datenaufteilung: Training → Validierung → Test (niemals zufällig!)
- [ ] Test-Set nur einmal am Ende verwenden (nicht für Feature- oder Modell-Auswahl)
- [ ] Paper-Trading-Modus zuerst, kein echtes Geld am Anfang
- [ ] Risikomanagement: Stop-Loss setzen, mit kleinen Lots starten (z.B. 0.01 Lot)
- [ ] Schritte und Entscheidungen dokumentieren (für Debugging und Nachvollziehbarkeit)
- [ ] Unit-Tests für zentrale Funktionen schreiben
- [ ] Modell-Erklärbarkeit prüfen (Feature Importance, SHAP)
- [ ] CI/CD für automatisierte Tests und Linting einrichten (z.B. GitHub Actions)

---

## ⚙️ PHASE 0 – Vorbereitung (vor Phase 1!)

**Ziel:** Stabiles Fundament für das gesamte Projekt legen

### Plattform-Test

- [x] MT5 auf Windows 11 Laptop vorinstalliert – `MetaTrader5` Python-Bibliothek funktioniert ✅
- [x] `vectorbt`, `lightgbm`, `xgboost` auf Linux-Server (ARM) installieren und Testimport prüfen ✅ (alle 12 Bibliotheken OK)
- [x] Python-Umgebung auf Windows 11 Laptop einrichten (`python -m venv venv`, `pip install -r requirements-laptop.txt`) ✅

### Versionskontrolle

- [x] Git-Repository initialisieren (`git init`) ✅
- [x] `.gitignore` anlegen (Modelle, Daten, API-Keys, `venv/` ausschließen) ✅
- [x] Ersten Commit erstellen (nur Dokumentation) ✅
- [x] Remote-Repository auf GitHub anlegen und verbinden ✅ ([mt5-ml-trading](https://github.com/sebastianstn/mt5-ml-trading))

### Projektfundament

- [x] `requirements-server.txt` für Linux-Server anlegen: ✅

  ```text
  # Linux-Server: Training, Backtesting, Feature Engineering
  pandas>=2.0.0
  numpy>=1.24.0
  pandas_ta>=0.3.14b
  xgboost>=2.0.0
  lightgbm>=4.0.0
  scikit-learn>=1.3.0
  vectorbt>=0.26.0
  optuna>=3.4.0
  python-dotenv>=1.0.0
  joblib>=1.3.0
  matplotlib>=3.7.0
  shap>=0.43.0
  ```

- [x] `requirements-laptop.txt` für Windows 11 Laptop anlegen: ✅

  ```text
  # Windows 11 Laptop: MT5-Verbindung, Live-Trading
  MetaTrader5>=5.0.45
  pandas>=2.0.0
  numpy>=1.24.0
  pandas_ta>=0.3.14b
  python-dotenv>=1.0.0
  joblib>=1.3.0
  ```

- [x] `.env.example` Template für API-Keys anlegen ✅
- [x] `python-dotenv` für sicheres Laden der API-Keys einrichten ✅
- [x] `README.md` mit Projektbeschreibung und Setup-Anleitung erstellen ✅
- [x] `.env` mit echten API-Keys befüllen (`cp .env.example .env`) ✅

**✅ Phase 0 abgeschlossen, wenn:** Git-Repo existiert, alle Bibliotheken importieren ohne Fehler, `.gitignore` und `.env` sind eingerichtet.

---

## ✅ PHASE 1 – Umgebung & Datenbeschaffung (abgeschlossen)

**Ziel:** Funktionierende Entwicklungsumgebung + erste Daten aus MT5

### Setup

- [x] Virtuelle Umgebung erstellt (`python -m venv venv`) ✅
- [x] Abhängigkeiten installiert: ✅

  **Linux-Server** – alle Pakete in `venv/` vorhanden:
  `pandas`, `numpy`, `pandas_ta`, `xgboost`, `lightgbm`, `scikit-learn`,
  `vectorbt`, `optuna`, `python-dotenv`, `joblib`, `matplotlib`, `seaborn`,
  `black`, `flake8`, `pytest`

  **Windows 11 Laptop** – Pakete in separatem `venv/` installiert:
  `MetaTrader5`, `pandas`, `numpy`, `pandas_ta`, `python-dotenv`
  (Hinweis: `pandas_ta` mit `--no-deps` wegen Python 3.14 / numba-Inkompatibilität)

- [x] Projektordner-Struktur angelegt: ✅

  ```text
  /mnt/1T-Data/XGBoost-LightGBM/
  ├── .github/
  ├── data/               # Rohdaten & Feature-CSVs
  ├── features/           # feature_engineering.py, regime_detection.py, labeling.py
  ├── models/             # 14 gespeicherte Modelle (.pkl)
  ├── backtest/
  ├── live/
  ├── notebooks/
  ├── plots/              # Regime- und Feature-Importance-Charts
  ├── tests/              # Unit-Tests
  ├── .env
  ├── .gitignore
  ├── requirements-server.txt
  ├── requirements-laptop.txt
  └── README.md
  ```

- [x] Linting und Code-Formatierung eingerichtet: ✅
  - `black 26.1.0` installiert – alle `.py`-Dateien formatiert
  - `flake8 7.2.0` installiert
- [x] `tests/`-Ordner angelegt + erste Test-Datei erstellt: ✅
  - `tests/test_features.py` – 7 Unit-Tests für `double_barrier_label`
  - Alle 7 Tests bestehen (`pytest tests/ -v`)

### Datenbeschaffung

- [x] `data_loader.py` geschrieben (läuft auf Windows Laptop mit MT5) ✅
- [x] 8 Jahre historische Daten geladen (2018–2026, alle 7 Forex-Hauptpaare) ✅
- [x] Daten als CSV gespeichert und geprüft (keine NaN-Werte, OHLC-Logik OK) ✅

**✅ Phase 1 abgeschlossen:** Umgebung läuft, Struktur steht, 7 × ~49.000 Kerzen als CSV verfügbar.

---

## ✅ PHASE 2 – Feature Engineering (abgeschlossen)

**Ziel:** Aus Rohdaten aussagekräftige Merkmale für das Modell erzeugen

### Technische Indikatoren

- [x] `feature_engineering.py` erstellt ✅
- [x] **Trend-Features:** SMA 20/50/200, EMA 12/26, MACD ✅
- [x] **Momentum-Features:** RSI (14), Stochastic, Williams %R ✅
- [x] **Volatilitäts-Features:** Bollinger Bands, ATR ✅
- [x] **Volumen-Features:** OBV (On-Balance Volume), Volume Rate of Change ✅

### Erweiterte Features

- [x] **Multi-Timeframe:** H4- und D1-Trend als Feature in H1-Daten einbauen ✅
- [x] **Order Flow:** Funding Rate und Open Interest aus Binance API ziehen ✅
  - BTC Funding Rate (8h) + BTC Open Interest (1h) als Risk-On/Off Proxy
  - Skript: `features/order_flow.py`, Output: `data/btc_funding_rate.csv`, `data/btc_open_interest.csv`
- [x] **Sentiment:** Fear & Greed Index täglich laden und als Feature einbauen ✅
  - Alternative.me API (kostenlos, kein Key), täglich → H1 forward-fill
  - Skript: `features/order_flow.py`, Output: `data/fear_greed.csv`
  - Neue Features: `fear_greed_value`, `fear_greed_class`, `btc_funding_rate`, `btc_oi_change`, `btc_oi_zscore`

### Datenqualität

- [x] Feature-DataFrame als CSV exportieren (7× SYMBOL_H1_features.csv, ~49.000 Kerzen, 56 Features) ✅
- [x] Feature-Korrelationsmatrix prüfen (hoch korrelierte Features entfernen) ✅
  - Skript: `features/correlation_analysis.py`
  - Output: `plots/correlation_matrix.png`, `plots/high_correlation_pairs.png`, `reports/feature_analysis.txt`
  - Ergebnis: Tree-basierte Modelle tolerieren hohe Korrelation gut (keine Pflicht zum Entfernen)
- [x] Alle Features normalisieren / skalieren wo nötig ✅
  - Ergebnis: XGBoost/LightGBM benötigen KEINE Normalisierung (baumbasiert)
  - Dokumentiert in `reports/feature_analysis.txt` inkl. Code-Beispiel für StandardScaler

**✅ Phase 2 abgeschlossen:** 7 Währungspaare × 56 Features, keine NaN-Werte.

---

## ✅ PHASE 3 – Regime Detection (abgeschlossen)

**Ziel:** Marktphasen automatisch erkennen (Trend ↑, Trend ↓, Seitwärts, Volatil)

### Statistische Methode

- [x] `regime_detection.py` erstellt ✅
- [x] Volatilität: ATR% vs. rollender Median(50) ✅
- [x] Trendstärke: ADX(14) ✅
- [x] Regelbasierte Regime-Klassifikation (Priorität: Vola > Trend > Seitwärts): ✅

  ```
  0 = Seitwärts         (~53–57%)
  1 = Aufwärtstrend     (~17–21%)
  2 = Abwärtstrend      (~21–25%)
  3 = Hohe Volatilität  (~2–4%)
  ```

- [x] `market_regime` + `adx_14` als neue Spalten in alle 7 Feature-CSVs eingefügt ✅

### Validierung

- [x] Regime-Verteilung geprüft – alle Klassen vorhanden, keine Dominanz >60% ✅
- [x] Visualisierung erstellt (plots/SYMBOL_regime.png für alle 7 Paare) ✅

**✅ Phase 3 abgeschlossen:** 7 Paare × 58 Features (inkl. market_regime + adx_14), ~48.960 Kerzen.

---

## 🔴 PHASE 4 – Labeling & Modelltraining

**Ziel:** Modell trainieren, das Kauf-/Verkaufssignale vorhersagt

---
**Empfohlene Umsetzungsidee:**

- Double-Barrier-Labeling mit 5-Barren-Horizont, Take-Profit 1%, Stop-Loss 0,5% (an Markt anpassen)
- Label: 1 (TP erreicht), -1 (SL erreicht), 0 (weder noch nach 5 Barren)
- Features: Technische Indikatoren, Order-Flow-Daten, Regime-Indikator (z.B. Hidden-Markov-Modell oder Volatilitäts-/Trendfilter)
- Modelltraining: LightGBM-Klassifikator mit `multi_logloss` als Zielfunktion, Hyperparameter-Optimierung mit Optuna
- System ist adaptiv durch Regime-Erkennung und hat Risikomanagement direkt integriert

---

### Labeling

- [x] `labeling.py` erstellt ✅
- [x] **Double-Barrier** mit TP=SL=0.3%, Horizon=5 H1-Barren ✅
- [x] Label-Verteilung geprüft: Long ~11-22%, Short ~9-24%, Neutral ~55-82% ✅

### Datenaufteilung (zeitlich!)

- [x] Training:    2018-04 bis 2021-12 (~23.000 Kerzen) ✅
- [x] Validierung: 2022 (~6.250 Kerzen) ✅
- [x] Test:        2023-01 bis 2026-02 (~19.500 Kerzen – HEILIG, nicht anfassen!) ✅

### Modelltraining (EURUSD)

- [x] `train_model.py` erstellt ✅
- [x] **XGBoost** Baseline: F1-Macro = 0.4452 ✅
- [x] **LightGBM** Baseline: F1-Macro = 0.4303 ✅
- [x] **XGBoost** Optuna (50 Trials): F1-Macro = 0.4810 ✅
- [x] **LightGBM** Optuna (50 Trials): F1-Macro = **0.4830** ← Bestes Modell ✅
- [x] Modelle gespeichert: 14 Modelle (XGBoost + LightGBM × 7 Symbole) ✅
- [x] Schwellenwert für Trade-Ausführung festlegen – `schwellenwert_analyse()` in `train_model.py` ✅

### Walk-Forward-Analyse

- [x] `walk_forward.py` erstellen ✅
- [x] 5 Expanding Windows (wachsendes Training, 6-Monate-Test-Block, 2019–2022) ✅
- [x] F1-Score pro Fenster aufgezeichnet ✅
- [x] Alle 7 Modelle stabil – kein Fenster > 0.10 unter dem Durchschnitt ✅

  ```
  EURUSD  Ø=0.4188  Min=0.3709  Schwankung=0.0748  ✅ STABIL
  GBPUSD  Ø=0.4681  Min=0.4384  Schwankung=0.0522  ✅ STABIL
  USDJPY  Ø=0.3988  Min=0.3513  Schwankung=0.1519  ✅ STABIL
  AUDUSD  Ø=0.4175  Min=0.3994  Schwankung=0.0410  ✅ STABIL
  USDCAD  Ø=0.4337  Min=0.3839  Schwankung=0.0772  ✅ STABIL
  USDCHF  Ø=0.4290  Min=0.4011  Schwankung=0.0641  ✅ STABIL
  NZDUSD  Ø=0.3943  Min=0.3601  Schwankung=0.0673  ✅ STABIL
  ```

### Tests & Modell-Erklärbarkeit (Modelltraining)

- [x] Modell mit SHAP erklären – `features/shap_analysis.py` erstellt ✅

**✅ Phase 4 (Kerntraining + Walk-Forward) abgeschlossen:** Alle 7 Modelle stabil, LightGBM F1-Macro Ø=0.42–0.47.

---

## 🟣 PHASE 5 – Backtesting

**Ziel:** Realistische Simulation des Systems auf historischen Daten

### Backtesting mit VectorBT

- [x] `backtest.py` erstellen ✅
- [x] Modellsignale in Buy/Sell-Orders umwandeln ✅
- [x] Backtest mit Double-Barrier-Regeln: fester SL/TP wie beim Labeling, nur Trades mit hoher Modellwahrscheinlichkeit ✅
- [x] Spread, Slippage und Kommission einrechnen ✅
- [x] Simulation durchlaufen ✅

### Risikomanagement (Details)

- [ ] Dynamische Positionsgrößenberechnung (z.B. Risiko pro Trade max. 1% des Kapitals)
- [ ] Dynamisches Stop-Loss-Management (z.B. ATR-basiert)
- [ ] Backtest auf verschiedene Märkte/Zeiträume ausweiten

### Auswertung

- [x] **Kennzahlen berechnen:** Gesamtrendite, Sharpe Ratio, Max. Drawdown, Gewinnfaktor, Anzahl Trades ✅
- [x] **Performance nach Regime analysieren:** Rendite + Win-Rate pro Regime ✅
- [x] **Monatliche Performance** als Heatmap darstellen ✅

**✅ Phase 5 abgeschlossen, wenn:** Sharpe >1.0 und Drawdown <20% auf dem Test-Set (2023+).

---

## ⚪ PHASE 6 – Live-Integration (MT5)

**Ziel:** System läuft automatisch auf dem Windows 11 Laptop und handelt live

### Infrastruktur

- [x] MT5 Terminal auf Windows 11 Laptop installiert ✅
- [ ] Python-Umgebung auf Windows 11 Laptop vollständig einrichten
- [ ] Trainiertes Modell (`.pkl`) vom Linux-Server auf den Laptop übertragen (z.B. per `scp` oder freigegebener Pfad)
- [ ] Laptop-Schlaf/Ruhemodus deaktivieren während Live-Trading läuft
- [ ] ⚠️ Für 24/7-Betrieb langfristig Windows-VPS in Betracht ziehen (Contabo, Vultr – ab ~5 €/Monat)

### CI/CD & Monitoring

- [ ] Automatisiertes Deployment (Modell-Update vom Server auf Laptop) einrichten
- [ ] Health-Checks und automatisierte Neustarts bei Fehlern

### Live-Skript

- [ ] `live_trader.py` erstellen mit folgendem Ablauf:

  ```
  Jede neue Kerze:
  1. Neue Daten von MT5 holen
  2. Features berechnen
  3. Regime erkennen
  4. Modell-Vorhersage machen
  5. Order senden (falls Signal)
  6. Risikomanagement prüfen (Max. Lots, Stop-Loss)
  ```

- [ ] Logging einbauen (jede Aktion in Datei schreiben)
- [ ] Error-Handling: Was passiert bei Verbindungsabbruch?
- [ ] Paper-Trading Modus (kein echtes Geld) 2 Wochen laufen lassen

**✅ Phase 6 abgeschlossen, wenn:** System läuft 2 Wochen stabil auf Windows 11 Laptop ohne Absturz im Paper-Trading.

---

## ⚫ PHASE 7 – Überwachung & Wartung

**Ziel:** System langfristig stabil und profitabel halten

### Monitoring

- [ ] Tägliche Performance-E-Mail einrichten (Python + SMTP)
- [ ] Alert bei Drawdown >10% (System pausieren)
- [ ] Modell-Drift überwachen (Accuracy auf Live-Daten wöchentlich prüfen)

### Code-Qualität & Wartung

- [ ] Automatisierte Tests regelmäßig laufen lassen (CI)
- [ ] Code- und Modell-Dokumentation aktuell halten

### Retraining

- [ ] Automatisches wöchentliches Retraining-Skript einrichten
- [ ] Neues Modell wird nur deployed, wenn Walk-Forward besser als altes Modell
- [ ] Versionierung der Modelle (z.B. `model_v1.pkl`, `model_v2.pkl`)

**✅ Phase 7 abgeschlossen, wenn:** System läuft 3+ Monate autonom mit positivem Ergebnis.

---

## 📊 Fortschritts-Übersicht

| Phase | Beschreibung | Status |
|-------|-------------|--------|
| 0 | Vorbereitung (Git, .env, Bibliothekstest) | ✅ Abgeschlossen |
| 1 | Umgebung & Daten | ✅ Abgeschlossen |
| 2 | Feature Engineering | ✅ Abgeschlossen |
| 3 | Regime Detection | ⬜ Offen |
| 4 | Labeling & Training | ⬜ Offen |
| 5 | Backtesting | ⬜ Offen |
| 6 | Live-Integration | ⬜ Offen |
| 7 | Wartung | ⬜ Offen |
| Q | Code-Qualität & CI/CD | ⬜ Offen |

> Status: ⬜ Offen | 🔄 In Arbeit | ✅ Abgeschlossen

---

Letzte Aktualisierung: 2026-02-26 – Phase 2 vollständig abgeschlossen
