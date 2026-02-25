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

## 🔵 PHASE 1 – Umgebung & Datenbeschaffung

**Ziel:** Funktionierende Entwicklungsumgebung + erste Daten aus MT5

### Setup

- [ ] Virtuelle Umgebung erstellen (`python -m venv venv`)
- [ ] Abhängigkeiten installieren:

  **Linux-Server** (Training, Backtesting, Feature Engineering):

  ```bash
  pip install pandas numpy pandas_ta xgboost lightgbm scikit-learn vectorbt optuna python-dotenv
  ```

  **Windows 11 Laptop** (MT5-Verbindung, Live-Trading):

  ```bash
  pip install MetaTrader5 pandas numpy pandas_ta python-dotenv
  ```

- [ ] Projektordner-Struktur anlegen:

  ```
  mt5_ml_trading/
  ├── .github/
  │   └── copilot-instructions.md
  ├── data/               # Rohdaten (CSV-Dateien)
  ├── features/           # Feature-Engineering Skripte
  │   └── feature_engineering.py
  ├── models/             # Gespeicherte Modelle (.pkl)
  ├── backtest/           # Backtesting Skripte
  │   └── backtest.py
  ├── live/               # Live-Trading Skripte
  │   └── live_trader.py
  ├── notebooks/          # Jupyter Notebooks zum Experimentieren
  ├── tests/              # Unit-Tests
  ├── .env                # API-Keys (niemals in Git!)
  ├── .gitignore
  ├── requirements.txt
  └── README.md
  ```

- [ ] Linting und Code-Formatierung einrichten (`black`, `flake8`)
- [ ] `tests/`-Ordner anlegen und erste Test-Datei erstellen

### Datenbeschaffung

- [ ] `data_loader.py` schreiben – verbindet sich mit MT5 und lädt OHLCV-Daten
- [ ] Mindestens 5 Jahre historische Daten laden (z.B. EURUSD H1)
- [ ] Daten als CSV speichern und prüfen (keine NaN-Werte, korrektes Datumsformat)

**✅ Phase 1 abgeschlossen, wenn:** Virtuelle Umgebung läuft, Projektstruktur steht, historische Daten als CSV gespeichert und geprüft.

---

## 🟡 PHASE 2 – Feature Engineering

**Ziel:** Aus Rohdaten aussagekräftige Merkmale für das Modell erzeugen

### Technische Indikatoren

- [ ] `feature_engineering.py` erstellen
- [ ] **Trend-Features:** SMA 20/50/200, EMA 12/26, MACD
- [ ] **Momentum-Features:** RSI (14), Stochastic, Williams %R
- [ ] **Volatilitäts-Features:** Bollinger Bands, ATR
- [ ] **Volumen-Features:** OBV (On-Balance Volume), Volume Rate of Change

### Erweiterte Features

- [ ] **Multi-Timeframe:** H4- und D1-Trend als Feature in H1-Daten einbauen
- [ ] **Order Flow:** Funding Rate und Open Interest aus Binance API ziehen
- [ ] **Sentiment:** Fear & Greed Index täglich laden und als Feature einbauen

### Datenqualität

- [ ] Feature-Korrelationsmatrix prüfen (hoch korrelierte Features entfernen)
- [ ] Alle Features normalisieren / skalieren wo nötig
- [ ] Feature-DataFrame als CSV exportieren und manuell prüfen

### Tests & Qualität

- [ ] Unit-Tests für Feature-Berechnung schreiben
- [ ] Erste Modell-Erklärbarkeit prüfen (z.B. Feature Importance, SHAP)

**✅ Phase 2 abgeschlossen, wenn:** Du einen Feature-DataFrame mit >20 sinnvollen Spalten hast, keine NaN-Werte.

---

## 🟠 PHASE 3 – Regime Detection

**Ziel:** Marktphasen automatisch erkennen (Trend ↑, Trend ↓, Seitwärts, Volatil)

### Statistische Methode (Einstieg)

- [ ] `regime_detection.py` erstellen
- [ ] Volatilität berechnen (Rolling ATR / Rolling Std der Returns)
- [ ] Trendstärke berechnen (ADX oder Autokorrelation der Returns)
- [ ] Regelbasierte Regime-Klassifikation:

  ```
  0 = Seitwärts    (niedrige Volatilität, kein Trend)
  1 = Trend aufwärts  (hohe Autokorrelation + steigende Preise)
  2 = Trend abwärts   (hohe Autokorrelation + fallende Preise)
  3 = Volatil/Chaotisch (hohe Volatilität, kein klarer Trend)
  ```

- [ ] Regime als neue Spalte `market_regime` in Feature-DataFrame einfügen

### Validierung

- [ ] Regime-Spalte visuell auf dem Chart überprüfen (matplotlib)
- [ ] Verteilung der Regime prüfen (keine Klasse sollte >60% haben)

### Tests & Modell-Erklärbarkeit (Regime Detection)

- [ ] Unit-Tests für Regime-Detection schreiben
- [ ] Regime-Feature mit SHAP/Feature Importance validieren

**✅ Phase 3 abgeschlossen, wenn:** Das Feature `market_regime` korrekt im DataFrame steht und visuell Sinn ergibt.

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

- [ ] `labeling.py` erstellen
- [ ] **Methode wählen** (eine davon):
  - Multi-Bar: Steigt der Kurs in den nächsten 5 Kerzen um >X Pips? → Label 1/0/-1
  - Double-Barrier-Labeling mit 5-Barren-Horizont: Wird TP 1% oder SL 0.5% zuerst erreicht? → Label 1/-1
- [ ] Label-Verteilung prüfen (ausgeglichene Klassen anstreben)

### Datenaufteilung (zeitlich!)

- [ ] Training: 2015–2020
- [ ] Validierung: 2021–2022
- [ ] Test (nie anfassen bis zum Schluss!): 2023+

### Modelltraining

- [ ] `train_model.py` erstellen
- [ ] **XGBoost** trainieren mit Basis-Parametern
- [ ] **LightGBM** trainieren mit `multi_logloss` als Zielfunktion und Regime-Indikator als Feature
- [ ] Schwellenwert für Trade-Ausführung festlegen (z.B. Wahrscheinlichkeit >60%)
- [ ] Hyperparameter-Tuning mit **Optuna** (je >50 Trials)
- [ ] Bestes Modell als `.pkl` speichern

### Walk-Forward-Analyse

- [ ] `walk_forward.py` erstellen
- [ ] 5 Fenster à 1 Jahr Training / 3 Monate Test durchlaufen
- [ ] Accuracy und F1-Score pro Fenster aufzeichnen
- [ ] Stabiles Modell auswählen (kein einzelnes Fenster deutlich schlechter)

### Tests & Modell-Erklärbarkeit (Modelltraining)

- [ ] Unit-Tests für Labeling schreiben
- [ ] Modell mit SHAP erklären (wichtigste Features visualisieren)

### CI/CD & Automatisierung

- [ ] GitHub Actions für automatisierte Tests und Linting einrichten
- [ ] Automatisiertes Deployment für Modelle vorbereiten

**✅ Phase 4 abgeschlossen, wenn:** Walk-Forward zeigt konsistente Accuracy >52% über alle Fenster.

---

## 🟣 PHASE 5 – Backtesting

**Ziel:** Realistische Simulation des Systems auf historischen Daten

### Backtesting mit VectorBT

- [ ] `backtest.py` erstellen
- [ ] Modellsignale in Buy/Sell-Orders umwandeln
- [ ] Backtest mit Double-Barrier-Regeln: fester SL/TP wie beim Labeling, nur Trades mit hoher Modellwahrscheinlichkeit
- [ ] Spread, Slippage und Kommission einrechnen
- [ ] Simulation durchlaufen

### Risikomanagement (Details)

- [ ] Dynamische Positionsgrößenberechnung (z.B. Risiko pro Trade max. 1% des Kapitals)
- [ ] Dynamisches Stop-Loss-Management (z.B. ATR-basiert)
- [ ] Backtest auf verschiedene Märkte/Zeiträume ausweiten

### Auswertung

- [ ] **Kennzahlen berechnen:**
  - Gesamtrendite (%)
  - Sharpe Ratio (Ziel: >1.0)
  - Max. Drawdown (Ziel: <20%)
  - Gewinnfaktor (Ziel: >1.3)
  - Anzahl Trades
- [ ] **Performance nach Regime analysieren:** Wie gut ist das System in Trend- vs. Seitwärtsphasen?
- [ ] **Monatliche Performance** als Heatmap darstellen

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
| 1 | Umgebung & Daten | 🔄 In Arbeit |
| 2 | Feature Engineering | ⬜ Offen |
| 3 | Regime Detection | ⬜ Offen |
| 4 | Labeling & Training | ⬜ Offen |
| 5 | Backtesting | ⬜ Offen |
| 6 | Live-Integration | ⬜ Offen |
| 7 | Wartung | ⬜ Offen |
| Q | Code-Qualität & CI/CD | ⬜ Offen |

> Status: ⬜ Offen | 🔄 In Arbeit | ✅ Abgeschlossen

---

*Letzte Aktualisierung: 2026-02-25*
