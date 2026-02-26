# 🗺️ MT5 ML-Trading-System – Roadmap

**Hauptwährungspaare:** `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD`, `USDCHF`, `NZDUSD`

**Ziel:** Ein automatisches Handelssystem mit XGBoost/LightGBM + Regime-Detection, das in MetaTrader 5 live handelt.

---

## 🖥️ System-Architektur

| Gerät | Rolle | Was läuft hier? |
|-------|-------|-----------------|
| Windows 11 Laptop | MT5-Host & Live-Trading | MT5 Terminal, MetaTrader5 Python-Lib, `live_trader.py`, Paper-Trading |
| Linux Server (1TB SSD) | Datenspeicher & Training | Rohdaten (CSV), Modelle (.pkl), `train_model.py`, `backtest.py` |
| VS Code Remote SSH | Entwicklung | Code wird auf dem Linux-Server bearbeitet und ausgeführt |

> **Hinweis:** `data_loader.py` und `live_trader.py` müssen auf dem Windows 11 Laptop laufen, da die MetaTrader5-Bibliothek eine laufende MT5-Instanz auf demselben Rechner benötigt. Alle anderen Skripte laufen auf dem Linux-Server.

---

## 📌 Wie du diese Roadmap benutzt

- Hake jede Aufgabe ab, wenn sie erledigt ist: `[ ]` → `[x]`
- Gehe niemals zur nächsten Phase, bevor die aktuelle abgeschlossen ist
- Bei jedem Schritt kannst du den KI-Assistenten mit dem beigelegten Prompt um Hilfe bitten

---

## 🔁 Übergreifende Qualitätsstandards

> **Gilt für JEDE Phase – wird hier einmalig definiert statt pro Phase wiederholt.**

- **Unit-Tests:** Für jede zentrale Funktion Tests schreiben (`pytest`). Testabdeckung bei Code-Reviews prüfen.
- **Dokumentation:** Alle Entscheidungen, Methoden und Berechnungen im Code und in `README.md` dokumentieren.
- **Manuelle Stichproben:** Bei neuen Features, Labels oder Trades stichprobenartig prüfen, ob die Werte sinnvoll sind.
- **Code-Reviews:** Vor jedem Merge in `main` mindestens ein Review durch ein Teammitglied.
- **Reproduzierbarkeit:** Alle Skripte versioniert, alle Berechnungsmethoden dokumentiert, `requirements.txt` aktuell.
- **CI/CD:** GitHub Actions für Linting (`black`, `flake8`) und Tests (`pytest`) bei jedem Pull Request.
- **Plattform-Konsistenz:** Alle Pfade mit `os.path`/`pathlib`, Skripte auf beiden Plattformen lauffähig.

---

## ⚠️ Review-Anmerkungen & Kritische Verbesserungsvorschläge

> **Die folgenden Punkte wurden nach einem externen Review identifiziert und sollten vom Team diskutiert und priorisiert werden.**

### 🔴 Kritisch – Vor Live-Trading lösen

| # | Problem | Aktion |
|---|---------|--------|
| 1 | **Edge ist dünn** – F1-Macro 0.42–0.48, moderat besser als Zufall | Profit Factor + erwartete Rendite pro Trade nach Kosten als Go/No-Go-Kriterien definieren |
| 2 | **Survivorship Bias** – 7 Paare trainiert, 2 selektiert = Data-Mining-Risiko | Bonferroni-Korrektur oder Durchschnitt aller 7 Paare als Benchmark |
| 3 | **Backtest-Renditen sehr klein** – +2% über ~3 Jahre | Transaction Cost Sensitivity Test: Spreads verdoppeln → noch profitabel? |
| 6 | **Risikomanagement unvollständig** – fester SL 0.3% ignoriert Volatilitätsprofile | ATR-basiertes SL + dynamische Positionsgröße **VOR Phase 6** implementieren |
| 8 | **Kein Kill-Switch** – nur Alert bei DD >10% | Harter Kill-Switch bei 15–20% DD automatisch im Code |

### 🟡 Wichtig – Vor oder während Live-Phase

| # | Problem | Aktion |
|---|---------|--------|
| 4 | **Externe APIs ohne SLA** – Fear & Greed + BTC Funding Rate können ausfallen | Fallback definieren: kein Trade / letzter Wert / Feature weglassen |
| 5 | **Look-Ahead-Bias möglich** – Regime auf Gesamtdaten berechnet? | Code-Review von `regime_detection.py`, Unit-Test für Future-Leak |
| 7 | **Retraining zu häufig** – wöchentlich bei ~120 neuen Kerzen sinnlos | Monatlich + Trigger bei Rolling Sharpe < 0.5 |
| 9 | **Paper-Trading zu kurz** – 2 Wochen ohne statistische Aussagekraft | Mindestens 3 Monate vor echtem Geld |

### 🟢 Empfohlen – Für langfristige Qualität

| # | Empfehlung |
|---|-----------|
| 10 | **Out-of-Sample Reality-Check:** Letzte 3 Monate Trades einzeln prüfen – machen Signale Sinn? |

---

## ⚙️ PHASE 0 – Vorbereitung ✅

**Ziel:** Stabiles Fundament für das gesamte Projekt legen

### Plattform-Test

- [x] MT5 auf Windows 11 Laptop – MetaTrader5 Python-Bibliothek funktioniert
- [x] vectorbt, lightgbm, xgboost auf Linux-Server (ARM) – alle 12 Bibliotheken OK
- [x] Python-Umgebung auf Windows 11 Laptop (`python -m venv venv`, `pip install -r requirements-laptop.txt`)

### Versionskontrolle

- [x] Git-Repository initialisiert
- [x] `.gitignore` angelegt (Modelle, Daten, API-Keys, venv/)
- [x] Erster Commit + Remote-Repository auf GitHub (mt5-ml-trading)
- [x] Branching-Strategie: `main` = stabiler Branch (wird direkt verwendet, Solo-Projekt)

### Projektfundament

- [x] `requirements-server.txt` + `requirements-laptop.txt` angelegt
- [x] `.env.example` Template + `python-dotenv` eingerichtet
- [x] `README.md` mit Projektbeschreibung und Setup-Anleitung
- [x] `.env` mit echten API-Keys befüllt

> ✅ **Phase 0 abgeschlossen.**

---

## ✅ PHASE 1 – Umgebung & Datenbeschaffung

**Ziel:** Funktionierende Entwicklungsumgebung + erste Daten aus MT5

### Setup

- [x] Virtuelle Umgebungen auf beiden Plattformen erstellt und getestet
- [x] Projektordner-Struktur angelegt:

```
/mnt/1T-Data/XGBoost-LightGBM/
├── .github/          ├── models/          ├── plots/
├── data/             ├── backtest/        ├── tests/
├── features/         ├── live/            ├── .env / .gitignore
                      ├── notebooks/       ├── requirements-*.txt
```

- [x] Linting: `black` 26.1.0 + `flake8` 7.2.0
- [x] Erste Tests: `tests/test_features.py` – 7 Unit-Tests, alle bestanden

### Datenbeschaffung

- [x] `data_loader.py` geschrieben (Windows Laptop mit MT5)
- [x] 8 Jahre historische Daten (2018–2026, alle 7 Paare)
- [x] CSV gespeichert und geprüft (keine NaN, OHLC-Logik OK)

> ✅ **Phase 1 abgeschlossen:** 7 × ~49.000 Kerzen als CSV verfügbar.

---

## ✅ PHASE 2 – Feature Engineering

**Ziel:** Aus Rohdaten aussagekräftige Merkmale für das Modell erzeugen

### Technische Indikatoren

- [x] `feature_engineering.py` erstellt
- [x] Trend: SMA 20/50/200, EMA 12/26, MACD
- [x] Momentum: RSI (14), Stochastic, Williams %R
- [x] Volatilität: Bollinger Bands, ATR
- [x] Volumen: OBV, Volume Rate of Change

### Erweiterte Features

- [x] Multi-Timeframe: H4- und D1-Trend als Feature in H1-Daten
- [x] Order Flow: BTC Funding Rate (8h) + BTC Open Interest (1h) via Binance API
- [x] Sentiment: Fear & Greed Index (Alternative.me, täglich → H1 forward-fill)

> ⚠️ **Review-Punkt 4:** Fear & Greed + BTC APIs haben kein SLA → Fallback-Mechanismus nötig.

### Datenqualität

- [x] 7× SYMBOL_H1_features.csv exportiert (~49.000 Kerzen, 56 Features)
- [x] Korrelationsmatrix geprüft (Tree-Modelle tolerieren hohe Korrelation)
- [x] Normalisierung nicht nötig für XGBoost/LightGBM (dokumentiert)

### 🔧 Optimierung (offen)

- [ ] Feature-Selektion: `SelectFromModel` oder Permutation Importance nach erstem Training
- [ ] Recursive Feature Elimination (RFE) pro Symbol
- [ ] Alternative Feature-Sets testen (nur Trend, nur Volatilität, etc.)
- [ ] Weitere Indikatoren evaluieren: Parabolic SAR, CCI, Keltner Channels, VWAP

> ✅ **Phase 2 abgeschlossen:** 7 Paare × 56 Features, keine NaN-Werte.

---

## ✅ PHASE 3 – Regime Detection

**Ziel:** Marktphasen automatisch erkennen (Trend ↑, Trend ↓, Seitwärts, Volatil)

### Statistische Methode

- [x] `regime_detection.py` erstellt
- [x] Volatilität: ATR% vs. rollender Median(50)
- [x] Trendstärke: ADX(14)
- [x] Regelbasierte Klassifikation (Priorität: Vola > Trend > Seitwärts):
  - 0 = Seitwärts (~53–57%), 1 = Aufwärtstrend (~17–21%)
  - 2 = Abwärtstrend (~21–25%), 3 = Hohe Volatilität (~2–4%)
- [x] `market_regime` + `adx_14` in alle Feature-CSVs eingefügt

> ⚠️ **Review-Punkt 5:** Code-Review sicherstellen, dass alle Rolling-Berechnungen nur historische Daten verwenden (kein Look-Ahead-Bias). Unit-Test dafür schreiben.

### Validierung

- [x] Regime-Verteilung geprüft – keine Dominanz >60%
- [x] Visualisierung: `plots/SYMBOL_regime.png` für alle 7 Paare

### 🔧 Optimierung (offen)

- [ ] Hidden-Markov-Modell (HMM) als Alternative testen (`hmmlearn`)
- [ ] Regime-Transition-Trigger: z.B. ADX > 25 für ≥3 Kerzen, um Fehlsignale zu reduzieren
- [ ] Separate Modelle pro Regime trainieren (Trend-Modell vs. Seitwärts-Modell)
- [ ] Regime-Performance-Analyse: Welche Regime sind profitabel, welche verlustreich?

> ✅ **Phase 3 abgeschlossen:** 7 Paare × 58 Features, ~48.960 Kerzen.

---

## ✅ PHASE 4 – Labeling & Modelltraining

**Ziel:** Modell trainieren, das Kauf-/Verkaufssignale vorhersagt

### Labeling

- [x] `labeling.py` erstellt
- [x] Double-Barrier: TP=SL=0.3%, Horizon=5 H1-Barren
- [x] Label-Verteilung: Long ~11–22%, Short ~9–24%, Neutral ~55–82%

### 🔧 Labeling-Optimierung (offen)

- [ ] ATR-basierte Barrieren statt fixem 0.3% (z.B. 1.5×ATR)
- [ ] Dynamischer Horizont (5–10 Kerzen, abhängig von Volatilität)
- [ ] Alternative Zielfunktionen: Regression auf erwartete Rendite
- [ ] Label-Noise-Analyse: Stabilität bei kleinen Barrieren-Änderungen prüfen

### Datenaufteilung (zeitlich!)

- [x] Training: 2018-04 bis 2021-12 (~23.000 Kerzen)
- [x] Validierung: 2022 (~6.250 Kerzen)
- [x] Test: 2023-01 bis 2026-02 (~19.500 Kerzen – **HEILIG, nicht anfassen!**)

### Modelltraining

- [x] `train_model.py` erstellt
- [x] XGBoost Baseline: F1-Macro = 0.4452
- [x] LightGBM Baseline: F1-Macro = 0.4303
- [x] XGBoost Optuna (50 Trials): F1-Macro = 0.4810
- [x] LightGBM Optuna (50 Trials): F1-Macro = 0.4830 ← **Bestes Modell**
- [x] 14 Modelle gespeichert (XGBoost + LightGBM × 7 Symbole)
- [x] Schwellenwert-Analyse in `train_model.py`

> ⚠️ **Review-Punkt 1:** F1-Macro 0.42–0.48 = dünner Edge. Zusätzlich **Profit Factor** und **erwartete Rendite pro Trade nach Kosten** als Go/No-Go definieren.

### 🔧 Modell-Optimierung (offen)

- [x] Profit Factor (= Gewinnfaktor) wird in `kennzahlen_berechnen()` berechnet und im Ziel-Check ausgegeben (Ziel: >1.3)
- [ ] Feature Importance → unwichtige Features entfernen
- [ ] Ensemble: XGBoost + LightGBM Vorhersagen kombinieren
- [x] Out-of-Sample Reality-Check: `reports/reality_check.py` erstellt (Review-Punkt 10)

### Walk-Forward-Analyse

- [x] `walk_forward.py` – 5 Expanding Windows, 2019–2022
- [x] Alle 7 Modelle stabil (kein Fenster > 0.10 unter Durchschnitt)

| Symbol | Ø F1 | Min F1 | Schwankung | Status |
|--------|------|--------|------------|--------|
| EURUSD | 0.4188 | 0.3709 | 0.0748 | ✅ STABIL |
| GBPUSD | 0.4681 | 0.4384 | 0.0522 | ✅ STABIL |
| USDJPY | 0.3988 | 0.3513 | 0.1519 | ✅ STABIL |
| AUDUSD | 0.4175 | 0.3994 | 0.0410 | ✅ STABIL |
| USDCAD | 0.4337 | 0.3839 | 0.0772 | ✅ STABIL |
| USDCHF | 0.4290 | 0.4011 | 0.0641 | ✅ STABIL |
| NZDUSD | 0.3943 | 0.3601 | 0.0673 | ✅ STABIL |

### Modell-Erklärbarkeit

- [x] SHAP-Analyse: `features/shap_analysis.py`

### 🔧 Drift-Erkennung (offen)

- [ ] Population Stability Index (PSI): Wöchentlich Vorhersage-Verteilung vs. Training vergleichen (Alarm bei PSI > 0.2)
- [ ] Kalibrierungsprüfung: Vorhergesagte Wahrscheinlichkeiten vs. tatsächliche Eintrittshäufigkeiten
- [ ] Feature-Drift-Monitoring: Verteilung der Top-SHAP-Features auf Live vs. Training
- [ ] Rolling-Performance: F1/Profit Factor auf letzten 100 Trades überwachen

> ✅ **Phase 4 abgeschlossen:** Alle 7 Modelle stabil, LightGBM F1-Macro Ø=0.42–0.47.

---

## 🟣 PHASE 5 – Backtesting

**Ziel:** Realistische Simulation des Systems auf historischen Daten

### Backtesting mit VectorBT

- [x] `backtest.py` erstellt
- [x] Modellsignale → Buy/Sell-Orders
- [x] Double-Barrier-Regeln, Schwellenwert-Filter
- [x] Spread, Slippage und Kommission eingerechnet
- [x] Simulation durchlaufen

### Risikomanagement

> 🔴 **PRIORITÄT – VOR Phase 6 abschliessen!** (Review-Punkt 6)

- [x] **Dynamische Positionsgröße:** `--kapital 10000 --risiko_pct 0.01` implementiert in `backtest.py`
- [x] **Dynamisches Stop-Loss:** ATR-basiert via `--atr_sl --atr_faktor 1.5` in `backtest.py`
- [x] **Transaction Cost Sensitivity Test:** `--spread_faktor 2.0` implementiert → Spreads verdoppeln und prüfen ob noch profitabel (Review-Punkt 3)
- [x] **Swap-Kosten einrechnen:** `--swap_aktiv` in `backtest.py` (SWAP_KOSTEN_LONG/SHORT, Mitternacht-Prüfung)
- [x] Backtest auf verschiedene Zeiträume ausweiten (`--zeitraum_von` / `--zeitraum_bis`)

### Auswertung

- [x] Kennzahlen: Gesamtrendite, Sharpe Ratio, Max. Drawdown, Gewinnfaktor, Anzahl Trades
- [x] Performance nach Regime analysiert (Rendite + Win-Rate pro Regime)
- [x] Monatliche Performance als Heatmap
- [x] **Survivorship-Bias-Korrektur:** Durchschnitt aller 7 Paare als Benchmark (Review-Punkt 2) → in `backtest.py` nach Schleife

### Backtest-Ergebnisse

| Symbol | Regime-Filter | Threshold | Sharpe | Rendite | Max.DD |
|--------|--------------|-----------|--------|---------|--------|
| USDCAD | 1,2 | 60% | 1.277 ✅ | +2.01% | -1.36% |
| USDJPY | 1 (nur Aufwärtstrend) | 60% | 1.073 ✅ | +2.59% | -3.15% |
| USDCHF | 1,2 | 60% | 0.271 | +1.54% | -4.72% |
| EURUSD | 1,2 | 60% | 0.027 | +0.11% | -4.95% |

> ⚠️ **Review-Punkte 2 & 3:** Renditen (+2% über ~3 Jahre) sind sehr gering. Survivorship Bias möglich. Ehrlichere Benchmark und Kosten-Stress-Test nötig.

> ✅ **Phase 5 abgeschlossen:** Sharpe >1.0 für USDCAD + USDJPY. Risikomanagement noch offen.

---

## 🔄 PHASE 6 – Live-Integration (MT5)

**Ziel:** System läuft automatisch auf dem Windows 11 Laptop und handelt live

### ⛔ Voraussetzungen (VOR Phase 6 prüfen!)

- [ ] Dynamisches Risikomanagement implementiert und getestet (Phase 5)
- [ ] Transaction Cost Sensitivity Test bestanden (Review-Punkt 3)
- [ ] Fallback für externe APIs implementiert (Review-Punkt 4)
- [ ] Out-of-Sample Reality-Check durchgeführt (Review-Punkt 10)
- [ ] Go/No-Go basierend auf Profit Factor definiert (Review-Punkt 1)
- [ ] Kill-Switch bei Max. Drawdown 15–20% implementiert (Review-Punkt 8)

### Infrastruktur

- [x] MT5 Terminal auf Windows 11 Laptop installiert
- [ ] `pip install -r requirements-laptop.txt` auf Laptop
- [ ] Modelle (.pkl) vom Linux-Server auf Laptop übertragen (`scp`)
- [ ] Laptop-Schlaf/Ruhemodus deaktivieren
- [ ] ⚠️ Langfristig Windows-VPS für 24/7-Betrieb evaluieren (~5 €/Monat)

### CI/CD & Monitoring

- [ ] Automatisiertes Deployment (Modell-Update Server → Laptop)
- [ ] Health-Checks und automatisierte Neustarts bei Fehlern

### Live-Skript

- [x] `live_trader.py` erstellt:

```
Jede neue H1-Kerze:
1. 500 H1-Barren von MT5 holen
2. Alle 45 Features berechnen (identisch mit Training)
3. Fear & Greed + BTC Funding Rate live laden (mit Fallback!)
4. Marktregime erkennen (ADX + ATR + SMA50)
5. LightGBM-Vorhersage + Schwellenwert-Filter (60%)
6. Regime-Filter anwenden (z.B. nur Regime 1,2)
7. Order senden (Paper-Modus: nur loggen!)
```

- [x] Logging: `logs/SYMBOL_live_trades.csv` + `live_trader.log`
- [x] Error-Handling: Auto-Restart nach 60s
- [x] Paper-Trading als Standard (`PAPER_TRADING=True`)
- [x] Stop-Loss ist Pflicht in jeder echten Order
- [x] **Fallback bei API-Ausfall** implementiert: Fear & Greed → 50/Neutral, BTC Funding → 0.0 (Review-Punkt 4)
- [x] **Kill-Switch bei Max. Drawdown** implementiert in `live_trader.py`: `--kill_switch_dd 0.15` (Review-Punkt 8)

### Paper-Trading

- [ ] **Mindestens 3 Monate** Paper-Trading laufen lassen (Review-Punkt 9):

```bash
python live/live_trader.py --symbol USDCAD --schwelle 0.60 --regime_filter 1,2
python live/live_trader.py --symbol USDJPY --schwelle 0.60 --regime_filter 1
```

> ✅ **Phase 6 abgeschlossen, wenn:** System läuft **3 Monate** stabil im Paper-Trading mit positiver Performance nach realistischen Kosten.

---

## ⚫ PHASE 7 – Überwachung & Wartung

**Ziel:** System langfristig stabil und profitabel halten

### Monitoring

- [ ] Tägliche Performance-E-Mail (Python + SMTP)
- [ ] Alert bei Drawdown >10% (System pausieren)
- [ ] **Harter Kill-Switch bei Drawdown >15–20%** (automatisch stoppen, Review-Punkt 8)
- [ ] Modell-Drift wöchentlich überwachen (PSI, Rolling Sharpe, Feature-Drift)

### Retraining

> ⚠️ **Review-Punkt 7:** Wöchentliches Retraining ist zu häufig. Monatlich empfohlen.

- [x] **Monatliches** Retraining-Skript einrichten → `retraining.py` erstellt
- [x] **Trigger:** Rolling Sharpe < 0.5 → Retraining anstoßen (`trigger_pruefen()`)
- [x] Neues Modell nur deployed, wenn F1 >= F1_alt - 1% (`modelle_vergleichen()`)
- [x] Modell-Versionierung (`lgbm_SYMBOL_v1.pkl` → `v2.pkl` → ...) + JSON-Historie

### Code-Qualität

- [ ] CI/CD-Pipeline (GitHub Actions) für Tests + Linting
- [ ] Monatliche Code-Reviews im Team

> ✅ **Phase 7 abgeschlossen, wenn:** System läuft 3+ Monate autonom mit positivem Ergebnis.

---

## 📊 Fortschritts-Übersicht

| Phase | Beschreibung | Status |
|-------|-------------|--------|
| 0 | Vorbereitung (Git, .env, Bibliothekstest) | ✅ Abgeschlossen |
| 1 | Umgebung & Daten | ✅ Abgeschlossen |
| 2 | Feature Engineering | ✅ Abgeschlossen |
| 3 | Regime Detection | ✅ Abgeschlossen |
| 4 | Labeling & Training | ✅ Abgeschlossen |
| 5 | Backtesting | 🔄 Risikomanagement offen |
| 6 | Live-Integration | ⬜ Offen |
| 7 | Wartung | ⬜ Offen |
| **R** | **Review-Punkte abarbeiten** | **⬜ Offen** |

> Status: ⬜ Offen | 🔄 In Arbeit | ✅ Abgeschlossen

**Letzte Aktualisierung:** 2026-02-26 – Phase 5 Backtesting abgeschlossen, Review-Feedback integriert
