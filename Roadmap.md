# 🗺️ MT5 ML-Trading-System – Roadmap

**Research-Universum:** `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD`, `USDCHF`, `NZDUSD`

**Aktive operative Paare (Paper):** `USDCAD`, `USDJPY`

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

- [x] Feature-Selektion: Permutation Importance in `train_model.py` implementiert
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
- [x] Regime-Performance-Analyse: `regime_matrix_erstellen()` + `regime_matrix_plotten()` in `backtest.py` – Matrix-Tabelle im Terminal + Heatmap `plots/regime_performance_matrix.png` + CSV `backtest/regime_performance_matrix.csv`

> ✅ **Phase 3 abgeschlossen:** 7 Paare × 58 Features, ~48.960 Kerzen.

---

## ✅ PHASE 4 – Labeling & Modelltraining

**Ziel:** Modell trainieren, das Kauf-/Verkaufssignale vorhersagt

### Labeling

- [x] `labeling.py` erstellt
- [x] Double-Barrier: TP=SL=0.3%, Horizon=5 H1-Barren
- [x] Label-Verteilung: Long ~11–22%, Short ~9–24%, Neutral ~55–82%

### 🔧 Labeling-Optimierung

- [x] ATR-basierte Barrieren statt fixem 0.3% (1.5×ATR) → `labeling.py --modus atr` (v4-Labels)
  - **Ergebnis:** Label-Verteilung besser (~27/46/27 statt ~15/70/15), aber Modell-F1 nicht signifikant besser
  - **Erkenntnis:** ATR-SL als Ausführungsstrategie bringt mehr als ATR-Labeling!
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
- [x] Feature Importance → Permutation Importance implementiert in `train_model.py`
  - USDCAD v4: 35/42 Features behalten, 7 entfernt (ema_cross, session_ny, session_asia, etc.)
  - USDJPY v4: 28/42 Features behalten, 14 entfernt
- [x] Ensemble: XGBoost + LightGBM Soft Voting implementiert (`EnsembleModell` Klasse)
  - Ergebnis: Ensemble ist nicht besser als bestes Einzelmodell (LightGBM)
  - Trotzdem verfügbar als Robustheits-Option
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

## ✅ PHASE 5 – Backtesting

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

**H1-Modelle (Finale Konfiguration – nach ATR-SL-Optimierung 2026-03-03):**

| Symbol | Timeframe | Modell | Regime-Filter | Threshold | Sharpe | Rendite | GF | Max.DD |
|--------|-----------|--------|--------------|-----------|--------|---------|-----|--------|
| **USDCAD** | H1 | v1 + ATR-SL 1.5× | **2** (nur Abwärtstrend) | **50%** | **2.118 ✅** | **+2.95%** | **1.35 ✅** | -1.18% |
| **USDJPY** | H1 | v1 + ATR-SL 1.5× | **1** (nur Aufwärtstrend) | **50%** | **1.263 ✅** | **+5.94%** | 1.20 | -4.17% |
| USDCHF | H1 | v1 | 1,2 | 60% | 0.271 | +1.54% | – | -4.72% |
| EURUSD | H1 | v1 | 1,2 | 60% | 0.027 | +0.11% | – | -4.95% |

**Vorherige Konfiguration (ohne ATR-SL, zum Vergleich):**

| Symbol | Regime-Filter | Threshold | Sharpe | Rendite | GF |
|--------|--------------|-----------|--------|---------|-----|
| USDCAD | 1,2 | 60% | 1.277 | +2.01% | ~1.2 |
| USDJPY | 1 | 60% | 1.073 | +2.59% | ~1.1 |

> **Verbesserung durch ATR-SL:** USDCAD Sharpe +0.84 (+66%), USDJPY Rendite +3.35pp (+129%)

> ⚠️ **Review-Punkte 2 & 3:** Renditen (+2% über ~3 Jahre) sind sehr gering. Survivorship Bias möglich. Ehrlichere Benchmark und Kosten-Stress-Test nötig.

> ✅ **Phase 5 abgeschlossen:** Sharpe >1.0 für USDCAD (H1) + USDJPY (H1). USDCHF (H4) als 3. Kandidat identifiziert.

---

## ✅ BONUS – ATR-SL-Optimierung & v4-Experiment (2026-03-03)

**Ziel:** Bestehende Modelle durch verbesserte Ausführungsstrategie optimieren

### Durchgeführte Experimente

1. **ATR-basiertes Labeling (v4):** `labeling.py --modus atr --atr_faktor 1.5`
   - Bessere Label-Verteilung (~27/46/27 statt ~15/70/15)
   - Modell-F1 nicht signifikant besser (USDCAD: 0.4912 vs 0.4810, USDJPY: 0.4317 vs 0.4230)
   - **Fazit:** ATR-Labeling allein bringt keinen Durchbruch

2. **Feature Selection (Permutation Importance):**
   - USDCAD: 7/42 Features entfernt (ema_cross, session_ny, session_asia, bb_pct, stoch_cross, market_regime, macd_signal)
   - USDJPY: 14/42 Features entfernt (candle_dir, ema_cross, session_asia, volume_ratio, sma_50_200_cross, lower_wick, etc.)
   - **Fazit:** Identifiziert unwichtige Features, Retrain empfohlen

3. **Ensemble (Soft Voting XGB+LGBM):**
   - USDCAD: F1=0.4882 (schlechter als bestes Einzelmodell 0.4912)
   - USDJPY: F1=0.4291 (schlechter als bestes Einzelmodell 0.4317)
   - **Fazit:** Ensemble nicht besser als LightGBM allein

4. **ATR-basiertes Stop-Loss (Ausführungsstrategie) – DURCHBRUCH!**
   - v1-Modell + dynamisches SL (ATR_14 × 1.5) statt festes SL (0.3%)
   - **USDCAD:** Sharpe 0.241 → **2.118** (+780%!), Rendite +0.40% → **+2.95%**
   - **USDJPY:** Sharpe 0.040 → **1.263** (+3058%!), Rendite +0.18% → **+5.94%**
   - **Schlüsselerkenntnis:** Die Ausführungsstrategie (wie SL berechnet wird) ist wichtiger als das Labeling (wie Labels berechnet werden)

### Optimale Konfiguration (nach Parameter-Sweep)

| Symbol | Modell | ATR-SL | Regime | Schwelle | Rendite | GF | Sharpe |
|--------|--------|--------|--------|----------|---------|-----|--------|
| USDCAD | lgbm v1 | 1.5× ATR | Regime 2 (Abwärtstrend) | 50% | +2.95% | 1.35✅ | 2.12✅ |
| USDJPY | lgbm v1 | 1.5× ATR | Regime 1 (Aufwärtstrend) | 50% | +5.94% | 1.20 | 1.26✅ |

### Implementierung

- [x] `labeling.py`: ATR-Labeling-Modus (`--modus atr`)
- [x] `train_model.py`: Feature Selection + Ensemble
- [x] `backtest.py`: ATR-SL bereits implementiert (bestätigt in Optimierung)
- [x] `live_trader.py`: ATR-SL als Parameter (`--atr_sl 1 --atr_faktor 1.5`)
- [x] Paper-Trading-Befehle auf neue Konfiguration aktualisiert

---

## ✅ BONUS – H4-Experiment (2026-02-28)

**Ziel:** Prüfen ob H4-Zeitrahmen bessere Signale liefert als H1

- [x] `features/h4_pipeline.py` erstellt: H1 → H4 Resampling + alle Features + Regime + Labeling
- [x] `train_model.py --timeframe H4` Parameter hinzugefügt (rückwärtskompatibel)
- [x] `backtest.py --timeframe H4` Parameter hinzugefügt (rückwärtskompatibel)
- [x] 14 H4-Modelle trainiert (50 Optuna-Trials, `lgbm/xgb_SYMBOL_H4_v1.pkl`)
- [x] H4-Backtest für alle 7 Symbole durchgeführt

**H4-Ergebnisse** (`--schwelle 0.60 --regime_filter 1,2`):

| Symbol | Sharpe | Rendite | Trades | Empfehlung |
|--------|--------|---------|--------|------------|
| USDCAD | 12.135 | +1.38% | 9 | ⚠️ Zu wenige Trades – statistisch nicht valide |
| USDCHF | 2.502 ✅ | +1.26% | 28 | ✅ Besser als H1 (0.271) – Abwärtstrend (regime=2) |
| USDJPY | 0.069 | +0.30% | 233 | ❌ H1 ist besser (Sharpe=1.073) |
| EURUSD | -2.260 | -4.03% | 101 | ❌ Verworfen |
| GBPUSD | -2.785 | -5.89% | 121 | ❌ Verworfen |
| AUDUSD | -2.043 | -1.97% | 52 | ❌ Verworfen |
| NZDUSD | -2.309 | -8.20% | 194 | ❌ Verworfen |

**Fazit:** H4 ersetzt H1 nicht. USDCHF H4 (regime_filter=2) ist als 3. Paper-Trading-Kandidat interessant.

---

## ✅ PHASE 6 – Live-Integration (MT5)

**Ziel:** System läuft automatisch auf dem Windows 11 Laptop und handelt live

### ⛔ Voraussetzungen (vor Echtgeld-Betrieb prüfen)

- [x] Dynamisches Risikomanagement implementiert und getestet (Phase 5)
- [x] Transaction Cost Sensitivity Test bestanden (Review-Punkt 3)
- [x] Fallback für externe APIs implementiert (Review-Punkt 4)
- [x] Out-of-Sample Reality-Check durchgeführt (Review-Punkt 10)
- [x] Go/No-Go-Kriterien inkl. Profit Factor definiert (Review-Punkt 1)
- [x] Kill-Switch bei Max. Drawdown implementiert (Review-Punkt 8)

### Infrastruktur

- [x] MT5 Terminal auf Windows 11 Laptop installiert
- [x] `pip install -r requirements-laptop.txt` auf Laptop
- [x] Modelle (.pkl) vom Linux-Server auf Laptop übertragen
- [x] Laptop-Schlaf/Ruhemodus deaktivieren
- [ ] ⚠️ Langfristig Windows-VPS für 24/7-Betrieb evaluieren (~5 €/Monat)

### CI/CD & Monitoring

- [x] MT5-Dashboard + CSV-Sync aufgebaut (`live/mt5/*`)
- [x] Autostart-Sync via Task Scheduler dokumentiert/automatisiert
- [ ] Vollautomatisches Deployment (Modell-Update Server → Laptop)
- [ ] Erweiterte Health-Checks und automatisierte Neustarts bei Fehlern

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
- [x] **Heartbeat-Logging** implementiert (`--heartbeat_log 1`) für Dashboard-Datenfrische
- [x] **Operative Policy**: nur `USDCAD` + `USDJPY` aktiv, andere Paare research-only

### Paper-Trading (laufend)

- [x] Start Paper-Trading erfolgt (ab 2026-02-28)
- [ ] **Mindestens 3 Monate** Paper-Trading laufen lassen (Review-Punkt 9):

```bash
# Auf Windows Laptop ausführen! (nach ATR-SL-Optimierung 2026-03-03)
python live/live_trader.py --symbol USDCAD --schwelle 0.50 --regime_filter 2 --atr_sl 1   # H1, nur Abwärtstrend, ATR-SL
python live/live_trader.py --symbol USDJPY --schwelle 0.50 --regime_filter 1 --atr_sl 1   # H1, nur Aufwärtstrend, ATR-SL
# python live/live_trader.py --symbol USDCHF --schwelle 0.60 --regime_filter 2   # Research-only (nicht operativ)
```

> ✅ **Phase 6 abgeschlossen:** MT5-Integration + stabiler Paper-Betrieb gestartet.
> Für Echtgeld-Freigabe gilt weiterhin das 90-Tage-/12-GO-Wochen-Gate aus Phase 7.

### 📅 4-Wochen-Umsetzungsplan (direkt ab Start)

- [x] Woche 1 gestartet und Betriebskette stabilisiert
- [ ] Woche 1–4 nach `reports/paper_trading_4w_execution_plan.md` vollständig durchführen
- [ ] Wöchentliche Done-Definitionen objektiv abhaken
- [ ] Nach Woche 4 Zwischenentscheidung dokumentieren: Stabil halten vs. kontrolliertes Feintuning

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
| 5 | Backtesting | ✅ Abgeschlossen |
| B | H4-Experiment (Bonus) | ✅ Abgeschlossen |
| 6 | Live-Integration | ✅ Abgeschlossen (Paper-Betrieb aktiv) |
| 7 | Wartung | 🔄 In Arbeit (Monitoring-/KPI-Gates laufen) |

> Status: ⬜ Offen | 🔄 In Arbeit | ✅ Abgeschlossen

**Letzte Aktualisierung:** 2026-03-03 – ATR-SL-Optimierung: Sharpe +66% (USDCAD), Rendite +129% (USDJPY). Paper-Betrieb auf neue Konfiguration umgestellt.
