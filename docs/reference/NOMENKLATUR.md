# 📖 Nomenklatur – MT5 ML-Trading-System

> Alle Abkürzungen, Fachbegriffe und Namenskonventionen des Projekts auf einen Blick.

---

## 📖 Inhaltsverzeichnis

- [1. Trading-Abkürzungen](#1-trading-abkürzungen)
- [2. ML-Abkürzungen](#2-ml-abkürzungen)
- [3. Deutsche Projektbegriffe](#3-deutsche-projektbegriffe)
- [4. Regime-Klassifikation](#4-regime-klassifikation)
- [5. Klassen-Labels](#5-klassen-labels)
- [6. Währungspaare](#6-währungspaare)
- [7. Zeitrahmen (Timeframes)](#7-zeitrahmen-timeframes)
- [8. Feature-Liste](#8-feature-liste-alle-spalten)
- [9. Datei-Namenskonventionen](#9-datei-namenskonventionen)
- [10. Labeling-Modi](#10-labeling-modi)
- [11. Datenaufteilung (zeitlich)](#11-datenaufteilung-zeitlich)
- [12. Wichtige Konstanten](#12-wichtige-konstanten)
- [13. Python-Bibliotheken](#13-python-bibliotheken)
- [14. Weitere Fachbegriffe](#14-weitere-fachbegriffe)
- [15. KPI-Snapshot](#15-kpi-snapshot-beispiel-aus-backtest)
- [16. KPI-Zielrichtung](#16-kpi-zielrichtung-höher-vs-niedriger)
- [17. Weitere wichtige Werte](#17-weitere-wichtige-werte-bereits-im-projekt-vorhanden)
- [18. LiveSignalDashboard (MT5-Chart-Anzeige)](#18-livesignaldashboard-mt5-chart-anzeige)

---

## 1. Trading-Abkürzungen

| Abkürzung | Ausgeschrieben | Bedeutung |
| --------- | ------------- | --------- |
| **MT5** | MetaTrader 5 | Trading-Plattform (läuft auf Windows Laptop) |
| **SL** | Stop-Loss | Verlustbegrenzung pro Trade |
| **TP** | Take-Profit | Gewinnziel pro Trade |
| **DD** | Drawdown | Maximaler Kapitalrückgang |
| **ATR** | Average True Range | Volatilitäts-Maß (14 Perioden) |
| **ADX** | Average Directional Index | Trendstärke (0–100), >25 = Trend |
| **SMA** | Simple Moving Average | Einfacher gleitender Durchschnitt |
| **EMA** | Exponential Moving Average | Exponentieller gleitender Durchschnitt |
| **RSI** | Relative Strength Index | Momentum-Indikator (14 Perioden) |
| **MACD** | Moving Average Convergence/Divergence | Momentum-Indikator (12, 26, 9) |
| **OBV** | On-Balance Volume | Volumenbasiertes Feature |
| **ROC** | Rate of Change | Prozentuale Preisveränderung |
| **BB** | Bollinger Bands | Volatilitätsbänder (20, 2σ) |
| **RRR** | Risk-Reward Ratio | Risiko-Ertrags-Verhältnis (z.B. 2:1) |
| **GF** | Gewinnfaktor | = Profit Factor (Gewinne ÷ Verluste) |
| **Pip** | Percentage in Point | Kleinstmögliche Preisänderung |
| **Lot** | Lot | Positionsgröße (0.01 = Micro-Lot) |
| **Swap** | Swap | Overnight-Haltekosten |
| **Spread** | Spread | Differenz Bid/Ask (Handelskosten) |
| **OHLCV** | Open/High/Low/Close/Volume | Die 4 Kerzenpreise + Volumen |
| **Stoch** | Stochastic Oscillator | Momentum-Oszillator (%K, %D) |
| **Williams %R** | Williams Percent Range | Überkauft/Überverkauft-Indikator |
| **+DI / -DI** | Directional Indicator | Richtungskomponenten des ADX |
| **MTF** | Multi-Timeframe | Features aus höheren Zeitrahmen |
| **HTF** | Higher Timeframe | Höherer Zeitrahmen (z.B. H1 für Bias) |
| **LTF** | Lower Timeframe | Niedriger Zeitrahmen (z.B. M15 für Entry) |
| **SMC** | Smart Money Concepts | Marktstruktur-/Liquiditätskonzepte |
| **FVG** | Fair Value Gap | Ungleichgewichtslücke im Preisverlauf |
| **BOS** | Break of Structure | Strukturbruch im Trendverlauf |
| **MSS** | Market Structure Shift | Früher Strukturwechsel vor BOS |
| **PDH/PDL** | Previous Day High/Low | Vortageshoch/Vortagestief |
| **PWH/PWL** | Previous Week High/Low | Vorwochenhoch/Vorwochentief |
| **HMM** | Hidden Markov Model | Probabilistisches Regime-Modell |
| **Kill Zone** | Kill Zone | Zeitfenster mit erhöhter Marktaktivität |
| **SAR** | Parabolic Stop and Reverse | (geplant) |

---

## 2. ML-Abkürzungen

| Abkürzung | Ausgeschrieben | Bedeutung |
| --------- | ------------- | --------- |
| **ML** | Machine Learning | Maschinelles Lernen |
| **XGB** | Extreme Gradient Boosting | Gradient-Boosting-Modell (Baseline) |
| **LGBM** | Light Gradient Boosting Machine | Gradient-Boosting-Modell (Hauptmodell) |
| **F1** | F1-Score | Harmonisches Mittel von Precision & Recall |
| **F1-Macro** | F1-Score Macro-Average | Gleichgewichteter F1 über alle Klassen |
| **F1-Weighted** | F1-Score Weighted-Average | Gewichteter F1 nach Klassenhäufigkeit |
| **PSI** | Population Stability Index | Drift-Erkennung |
| **SHAP** | SHapley Additive exPlanations | Modell-Erklärbarkeit |
| **TPE** | Tree-structured Parzen Estimator | Optuna-Sampler für Hyperparameter |
| **Acc** | Accuracy | Genauigkeit |
| **Prec** | Precision | Präzision |
| **Rec** | Recall | Trefferquote |
| **Val** | Validation | Validierungsdatensatz |
| **NaN** | Not a Number | Fehlende Werte |
| **KPI** | Key Performance Indicator | Leistungskennzahl |
| **EWM** | Exponentially Weighted Moving | Exponentielles Glätten |
| **CI/CD** | Continuous Integration/Deployment | Automatisierte Tests & Deployment |

---

## 3. Deutsche Projektbegriffe

| Begriff | Bedeutung |
| ------- | --------- |
| **Schwelle** | Mindest-Wahrscheinlichkeit für Trade-Auslösung (z.B. 0.50 = 50%) |
| **Kerze / Barren** | Ein Datenpunkt (OHLCV) in einem Zeitrahmen |
| **Schranke** | Barriere im Double-Barrier-Labeling (TP/SL-Level) |
| **Zeitschranke** | Maximal erlaubte Haltedauer (Horizon = 5 Kerzen) |
| **Regime** | Marktphase (0–3, siehe Tabelle unten) |
| **Gewinnfaktor** | Profit Factor (Brutto-Gewinne ÷ Brutto-Verluste) |
| **Gesamtrendite** | Kumulative Rendite über den Backtest-Zeitraum |
| **Erzwingen** | Retraining manuell auslösen (ignoriert Trigger) |
| **Toleranz** | F1-Abweichung, unter der ein neues Modell noch deployed wird |
| **Warm-Up** | Anfangsperiode mit NaN (z.B. SMA200 braucht 200 Kerzen) |
| **Paper-Trading** | Simulierter Handel ohne echtes Geld |
| **Two-Stage** | Zwei-Stufen-Modell mit HTF-Bias + LTF-Entry (z.B. H1→M15) |
| **Shadow-Compare** | Kontrollierter Vergleich alt vs. neu im Paper-Betrieb |
| **Heartbeat** | Regelmäßiges Lebenszeichen-Log pro Kerze |
| **Kill-Switch** | Automatischer Stopp bei zu hohem Drawdown (>15%) |
| **Ausschluss-Spalten** | Spalten die NICHT als ML-Input genutzt werden |
| **Labeling** | Zuweisung von Labels (Short, Neutral, Long) an historische Daten |
| **Drift** | Verteilungsverschiebung von Features/Modell über die Zeit |
| **Stale** | Operativer Monitoring-Begriff: Daten/Logs sind zu alt und nicht mehr frisch genug für verlässliche Bewertung |
| **WATCH** | Operativer Status: beobachten – noch kein harter Fehler, aber Aktivität oder Datenlage ist nicht ideal |
| **DRIFT (Monitoring)** | Operativer Monitoring-Begriff: Datei wirkt frisch, aber der Inhalt hinkt hinter Runtime/Heartbeat oder Erwartung hinterher |
| **INCIDENT** | Operativer Status: echter Störfall – sofort prüfen, weil Datenfluss, Sync oder Trader-Lauf nicht vertrauenswürdig ist |
| **Look-Ahead-Bias** | Fehler: Zukunftsdaten fließen in Features ein |
| **Survivorship Bias** | Fehler: Nur erfolgreiche Paare werden ausgewählt |
| **GO-Woche** | Woche in der alle KPI-Gates erfüllt sind (12 konsekutive nötig) |

---

## 4. Regime-Klassifikation

| Nr. | Name | Bedingung |
| --- | ---- | --------- |
| **0** | Seitwärts | ADX < 25 und normale Volatilität |
| **1** | Aufwärtstrend | ADX > 25 und Close > SMA50 |
| **2** | Abwärtstrend | ADX > 25 und Close < SMA50 |
| **3** | Hohe Volatilität | ATR_pct > 1.5 × Median-ATR(50) |

> Priorität: 3 → 1/2 → 0 (Hohe Vola hat Vorrang)

---

## 5. Klassen-Labels

| Wert (intern) | Wert (CSV) | Name | Bedeutung |
| ------------- | ---------- | ---- | --------- |
| 0 | -1 | **Short** | Verkaufs-Signal |
| 1 | 0 | **Neutral** | Kein Signal (kein Trade) |
| 2 | +1 | **Long** | Kauf-Signal |

---

## 6. Währungspaare

| Code | Basis / Quote | Status |
| ---- | ------------ | ------ |
| **USDCAD** | US-Dollar / Kanad. Dollar | ✅ Aktiv (Paper) – Two-Stage v4 (H1+M15) |
| **USDJPY** | US-Dollar / Jap. Yen | ✅ Aktiv (Paper) – Two-Stage v4 (H1+M15) |
| **EURUSD** | Euro / US-Dollar | Research-only |
| **GBPUSD** | Brit. Pfund / US-Dollar | Research-only |
| **AUDUSD** | Austral. Dollar / US-Dollar | Research-only |
| **USDCHF** | US-Dollar / Schweizer Franken | Research-only |
| **NZDUSD** | Neuseel. Dollar / US-Dollar | Research-only |

---

## 7. Zeitrahmen (Timeframes)

| Code | Bedeutung | Minuten/Kerze |
| ---- | --------- | ------------- |
| **M15** | 15-Minuten | 15 |
| **M5** | 5-Minuten | 5 |
| **M30** | 30-Minuten | 30 |
| **M60** | 60-Minuten (= H1) | 60 |
| **H1** | 1-Stunde (Standard) | 60 |
| **H4** | 4-Stunden | 240 |
| **D1** | Daily / Täglich | 1440 |

---

## 8. Feature-Liste (alle Spalten)

### Trend-Features

`price_sma20_ratio`, `price_sma50_ratio`, `price_sma200_ratio`, `sma_20_50_cross`, `sma_50_200_cross`, `ema_cross`, `macd_line`, `macd_signal`, `macd_hist`

### Momentum-Features

`rsi_14`, `rsi_centered`, `stoch_k`, `stoch_d`, `stoch_cross`, `williams_r`, `roc_10`

### Volatilitäts-Features

`atr_pct`, `bb_width`, `bb_pct`, `hist_vol_20`

### Volumen-Features

`obv_zscore`, `volume_roc`, `volume_ratio`

### Kerzenmuster-Features

`return_1h`, `return_4h`, `return_24h`, `candle_body`, `upper_wick`, `lower_wick`, `candle_dir`, `hl_range`

### Multi-Timeframe-Features

`trend_h4`, `rsi_h4`, `trend_d1`

### Zeit-Features

`hour`, `day_of_week`, `session_london`, `session_ny`, `session_asia`, `session_overlap`

### Regime-Features

`adx_14`, `market_regime`

### SMC/MTF-Feature-Familien (Phase 7B)

`key_levels_*` (z.B. PDH/PDL/PWH/PWL), `fvg_*`, `bos_*`, `mss_*`, `killzone_*`, `market_regime_hmm`

### Externe Features

`fear_greed_value`, `fear_greed_class`, `btc_funding_rate`

### Ausschluss-Spalten (NICHT als ML-Input)

`time`, `open`, `high`, `low`, `close`, `volume`, `spread`, `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26`, `atr_14`, `bb_upper`, `bb_mid`, `bb_lower`, `obv`, `label`

---

## 9. Datei-Namenskonventionen

### CSV-Dateien (data/)

| Muster | Beispiel | Inhalt |
| ------ | -------- | ------ |
| `SYMBOL_TF.csv` | `USDCAD_H1.csv` | Rohdaten (OHLCV) |
| `SYMBOL_TF_features.csv` | `USDCAD_H1_features.csv` | + Indikatoren + Regime |
| `SYMBOL_TF_labeled.csv` | `USDCAD_H1_labeled.csv` | + Labels (v1) |
| `SYMBOL_TF_labeled_vN.csv` | `USDCAD_H1_labeled_v3.csv` | Labeling-Version N |

### Modell-Dateien (models/)

| Muster | Beispiel | Inhalt |
| ------ | -------- | ------ |
| `lgbm_symbol_vN.pkl` | `lgbm_usdcad_v1.pkl` | LightGBM H1 |
| `xgb_symbol_vN.pkl` | `xgb_usdcad_v1.pkl` | XGBoost H1 |
| `lgbm_symbol_TF_vN.pkl` | `lgbm_usdcad_M15_v1.pkl` | Anderer Timeframe |
| `lgbm_symbol_rtN.pkl` | `lgbm_usdcad_rt1.pkl` | Retraining-Version |
| `ensemble_symbol_vN.pkl` | `ensemble_usdcad_v4.pkl` | XGB+LGBM Ensemble |
| `SYMBOL_f1_history.json` | `USDCAD_f1_history.json` | F1-Score-Historie |

### Versions-Schema

| Prefix | Bedeutung |
| ------ | --------- |
| **v1** | Standard-Labeling (TP=SL=0.3%) |
| **v2** | Horizon=10 |
| **v3** | Asymmetrisch (RRR 2:1) |
| **v4** | ATR-basiertes Labeling |
| **rt1, rt2, ...** | Retraining-Versionen |

---

## 10. Labeling-Modi

| Modus | Methode | Barrieren |
| ----- | ------- | --------- |
| **standard** | Feste symmetrische Barrieren | TP = SL = 0.3% |
| **rrr** | Asymmetrisch (Risk-Reward) | z.B. TP=0.6%, SL=0.3% |
| **atr** | Dynamisch (volatilitätsbasiert) | TP = SL = ATR_14 × Faktor |

---

## 11. Datenaufteilung (zeitlich)

| Split | Zeitraum | Zweck |
| ----- | -------- | ----- |
| **Training** | 2018-04 bis 2021-12 | Muster lernen (~23.000 Kerzen) |
| **Validierung** | 2022-01 bis 2022-12 | Modellselektion & Tuning |
| **Test** | 2023-01 bis heute | **HEILIG** – nur finale Bewertung |

> ⚠️ **NIEMALS** `train_test_split` mit `shuffle=True` auf Zeitreihendaten!

---

## 12. Wichtige Konstanten

| Konstante | Wert | Bedeutung |
| --------- | ---- | --------- |
| `TP_PCT` | 0.003 | Take-Profit 0.3% |
| `SL_PCT` | 0.003 | Stop-Loss 0.3% |
| `LOT` | 0.01 | Micro-Lot |
| `HORIZON` | 5 | 5 Kerzen Zeitschranke |
| `ATR_SL_FAKTOR` | 2.0 | SL = ATR × 2.0 |
| `KILL_SWITCH_DD` | 0.15 | Stopp bei 15% Drawdown |
| `MAGIC_NUMBER` | 20260101 | MT5-Kennung für ML-Trades |
| `N_BARREN` | 500 | Mindest-Kerzen für Features |
| `OPTUNA_TRIALS` | 50 | Standard-Tuning-Durchläufe |
| `SHARPE_GRENZWERT` | 0.5 | Retraining-Trigger |
| `F1_TOLERANZ` | 0.01 | 1% Toleranz bei Deployment |

---

## 13. Python-Bibliotheken

| Bibliothek | Zweck | Gerät |
| ---------- | ----- | ----- |
| **MetaTrader5** | MT5-API (Daten, Orders) | 🪟 Windows |
| **pandas** | DataFrames, CSV I/O | Beide |
| **numpy** | Numerische Berechnungen | Beide |
| **pandas_ta** | Technische Indikatoren | Beide |
| **xgboost** | XGBoost-Klassifikator | 🐧 Linux |
| **lightgbm** | LightGBM-Klassifikator | 🐧 Linux |
| **scikit-learn** | Metriken, Gewichte | 🐧 Linux |
| **optuna** | Hyperparameter-Optimierung | 🐧 Linux |
| **joblib** | Modell-Serialisierung (.pkl) | Beide |
| **shap** | Modell-Erklärbarkeit | 🐧 Linux |
| **vectorbt** | Backtesting-Framework | 🐧 Linux |
| **matplotlib** | Plots (Agg-Backend) | 🐧 Linux |
| **seaborn** | Statistische Visualisierungen | 🐧 Linux |
| **requests** | HTTP-APIs (Fear & Greed, BTC) | Beide |
| **python-dotenv** | .env für API-Keys | Beide |
| **pathlib** | Plattformunabhängige Pfade | Beide |
| **pytest** | Unit-Tests | 🐧 Linux |

---

## 14. Weitere Fachbegriffe

| Begriff | Bedeutung |
| ------- | --------- |
| **Double-Barrier** | Labeling-Methode: TP-Barriere + SL-Barriere + Zeitschranke |
| **Walk-Forward** | Expanding-Window-Validierung (5 Fenster) |
| **Equity-Kurve** | Kapitalverlauf über Zeit |
| **Soft Voting** | Ensemble: Durchschnitt der Vorhersage-Wahrscheinlichkeiten |
| **Early Stopping** | Training endet wenn Val-Metrik stagniert (30 Runden) |
| **Permutation Importance** | Feature-Wichtigkeit durch Zufallsmischung |
| **Sample Weight** | Klassen-Ausgleichsgewichte (balanced) |
| **Wilder-Smoothing** | EMA-Variante mit α = 1/Länge (MetaTrader-Standard) |
| **Z-Score** | Standardisierung: (Wert − Mean) / StdDev |
| **Forward-Fill** | Fehlende Werte mit letztem Wert auffüllen |
| **Kontraktgröße** | 100.000 Einheiten = 1 Standard-Lot (Forex) |
| **Magic Number** | Eindeutige Order-ID in MT5 für ML-Trades |
| **Bonferroni-Korrektur** | Statistische Anpassung für Mehrfachtests |
| **Bullish / Bearish** | Aufwärts- / Abwärtsgerichtet |
| **Überkauft / Überverkauft** | RSI > 70 / RSI < 30 |

---

## 15. KPI-Snapshot (Beispiel aus Backtest)

### USDJPY – H1 (Konfiguration: Schwelle 0.55, Regime 1, ATR-SL 1.5, Option B)

- **Trades: 216**  
  Nützlich, um die statistische Belastbarkeit zu beurteilen (zu wenige Trades = hohe Zufallsschwankung).
- **Rendite: +4.45%**  
  Zeigt den absoluten Kapitalzuwachs im Testzeitraum und damit die praktische Wirtschaftlichkeit.
- **PF (Profit Factor): 1.208**  
  Verhältnis Brutto-Gewinn zu Brutto-Verlust; misst Robustheit nach Kosten (Projektziel: > 1.3).
- **Sharpe: 1.297**  
  Risikoadjustierte Rendite; ideal zum Vergleich mehrerer Varianten mit ähnlicher Rendite.
- **MaxDD: -3.92%**  
  Größter zwischenzeitlicher Kapitalrückgang; wichtig für Risiko- und Stress-Toleranz im Phase-7-Betrieb.

> Praxisregel für Phase 7: Eine Konfiguration gilt nur dann als nachhaltig „GO", wenn **Ertrag + Risiko + Stabilität** gemeinsam passen (nicht nur eine einzelne Kennzahl).

### Aktueller Two-Stage Stress-Status (2026-03-05)

- USDCAD v5: Sharpe -23.002, PF 0.011, MaxDD -164.30% → **NO-GO**
- USDJPY v5: Sharpe -2.140, PF 0.692, MaxDD -34.36% → **NO-GO**

Konsequenz: v5 eingestellt. Aktives Setup ist **Two-Stage v4 (H1-Bias + M15-Entry)** im Paper-Modus.

---

## 16. KPI-Zielrichtung (höher vs. niedriger)

- **Trades (Anzahl):** kein „je mehr desto besser", sondern **ausreichend hoch** für belastbare Statistik.
- **Rendite (%):** **höher ist besser** (positiv soll sie sein).
- **PF / Gewinnfaktor:** **höher ist besser**. Praxis: >1.0 profitabel, Projektziel robust: >1.3.
- **Sharpe Ratio:** **höher ist besser** (mehr Rendite pro Risiko).
- **MaxDD (%):** **weniger negativ ist besser** (näher an 0). Beispiel: -4% ist besser als -10%.

Merksatz:

- **Hoch gut:** Rendite, PF, Sharpe
- **Niedrig gut:** MaxDD (in Absolutverlust)
- **Ausreichend gut:** Trades

---

## 17. Weitere wichtige Werte (bereits im Projekt vorhanden)

- **Win-Rate (%)**
  Anteil gewonnener Trades. Nützlich als Stabilitätsindikator (aber nie allein bewerten).
- **Live-Signale pro Woche**
  Zeigt, ob im Paper-Betrieb genug Aktivität für eine Wochenbewertung vorhanden ist.
- **Exits TP / SL / Horizon**
  Aufschlüsselung, ob Gewinne eher über TP kommen oder ob viele Trades in SL/Horizon enden.
- **Regime-Performance (pro Regime)**
  Zeigt, in welchen Marktphasen die Strategie wirklich funktioniert (z.B. USDJPY nur Regime 1).
- **Jahres-Performance (pro Jahr)**
  Stabilitätscheck gegen Überanpassung: wichtig, damit ein gutes Gesamtergebnis nicht nur aus einem starken Jahr stammt.
- **Konsekutive GO-Wochen (0 bis 12)**
  Operativer Freigabe-Indikator in Phase 7; Ziel ist eine stabile Serie ohne Unterbrechung.

Hinweis:
Für Go/No-Go im laufenden Betrieb immer **mehrere KPIs gemeinsam** betrachten (mindestens PF, Sharpe, MaxDD, Win-Rate, Signalanzahl).

---

## 18. LiveSignalDashboard (MT5-Chart-Anzeige)

Das **LiveSignalDashboard** (`LiveSignalDashboard.mq5`) ist ein MQL5-Indikator auf dem Windows-Laptop.
Es liest die CSV-Signaldateien von Python und zeigt alle Informationen direkt auf dem MT5-Chart an.

### Kopfbereich (oben links)

| Zeile | Beispiel | Bedeutung |
| ----- | -------- | --------- |
| **Status** | `Live Dashboard \| CONNECTED` | Verbindungsstatus. CONNECTED = CSV wird gelesen. STALE/OFFLINE = Verbindung gestört. **Erwünscht: immer CONNECTED.** |
| **Grenze** | `Grenze: 25min (SignalTF)` | Maximale Gültigkeitsdauer eines Signals. Nach Ablauf wird es als veraltet ignoriert. Standard: 25min für M15. |
| **Countdown** | `Update in 16:23` | Zeit bis zum nächsten Signal-Update von Python. **Je kleiner, desto frischer werden die Daten bald.** |

### Symbol-Status (pro Währungspaar)

| Feld | Beispiel | Bedeutung |
| ---- | -------- | --------- |
| **Symbol** | `USDJPY` | Welches Währungspaar. |
| **State** | `SIGNAL` / `IDLE` | SIGNAL = aktives Handelssignal. IDLE = kein Signal. **SIGNAL = Modell hat etwas gefunden.** |
| **Frische** | `OK (63 min)` | OK = Daten innerhalb der Grenze. Zahl = Alter der letzten CSV-Zeile. **Je kleiner die Zahl, desto frischer.** |
| **Richtung** | `Short` / `Long` / `Kein` | Handelsrichtung des Modells. "Kein" = kein Trade empfohlen. |
| **P=** | `P=0.45` | Wahrscheinlichkeit (Probability) des Modells für das stärkste Signal (0.00–1.00). **Je höher, desto sicherer das Signal.** Aktuell muss P ≥ Schwelle sein (z.B. 0.45). |
| **R=** | `R=463` | Regime-ID – interne Kennung für den aktuellen Marktmodus. Siehe Abschnitt 4 für Zuordnung. |

### Technische Marktdaten (Mitte links)

| Zeile | Beispiel | Bedeutung |
| ----- | -------- | --------- |
| **Spread** | `Spread: 1.8 Pips` | Differenz Bid/Ask = Eintrittskosten pro Trade. **Je niedriger, desto besser.** Typisch: USDJPY 1.0–2.5 Pips, USDCAD 1.5–3.0 Pips. |
| **ATR** | `ATR: 19 Pips (NIEDRIG)` | Average True Range = durchschnittliche Schwankungsbreite. NIEDRIG/NORMAL/HOCH beeinflusst SL/TP-Berechnung. **Kein "besser" – nur Kontext.** NIEDRIG = engere SL/TP, HOCH = weitere SL/TP. |
| **Schwelle** | `Schwelle: 45%` | Mindest-Wahrscheinlichkeit, ab der ein Signal als handelbar gilt. **Höhere Schwelle = weniger Trades, aber höhere Qualität.** Typisch: 40–55%. |

### Indikator-Analyse

| Zeile | Beispiel | Bedeutung |
| ----- | -------- | --------- |
| **MACD** | `MACD: bearish` | MACD-Trendrichtung: bullish (aufwärts) oder bearish (abwärts). **Kein "besser" – zeigt nur Richtung.** |
| **M=** | `M=0.00411` | MACD-Linie = Differenz zwischen EMA12 und EMA26. **Positiv = bullish Tendenz, Negativ = bearish.** |
| **S=** | `S=0.01351` | Signal-Linie = geglätteter Durchschnitt der MACD-Linie. **M > S = bullish Crossover, S > M = bearish.** |
| **H=** | `H=-0.00940` | Histogramm = M minus S. **Je weiter von 0 entfernt, desto stärker das Momentum.** Negativ = bearish, Positiv = bullish. |
| **Ichimoku** | `Ichimoku: bullish \| ueber Kumo` | Ichimoku-Wolke: Preis über Kumo = bullisch, unter Kumo = bärisch, in Kumo = neutral. **"ueber Kumo" ist das stärkste bullische Signal.** |
| **A=** | `A=159.28550` | Senkou Span A (obere/untere Wolkengrenze). **A > B = bullische Wolke (grün).** |
| **B=** | `B=159.19750` | Senkou Span B (obere/untere Wolkengrenze). **B > A = bärische Wolke (rot).** Je größer der Abstand A↔B, desto stärker die Wolke. |

### Regime, Session & Two-Stage

| Zeile | Beispiel | Bedeutung |
| ----- | -------- | --------- |
| **Regime** | `Regime: Hohe Volatilität \| PAPER` | Vom ML-Modell erkannte Marktphase (siehe Abschnitt 4). PAPER = kein echtes Geld. **Aufwärtstrend/Abwärtstrend = klarer Markt (gut). Hohe Volatilität = vorsichtig. Seitwärts = wenig Bewegung.** |
| **Session** | `Session: New York` | Aktive Handelssitzung: Asia, London, New York oder Overlap (London+NY). **Overlap (London+NY) = höchste Liquidität und engste Spreads. Asia = ruhigster Markt.** |
| **Two-Stage** | `Two-Stage: HTF=Neutral \| LTF=Short` | HTF = Higher Timeframe (H1) Bias, LTF = Lower Timeframe (M15) Entry. **Idealfall: beide zeigen in dieselbe Richtung (z.B. HTF=Long + LTF=Long). Widerspruch = schwaches Signal.** |

### Trade-Info-Labels (oben rechts, 3 Zeilen)

| Zeile | Beispiel | Bedeutung |
| ----- | -------- | --------- |
| **Zeile 1** | `USDJPY Short @ 159.20100 \| Prob=49%` | Symbol, Richtung, Entry-Preis und Modell-Wahrscheinlichkeit. **Prob: je höher, desto besser. Ab Schwelle (~45%) wird gehandelt. >60% = starkes Signal.** |
| **Zeile 2** | `SL=159.50796 \| TP=158.58781 \| Hohe Volatilität` | Stop-Loss, Take-Profit und aktuelles Regime. **TP sollte weiter vom Entry entfernt sein als SL (gutes RRR). SL immer gesetzt = Pflicht.** |
| **Zeile 3** | `HTF=Short \| LTF=Short` | Two-Stage HTF-Bias und LTF-Entry-Signal. **Beide gleich (z.B. Short+Short) = stärkstes Signal. Widerspruch = vorsichtig.** |

### EMA-Struktur-Label (oben rechts)

| Beispiel | Bedeutung |
| -------- | --------- |
| `BULL: EMA20>EMA50>EMA200 \| Preis ueber allen` | Optimaler Aufwärtstrend: alle EMAs gestapelt + Preis darüber. **Stärkstes bullisches Setup.** |
| `BEAR: EMA20<EMA50<EMA200 \| Preis unter allen` | Optimaler Abwärtstrend: EMAs umgekehrt + Preis darunter. **Stärkstes bärisches Setup.** |
| `BULL Stack, Preis in Pullback-Zone` | EMAs bullisch, aber Preis zwischen den EMAs (Rücksetzer). **Möglicher Einstieg, aber riskanter.** |
| `MIXED` | Keine klare EMA-Reihenfolge – Seitwärtsmarkt oder Übergang. **Schwächstes Setup – besser abwarten.** |

### Ichimoku-Cloud-Label (oben rechts)

| Beispiel | Bedeutung |
| -------- | --------- |
| `Cloud: BULL \| Preis ueber Kumo` | Preis über bullischer Wolke = starker Aufwärtstrend. **Stärkstes bullisches Ichimoku-Signal.** |
| `Cloud: BEAR \| Preis unter Kumo` | Preis unter bärischer Wolke = starker Abwärtstrend. **Stärkstes bärisches Ichimoku-Signal.** |
| `Cloud: NEUTRAL \| Preis in Kumo` | Preis innerhalb der Wolke = unklare Richtung. **Kein klares Signal – besser abwarten.** |
| `Cloud: BULL/BEAR Bias \| Preis nahe Kumo` | Wolke zeigt Tendenz, aber Preis ist nahe an der Wolke. **Schwaches Signal – Richtung unsicher.** |

### Ampel-System (unten links)

| Farbe | Text | Bedeutung |
| ----- | ---- | --------- |
| 🟢 **GRÜN** | `GRUEN: HANDELN` | Signal aktiv + hohe Wahrscheinlichkeit → Trade möglich. |
| 🟡 **GELB** | `GELB: BEOBACHTEN` | Daten frisch, aber kein/schwaches Signal → abwarten. |
| 🔴 **ROT** | `ROT: WARTEN` | Daten veraltet oder keine CSV → Finger weg! |

### Datenfluss

```text
Python (live_trader.py auf Windows)
  → trade_logger.py → CSV schreiben (USDJPY_signals.csv)
  → mt5_connector.py → CSV nach MT5 Common/Files kopieren

MQL5 (LiveSignalDashboard.mq5 auf MT5)
  → liest CSV alle 5 Sekunden
  → zeichnet Dashboard, Trade-Linien, Ampel auf den Chart
```

---

Letzte Aktualisierung: 2026-03-14
