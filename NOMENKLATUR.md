# 📖 Nomenklatur – MT5 ML-Trading-System

> Alle Abkürzungen, Fachbegriffe und Namenskonventionen des Projekts auf einen Blick.

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
| **LTF** | Lower Timeframe | Niedriger Zeitrahmen (z.B. M5 für Entry) |
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
| **Two-Stage** | Zwei-Stufen-Modell mit HTF-Bias + LTF-Entry (z.B. H1→M5) |
| **Shadow-Compare** | Kontrollierter Vergleich alt vs. neu im Paper-Betrieb |
| **Heartbeat** | Regelmäßiges Lebenszeichen-Log pro Kerze |
| **Kill-Switch** | Automatischer Stopp bei zu hohem Drawdown (>15%) |
| **Ausschluss-Spalten** | Spalten die NICHT als ML-Input genutzt werden |
| **Labeling** | Zuweisung von Labels (Short, Neutral, Long) an historische Daten |
| **Drift** | Verteilungsverschiebung von Features/Modell über die Zeit |
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
| **USDCAD** | US-Dollar / Kanad. Dollar | ✅ Aktiv (Paper) – Regime 2 |
| **USDJPY** | US-Dollar / Jap. Yen | ✅ Aktiv (Paper) – Regime 1 |
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
| `ATR_SL_FAKTOR` | 1.5 | SL = ATR × 1.5 |
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

Konsequenz: Betrieb weiterhin **PAPER_ONLY** und Shadow-Compare-Laufzeit abwarten.

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

Letzte Aktualisierung: 2026-03-05
