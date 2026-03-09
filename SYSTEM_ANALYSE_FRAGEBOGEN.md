# 🔍 System-Analyse-Fragebogen – MT5 ML-Trading-System

**Zweck:** Systematische Analyse deines Trading-Systems für optimale Feineinstellung.

**Version:** 1.0 (Stand: 8. März 2026)

**Anleitung:** Arbeite dich durch jeden Abschnitt und beantworte die Fragen. Notiere deine Erkenntnisse und nutze sie für gezielte Optimierungen.

---

## 📋 INHALTSVERZEICHNIS

1. [Systemarchitektur & Infrastruktur](#1-systemarchitektur--infrastruktur)
2. [Datenqualität & Features](#2-datenqualität--features)
3. [Modellperformance & Metriken](#3-modellperformance--metriken)
4. [Signalgenerierung & Entscheidungslogik](#4-signalgenerierung--entscheidungslogik)
5. [Risikomanagement](#5-risikomanagement)
6. [Live-Trading-Verhalten](#6-live-trading-verhalten)
7. [Regime-Detection & Marktphasen](#7-regime-detection--marktphasen)
8. [Two-Stage-Architektur](#8-two-stage-architektur)
9. [Backtesting & Validierung](#9-backtesting--validierung)
10. [Monitoring & Wartung](#10-monitoring--wartung)
11. [Fehlerbehandlung & Stabilität](#11-fehlerbehandlung--stabilität)
12. [Performance-Optimierung](#12-performance-optimierung)

---

## 1. SYSTEMARCHITEKTUR & INFRASTRUKTUR

### 1.1 Geräteverteilung

**Frage 1.1.1:** Auf welchem Gerät läuft welcher Teil des Systems?

- **Was prüfen:** `README.md` System-Architektur-Tabelle
- **Warum wichtig:** Falsche Ausführung auf falscher Plattform führt zu Fehlern (z.B. MT5 nur auf Windows)
- **Kommando:** `cat README.md | grep -A 5 "System-Architektur"`
- **Erwartung:**
  - Windows Laptop: MT5 Terminal, `live_trader.py`, Dashboard
  - Linux Server: Training, Backtesting, Feature-Engineering

**Frage 1.1.2:** Sind alle Umgebungsvariablen korrekt gesetzt?

- **Was prüfen:** `MT5_SERVER`, `MT5_LOGIN`, `MT5_PASSWORD`, `FEAR_GREED_FILE`
- **Wo prüfen:** Windows-CMD: `echo %MT5_SERVER%`
- **Warum wichtig:** Fehlende Credentials = keine MT5-Verbindung
- **Aktion bei Fehler:** `setx MT5_SERVER "dein-broker-server"` (dauerhaft speichern)

**Frage 1.1.3:** Ist die virtuelle Umgebung aktiviert?

- **Was prüfen:** Prompt zeigt `(.venv)` oder `(venv)`
- **Windows:** `.venv\Scripts\activate`
- **Linux:** `source .venv/bin/activate`
- **Warum wichtig:** Falsche Python-Version/Bibliotheken können zu stillen Fehlern führen

**Frage 1.1.4:** Sind alle Dependencies installiert?

- **Was prüfen:** `pip list | grep -E "(lightgbm|xgboost|MetaTrader5|pandas_ta)"`
- **Linux:** `pip install -r requirements-server.txt`
- **Windows:** `pip install -r requirements-laptop.txt`
- **Warum wichtig:** Fehlende Bibliotheken = Crashes

---

## 2. DATENQUALITÄT & FEATURES

### 2.1 Rohdaten

**Frage 2.1.1:** Wie viele historische Kerzen habe ich pro Symbol?

- **Kommando:** `wc -l data/*.csv | head -20`
- **Erwartung:** ~49.000 Zeilen für 8 Jahre H1-Daten (2018-2026)
- **Warum wichtig:** Zu wenige Daten = schlechte Modellgeneralisierung

**Frage 2.1.2:** Gibt es fehlende Werte (NaN - Not a Number) in meinen Rohdaten?

- **Kommando:** `python -c "import pandas as pd; df=pd.read_csv('data/USDCAD_H1.csv'); print(df.isnull().sum())"`
- **Erwartung:** Alle Spalten = 0 NaN
- **Warum wichtig:** NaN kann zu Fehlern im Feature-Engineering führen

**Frage 2.1.3:** Ist meine Zeitreihe lückenlos?

- **Was prüfen:** Zeitabstand zwischen aufeinanderfolgenden Kerzen
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('data/USDCAD_H1.csv')
  df['time'] = pd.to_datetime(df['time'])
  df['time_diff'] = df['time'].diff()
  print(df['time_diff'].value_counts().head(10))
  ```

- **Erwartung:** 99% der Diffs = 1 Stunde (für H1) oder 1 Tag am Wochenende
- **Warum wichtig:** Große Lücken verfälschen Indikatoren (SMA, EMA, RSI)

### 2.2 Feature-Engineering

**Frage 2.2.1:** Welche Features hat mein Modell gelernt?

- **Kommando:**

  ```python
  import joblib
  model = joblib.load('models/lgbm_usdcad_v5.pkl')
  print(model.feature_name_)  # LightGBM
  # oder
  print(model.get_booster().feature_names)  # XGBoost
  ```

- **Warum wichtig:** Fehlende Features im Live-Trading = Fehler

**Frage 2.2.2:** Welche Features sind am wichtigsten?

- **Was prüfen:** Feature Importance der Modelle
- **Kommando:**

  ```python
  import joblib
  import pandas as pd
  model = joblib.load('models/lgbm_usdcad_v5.pkl')
  importance = pd.DataFrame({
      'feature': model.feature_name_,
      'importance': model.feature_importances_
  }).sort_values('importance', ascending=False)
  print(importance.head(20))
  ```

- **Warum wichtig:** Zeigt, auf welche Signale das Modell reagiert

**Frage 2.2.3:** Verwende ich dieselbe Feature-Berechnung im Training und Live?

- **Was prüfen:** `feature_engineering.py` vs. `live_trader.py` (Zeilen 490-720)
- **Kritisch:** Look-Ahead-Bias vermeiden (`.shift(1)` verwenden!)
- **Kommando:** `grep -n "\.shift" features/feature_engineering.py live/live_trader.py`
- **Erwartung:** Beide Dateien verwenden `.shift(1)` bei Rolling Features

**Frage 2.2.4:** Sind meine Multi-Timeframe-Features korrekt berechnet?

- **Was prüfen:** H4/D1-Features in H1-Daten, H1-Features in M5-Daten
- **Kommando:** `grep -A 10 "Multi-Timeframe" live/live_trader.py`
- **Warum wichtig:** Falsche Resampling-Logik = falsches Signal

---

## 3. MODELLPERFORMANCE & METRIKEN

### 3.1 Trainingsmetriken

**Frage 3.1.1:** Wie gut sind meine Modelle auf Test-Daten?

- **Was prüfen:** `logs/train/*.log` für Metriken
- **Kommando:** `grep -E "Test.*F1|Test.*Accuracy" logs/train/*.log | tail -20`
- **Erwartung (laut Review):**
  - F1-Macro: 0.42–0.48 (akzeptabel für 3-Klassen-Problem)
  - Accuracy: ~0.55–0.60
- **Warum wichtig:** Zu niedriger F1 (<0.40) = kein Edge

**Frage 3.1.2:** Gibt es Class-Imbalance in meinen Labels?

- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('data/USDCAD_H1_labeled_v5.csv')
  print(df['label'].value_counts(normalize=True))
  ```

- **Erwartung:**
  - Long (2): 30–40%
  - Neutral (1): 20–30%
  - Short (0): 30–40%
- **Warum wichtig:** Starkes Imbalance (z.B. 10% Long) → Modell ignoriert Minderheitsklasse

**Frage 3.1.3:** Wie oft trainiere ich mein Modell neu?

- **Was prüfen:** `retraining.py` Logik
- **Kommando:** `grep -A 5 "RETRAINING_INTERVAL" retraining.py`
- **Empfehlung (laut Review):** Monatlich + Trigger bei Rolling Sharpe < 0.5
- **Warum wichtig:** Zu häufig = Overfitting, zu selten = veraltetes Modell

**Frage 3.1.4:** Welche Modellversion läuft aktuell im Paper-Trading?

- **Was prüfen:** Batch-Dateien (`*.bat`) und `models/`
- **Kommando:** `grep "two_stage_version" start_*.bat`
- **Aktueller Stand:**
  - USDCAD: v4 (stabil)
  - USDJPY: v5 (Kandidat)
- **Warum wichtig:** Falsche Version = ungetestetes Modell im Live-Betrieb

### 3.2 Walk-Forward-Analyse

**Frage 3.2.1:** Habe ich Walk-Forward-Validierung durchgeführt?

- **Was prüfen:** `walk_forward.py` Ergebnisse
- **Kommando:** `ls -lh reports/walk_forward_*.csv`
- **Erwartung:** Konsistente Performance über mehrere Zeitfenster
- **Warum wichtig:** Modell muss auf ungesehenen Zeiträumen funktionieren

**Frage 3.2.2:** Verschlechtert sich die Performance in Out-of-Sample-Perioden?

- **Was prüfen:** Test-Set-Performance vs. Walk-Forward-Performance
- **Kommando:** `grep "Sharpe" reports/walk_forward_*.csv`
- **Red Flag:** Sharpe > 2.0 im Test-Set, aber < 0.5 in Walk-Forward
- **Warum wichtig:** Zeigt Overfitting

---

## 4. SIGNALGENERIERUNG & ENTSCHEIDUNGSLOGIK

### 4.1 Schwellenwerte (Thresholds)

**Frage 4.1.1:** Welche Signalschwellen verwende ich aktuell?

- **Was prüfen:** `--schwelle` und `--short_schwelle` in Batch-Dateien
- **Kommando:** `grep -E "schwelle|short_schwelle" start_*.bat`
- **Aktueller Stand (Shadow-Compare):**
  - USDCAD: `--schwelle 0.55 --short_schwelle 0.45` (streng)
  - USDJPY: verschiedene Schwellen im Test
- **Warum wichtig:** Höhere Schwelle = weniger Trades, aber höhere Qualität

**Frage 4.1.2:** Wie unterscheiden sich die Schwellen 0.50 / 0.55 / 0.60 in der Praxis?

- **Was analysieren:** Anzahl Trades, Winrate, Profit-Factor
- **Kommando:**

  ```bash
  python scripts/evaluate_threshold_kpis.py \
    --symbols USDCAD,USDJPY \
    --hours 168 \
    --thresholds 0.50,0.55,0.60 \
    --threshold_operator ge
  ```

- **Erwartung:** Höhere Schwelle → weniger Trades, höhere Winrate
- **Entscheidungshilfe:** `reports/threshold_eval/summary.csv`

**Frage 4.1.3:** Ist mein Modell kalibriert?

- **Was prüfen:** Sind die Wahrscheinlichkeiten zuverlässig?
- **Test:**
  - Sammle alle Trades mit `prob >= 0.70`
  - Berechne tatsächliche Winrate
  - Erwartung: ~70% Winrate
- **Wenn nicht kalibriert:**

  ```bash
  python scripts/calibrate_probabilities.py \
    --model_path models/lgbm_usdcad_v5.pkl \
    --method sigmoid \
    --output_model models/lgbm_usdcad_v5_calibrated.pkl
  ```

- **Warum wichtig:** Unkalibrierte Probs = falsche Schwellenwahl

**Frage 4.1.4:** Welche Decision-Mapping verwende ich?

- **Was prüfen:** `--decision_mapping` Parameter
- **Optionen:**
  - `class`: Argmax der 3 Klassen (0=Short, 1=Neutral, 2=Long)
  - `long_prob`: Nur Prob(Long) betrachten → Short bei prob_long < 0.45
- **Empfehlung:** `long_prob` + asymmetrische Schwellen (0.55/0.45)
- **Kommando:** `grep "decision_mapping" start_*.bat`

### 4.2 Regime-Filter

**Frage 4.2.1:** Welche Regime-Detection-Methode verwende ich?

- **Was prüfen:** `--regime_source` Parameter
- **Optionen:**
  - `adx_regime`: ADX-basiert (Trend/Range/Choppy)
  - `market_regime_hmm`: Hidden-Markov-Modell
  - `None`: Kein Regime-Filter
- **Kommando:** `grep "regime_source" start_*.bat`
- **Aktuell:** Meist `adx_regime`

**Frage 4.2.2:** In welchen Regimen handle ich?

- **Was prüfen:** `--regime_filter` Parameter
- **Kommando:** `grep "regime_filter" start_*.bat`
- **Bedeutung:**
  - `0`: Choppy/Range (risikoreicher)
  - `1`: Trend (optimal für ML-Signale)
  - `2`: Strong Trend (kann zu aggressiv sein)
- **Empfehlung:** `--regime_filter 1,2` (nur Trends handeln)

**Frage 4.2.3:** Wie verteilen sich meine Trades über die Regime?

- **Was analysieren:** `logs/*_live_trades.csv`
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('logs/USDCAD_live_trades.csv')
  print(df['regime'].value_counts(normalize=True))
  print(df.groupby('regime')['pnl'].mean())
  ```

- **Ziel:** Die besten Trades sollten in Regime 1/2 liegen

---

## 5. RISIKOMANAGEMENT

### 5.1 Positionsgröße & Lot-Size

**Frage 5.1.1:** Mit welcher Positionsgröße handle ich?

- **Was prüfen:** `--lot_size` Parameter
- **Kommando:** `grep "lot_size" start_*.bat`
- **Empfehlung:** Start mit 0.01 Lot (Micro-Lot = 1000 Einheiten)
- **Warum wichtig:** 1 Lot = 100.000 Einheiten → 0.01 Lot reicht für Paper-Trading

**Frage 5.1.2:** Ist meine Positionsgröße dynamisch oder fix?

- **Was prüfen:** `live_trader.py` – Funktion `berechne_positionsgroesse()`
- **Kommando:** `grep -A 20 "def berechne_positionsgroesse" live/live_trader.py`
- **Empfehlung:** ATR-basiert + Kelly-Criterion für optimale Größe
- **Formel:** `lot_size = (equity * risk%) / (ATR_Stop_in_Pips * pip_value)`

**Frage 5.1.3:** Wie viel Prozent meines Kapitals riskiere ich pro Trade?

- **Was berechnen:** `(Stop-Loss in Pips × Lot-Size × Pip-Value) / Equity × 100`
- **Empfehlung:** 1–2% pro Trade (defensiv), maximal 5%
- **Warum wichtig:** 10% Risk = Ruin bei 10 verlorenen Trades in Folge

### 5.2 Stop-Loss & Take-Profit

**Frage 5.2.1:** Welche Stop-Loss-Methode verwende ich?

- **Was prüfen:** `--atr_sl` Parameter
- **Kommando:** `grep "atr_sl" start_*.bat`
- **Aktuell:** `--atr_sl 1` (aktiviert ATR-Stop-Loss)
- **Berechnung:** `SL = Entry ± (ATR × Multiplikator)`
- **Empfehlung laut Review:** 1.5× ATR

**Frage 5.2.2:** Wie groß ist mein durchschnittlicher Stop-Loss in Pips?

- **Was analysieren:** `logs/*_live_trades.csv` – Spalte `sl_pips`
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('logs/USDCAD_live_trades.csv')
  print(f"Durchschnitt SL: {df['sl_pips'].mean():.1f} Pips")
  print(f"Min/Max SL: {df['sl_pips'].min():.1f} / {df['sl_pips'].max():.1f}")
  ```

- **Erwartung:** 20–60 Pips für USDCAD (abhängig von Volatilität)

**Frage 5.2.3:** Habe ich einen Take-Profit gesetzt?

- **Was prüfen:** `--tp_atr_mult` Parameter
- **Kommando:** `grep "tp_atr_mult" start_*.bat`
- **Empfehlung:** TP = 2–3× Stop-Loss (Risk:Reward = 1:2 oder 1:3)
- **Berechnung:** `TP = Entry ± (ATR × TP_Multiplikator)`

**Frage 5.2.4:** Verwende ich Trailing-Stop?

- **Was prüfen:** `--trailing_stop` Parameter
- **Kommando:** `grep "trailing_stop" start_*.bat`
- **Funktion:** SL nachziehen wenn Trade im Profit ist
- **Warum sinnvoll:** Gewinne sichern bei Trendumkehr

### 5.3 Kill-Switch & Notfall-Stop

**Frage 5.3.1:** Bei wie viel Prozent Drawdown stoppt mein System?

- **Was prüfen:** `--kill_switch_dd` Parameter
- **Kommando:** `grep "kill_switch_dd" start_*.bat`
- **Empfehlung laut Review:** 15% (0.15)
- **Warum wichtig:** Schützt vor Totalverlust bei System-Fehler

**Frage 5.3.2:** Wie oft prüft das System den Kill-Switch?

- **Was prüfen:** `live_trader.py` – Zeile ~1970
- **Kommando:** `grep -A 10 "Kill-Switch prüfen" live/live_trader.py`
- **Erwartung:** Bei jeder neuen Kerze (H1, M5, etc.)

**Frage 5.3.3:** Was passiert bei Auslösung des Kill-Switch?

- **Was prüfen:** `live_trader.py` – Kill-Switch-Aktion
- **Erwartung:**
  - Alle offenen Positionen schließen
  - Trading stoppen
  - Alarm/Log-Eintrag
  - Automatisches Beenden des Skripts

---

## 6. LIVE-TRADING-VERHALTEN

### 6.1 Ausführungsmodus

**Frage 6.1.1:** Bin ich im Paper-Trading oder Live-Modus?

- **Was prüfen:** `--paper_mode` Parameter
- **Kommando:** `grep "paper_mode" start_*.bat`
- **Wichtig:** `1` = Paper-Trading (kein echtes Geld), `0` = Live
- **Regel:** Mindestens 3 Monate Paper + 12 GO-Wochen vor Live!

**Frage 6.1.2:** Wie lange läuft mein Paper-Trading bereits?

- **Was prüfen:** Erste Zeile in `logs/*_live_trades.csv`
- **Kommando:**

  ```bash
  head -2 logs/USDCAD_live_trades.csv
  # Erste Entry-Timestamp prüfen
  ```

- **Erwartung:** Mindestens 3 Monate vor Live-Betrieb

**Frage 6.1.3:** Welche Symbole handle ich aktuell?

- **Was prüfen:** Laufende `live_trader.py` Instanzen
- **Windows:** Task-Manager → Details → `python.exe` Kommandozeile anzeigen
- **Oder:** `ps aux | grep live_trader` (Linux/Git Bash)
- **Aktueller Stand:** USDCAD (v4) + USDJPY (v5)

### 6.2 Ausführungsfrequenz

**Frage 6.2.1:** Wie oft prüft mein System neue Signale?

- **Was prüfen:** `--check_interval` Parameter
- **Kommando:** `grep "check_interval" start_*.bat live/live_trader.py`
- **Bedeutung:** Sekunden zwischen Prüfungen (z.B. 60 = jede Minute)
- **Empfehlung:**
  - H1-Daten: 300s (5 Minuten)
  - M5-Daten: 60s (1 Minute)

**Frage 6.2.2:** Welchen Timeframe verwende ich für Two-Stage LTF?

- **Was prüfen:** `--two_stage_ltf_timeframe` Parameter
- **Kommando:** `grep "two_stage_ltf_timeframe" start_*.bat`
- **Aktuell:** M5 (5-Minuten-Kerzen)
- **Warum wichtig:** M5 = häufigere Signale, aber mehr Noise

**Frage 6.2.3:** Cache ich HTF-Daten oder lade ich sie jedes Mal neu?

- **Was prüfen:** `live_trader.py` – HTF-Cache-Logik (Zeile ~1950)
- **Kommando:** `grep -A 10 "HTF-Cache" live/live_trader.py`
- **Warum wichtig:** Caching = weniger API-Calls, schnellere Ausführung

### 6.3 Logging & Aufzeichnung

**Frage 6.3.1:** Wo werden meine Trades gespeichert?

- **Was prüfen:** `logs/*_live_trades.csv`
- **Kommando:** `ls -lh logs/*.csv`
- **Wichtig:** Diese Dateien sind Basis für Threshold-KPI-Analyse

**Frage 6.3.2:** Welche Informationen werden pro Trade geloggt?

- **Was prüfen:** Spalten in `logs/USDCAD_live_trades.csv`
- **Kommando:** `head -1 logs/USDCAD_live_trades.csv`
- **Erwartung:**
  - Timestamp, Symbol, Signal, Probability
  - Regime, HTF-Bias (bei Two-Stage)
  - Entry, SL, TP, Lot-Size
  - PnL, Fees, Winrate

**Frage 6.3.3:** Werden Fehler geloggt?

- **Was prüfen:** `logs/*.log` oder Console-Output
- **Kommando:** `grep -i "error\|exception" logs/*.log | tail -20`
- **Warum wichtig:** Stille Fehler können Signale verhindern

---

## 7. REGIME-DETECTION & MARKTPHASEN

### 7.1 Regime-Verteilung

**Frage 7.1.1:** Wie oft ist der Markt in welchem Regime?

- **Was analysieren:** Regime-Verteilung in historischen Daten
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('data/USDCAD_H1_labeled_v5.csv')
  print(df['adx_regime'].value_counts(normalize=True))
  ```

- **Erwartung:**
  - Choppy (0): 30–50%
  - Trend (1): 30–40%
  - Strong Trend (2): 10–20%

**Frage 7.1.2:** In welchem Regime ist der Markt JETZT?

- **Was prüfen:** Letzter Log-Eintrag oder Console
- **Kommando:** `tail -1 logs/USDCAD_live_trades.csv | awk -F',' '{print $NF}'`
- **Oder:** Live-Dashboard (falls implementiert)

**Frage 7.1.3:** Performen meine Modelle in allen Regimen gleich gut?

- **Was analysieren:** PnL pro Regime
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('backtest/USDCAD_trades.csv')
  print(df.groupby('regime')['pnl'].agg(['mean', 'sum', 'count']))
  print(df.groupby('regime')['pnl'].apply(lambda x: (x > 0).sum() / len(x)))
  ```

- **Red Flag:** Negative PnL in Regime 0 → Choppy-Filter aktivieren!

### 7.2 HMM-Regime

**Frage 7.2.1:** Verwende ich HMM-Regime-Detection?

- **Was prüfen:** `--regime_source market_regime_hmm`
- **Kommando:** `grep "market_regime_hmm" start_*.bat`
- **Vorteil:** Dynamischer als fixer ADX-Schwellenwert
- **Nachteil:** Kann bei zu wenig Daten fehlschlagen

**Frage 7.2.2:** Funktioniert mein HMM-Fallback?

- **Was prüfen:** `live_trader.py` – Zeile ~690
- **Kommando:** `grep -A 15 "HMM-Regime (optional" live/live_trader.py`
- **Erwartung:** Bei HMM-Fehler → Fallback auf ADX-Regime
- **Test:** HMM-Datei löschen und prüfen ob System weiterläuft

---

## 8. TWO-STAGE-ARCHITEKTUR

### 8.1 Modellkombination

**Frage 8.1.1:** Ist Two-Stage aktiviert?

- **Was prüfen:** `--two_stage_enable` Parameter
- **Kommando:** `grep "two_stage_enable" start_*.bat`
- **Aktuell:** `1` für USDCAD + USDJPY

**Frage 8.1.2:** Welche Modelle kombiniere ich?

- **Was prüfen:** `models/` Verzeichnis
- **Kommando:**

  ```bash
  ls -lh models/lgbm_htf_bias_*  # HTF-Bias-Modelle (H1)
  ls -lh models/lgbm_ltf_entry_*  # LTF-Entry-Modelle (M5)
  ```

- **Erwartung:** Für jedes Symbol (USDCAD/USDJPY) und Version (v4/v5) je 2 Modelle

**Frage 8.1.3:** Wie funktioniert die Kongruenz-Prüfung?

- **Was prüfen:** `--two_stage_kongruenz` Parameter + Code
- **Kommando:** `grep -A 20 "Kongruenz-Filter" live/live_trader.py`
- **Logik:**
  - HTF sagt "Long" → LTF darf nur Long-Signal geben
  - HTF sagt "Short" → LTF darf nur Short-Signal geben
  - HTF sagt "Neutral" → kein Trade
- **Warum wichtig:** Filtert widersprüchliche Signale

**Frage 8.1.4:** Wie oft stimmen HTF und LTF überein?

- **Was analysieren:** `logs/*_live_trades.csv` – Spalte `htf_bias`
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('logs/USDCAD_live_trades.csv')
  match = (df['signal'] == df['htf_bias']).sum()
  total = len(df)
  print(f"Kongruenz-Rate: {match/total*100:.1f}%")
  ```

- **Erwartung:** >80% Übereinstimmung

### 8.2 Fallback-Strategie

**Frage 8.2.1:** Was passiert wenn LTF-Modell fehlt?

- **Was prüfen:** `live_trader.py` – Zeile ~1090
- **Kommando:** `grep -A 10 "Hard Fallback" live/live_trader.py`
- **Erwartung:** Automatischer Fallback auf Single-Stage (nur HTF)

**Frage 8.2.2:** Wird der Fallback geloggt?

- **Was prüfen:** Log-Einträge "Fallback aktiviert"
- **Kommando:** `grep -i "fallback" logs/*.log`
- **Warum wichtig:** Zeigt Modell-Verfügbarkeitsprobleme

---

## 9. BACKTESTING & VALIDIERUNG

### 9.1 Backtest-Ergebnisse

**Frage 9.1.1:** Welche Performance hat mein System im Backtest?

- **Was prüfen:** `backtest/*_summary.csv`
- **Kommando:**

  ```bash
  cat backtest/two_stage_backtest_summary_v5_20260305.csv
  ```

- **Key-Metriken:**
  - Sharpe Ratio: >1.5 (gut), >2.0 (exzellent)
  - Max Drawdown: <20%
  - Profit Factor: >1.5
  - Winrate: >50% (für Trend-Strategie)

**Frage 9.1.2:** Sind meine Backtest-Renditen realistisch?

- **Was prüfen:** Spreads und Kommissionen eingerechnet?
- **Kommando:** `grep "spread" backtest/backtest.py backtest/two_stage_backtest.py`
- **Empfehlung laut Review:** `--spread_faktor 2.0` für Stress-Test
- **Warum wichtig:** Zu optimistische Backtests = böse Überraschung im Live

**Frage 9.1.3:** Habe ich Survivorship-Bias?

- **Was prüfen:** Teste ich nur die 2 besten Paare (USDCAD + USDJPY)?
- **Lösung:** Durchschnitt über alle 7 Paare als Benchmark
- **Kommando:** `python backtest/backtest.py --symbols EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCHF,USDCAD,USDJPY`
- **Erwartung:** System sollte auf Average-7-Performance designt sein

### 9.2 Trade-Analyse

**Frage 9.2.1:** Wie viele Trades generiert mein System pro Monat?

- **Was analysieren:** `backtest/*_trades.csv`
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('backtest/USDCAD_M5_two_stage_trades.csv')
  df['entry_time'] = pd.to_datetime(df['entry_time'])
  monthly = df.groupby(df['entry_time'].dt.to_period('M')).size()
  print(f"Durchschnitt: {monthly.mean():.0f} Trades/Monat")
  print(f"Min/Max: {monthly.min()} / {monthly.max()}")
  ```

- **Erwartung:** 10–40 Trades/Monat (für H1-basiert), 50–150 (für M5)

**Frage 9.2.2:** Wie lange halte ich Trades durchschnittlich?

- **Was analysieren:** `exit_time - entry_time`
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('backtest/USDCAD_trades.csv')
  df['entry_time'] = pd.to_datetime(df['entry_time'])
  df['exit_time'] = pd.to_datetime(df['exit_time'])
  df['hold_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
  print(f"Durchschnittliche Haltedauer: {df['hold_hours'].mean():.1f}h")
  ```

- **Erwartung:** 10–50 Stunden für Swing-Trading

**Frage 9.2.3:** Gibt es Zeitfenster mit besonders vielen Verlusttrades?

- **Was analysieren:** PnL nach Wochentag, Tageszeit
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('backtest/USDCAD_trades.csv')
  df['entry_time'] = pd.to_datetime(df['entry_time'])
  df['weekday'] = df['entry_time'].dt.day_name()
  df['hour'] = df['entry_time'].dt.hour
  print(df.groupby('weekday')['pnl'].mean().sort_values())
  print(df.groupby('hour')['pnl'].mean().sort_values())
  ```

- **Mögliche Aktion:** Zeitfilter (`--trading_hours`) setzen

---

## 10. MONITORING & WARTUNG

### 10.1 Echtzeit-Überwachung

**Frage 10.1.1:** Wie überwache ich mein System während es läuft?

- **Was prüfen:** Dashboard, Log-Tailing, Telegram-Bot
- **Kommando:** `tail -f logs/USDCAD_live_trades.csv` (Linux/Git Bash)
- **Empfehlung:** Dashboard mit Plotly Dash für Echtzeit-KPIs

**Frage 10.1.2:** Erhalte ich Benachrichtigungen bei Trades?

- **Was implementieren:** Telegram-Bot oder E-Mail-Alerts
- **Warum sinnvoll:** Sofortiges Feedback bei neuen Trades/Fehlern

**Frage 10.1.3:** Wie erkenne ich Modell-Degradation?

- **Was tracken:** Rolling Sharpe Ratio über 30 Tage
- **Kommando:**

  ```python
  import pandas as pd
  df = pd.read_csv('logs/USDCAD_live_trades.csv')
  df['pnl_cumsum'] = df['pnl'].cumsum()
  rolling_sharpe = df['pnl'].rolling(30).mean() / df['pnl'].rolling(30).std()
  print(f"Aktueller 30-Tage-Sharpe: {rolling_sharpe.iloc[-1]:.2f}")
  ```

- **Trigger:** Sharpe < 0.5 → Retraining einleiten

### 10.2 Wartungsintervalle

**Frage 10.2.1:** Wie oft prüfe ich mein System?

- **Empfehlung:**
  - Täglich: Quick-Check (läuft es noch?)
  - Wöchentlich: Performance-Review (Gewinn/Verlust, Sharpe)
  - Monatlich: Retraining, Walk-Forward-Update
- **Tool:** `scripts/evaluate_threshold_kpis.py --hours 168` (wöchentlich)

**Frage 10.2.2:** Wann lösche ich alte Logs?

- **Was prüfen:** Speicherplatz in `logs/`
- **Kommando:** `du -sh logs/`
- **Empfehlung:** Logs älter als 6 Monate archivieren (komprimieren, nicht löschen!)

**Frage 10.2.3:** Habe ich ein Backup-Konzept?

- **Was sichern:**
  - Modelle (`models/*.pkl`)
  - Logs (`logs/*.csv`)
  - Konfiguration (`.env`, `*.bat`)
- **Frequenz:** Wöchentlich auf externen Server/Cloud

---

## 11. FEHLERBEHANDLUNG & STABILITÄT

### 11.1 Verbindungsprobleme

**Frage 11.1.1:** Was passiert bei MT5-Verbindungsabbruch?

- **Was prüfen:** `live_trader.py` – Reconnection-Logik
- **Kommando:** `grep -A 10 "MT5.initialize" live/live_trader.py`
- **Erwartung:** Automatischer Reconnect nach 5–10 Sekunden
- **Test:** MT5 beenden während Trader läuft

**Frage 11.1.2:** Was passiert bei API-Timeouts (Fear & Greed, BTC Funding)?

- **Was prüfen:** `feature_engineering.py` – Fallback-Werte
- **Kommando:** `grep -A 5 "fallback" features/feature_engineering.py live/live_trader.py`
- **Erwartung laut Review:**
  - Fear & Greed → 50 (Neutral)
  - BTC Funding → 0.0

**Frage 11.1.3:** Loggt das System Exceptions?

- **Was prüfen:** Try-Except-Blöcke und Logging
- **Kommando:** `grep -n "except.*Exception" live/live_trader.py | wc -l`
- **Erwartung:** Alle kritischen Bereiche abgesichert

### 11.2 Datenfehler

**Frage 11.2.1:** Was passiert bei NaN im Live-Feature-DataFrame?

- **Was prüfen:** `live_trader.py` – NaN-Handling
- **Kommando:** `grep -A 5 "dropna\|fillna" live/live_trader.py`
- **Erwartung:** Entweder fillna(0) oder Signal überspringen

**Frage 11.2.2:** Was passiert wenn MT5 keine neuen Daten liefert?

- **Was prüfen:** Timeout-Logik bei `MT5.copy_rates_from_pos()`
- **Erwartung:** Max. 3 Retries, dann Signal überspringen

---

## 12. PERFORMANCE-OPTIMIERUNG

### 12.1 Latenz & Geschwindigkeit

**Frage 12.1.1:** Wie lange dauert ein Signal-Zyklus (Daten laden → Signal → Ausführung)?

- **Was messen:** Timestamps im Log
- **Kommando:**

  ```python
  # In live_trader.py am Anfang und Ende des Signal-Zyklus:
  import time
  start = time.time()
  # ... Signal-Generierung ...
  print(f"Signal-Zyklus: {time.time() - start:.2f}s")
  ```

- **Ziel:** <5 Sekunden für H1, <2 Sekunden für M5

**Frage 12.1.2:** Welche Teile sind die langsamsten?

- **Was profilen:** Feature-Berechnung, Modell-Inferenz, MT5-API
- **Tool:** Python `cProfile`

  ```bash
  python -m cProfile -s cumtime live/live_trader.py --symbol USDCAD > profile.txt
  ```

- **Erwartung:** Modell-Inferenz < 100ms, Feature-Calc < 1s

### 12.2 Speicher & Ressourcen

**Frage 12.2.1:** Wie viel RAM verbraucht mein Trader?

- **Windows:** Task-Manager → `python.exe` Memory
- **Linux:** `ps aux | grep live_trader | awk '{print $6}'`
- **Erwartung:** <500 MB pro Trader-Instanz

**Frage 12.2.2:** Gibt es Memory-Leaks?

- **Was testen:** System 72h laufen lassen, RAM-Verbrauch loggen
- **Tool:** `psutil` in Python

  ```python
  import psutil
  import os
  process = psutil.Process(os.getpid())
  print(f"RAM: {process.memory_info().rss / 1024**2:.1f} MB")
  ```

---

## 📊 AUSWERTUNGS-CHECKLISTE

### Wöchentliche Quick-Checks

- [ ] System läuft ohne Crashes?
- [ ] Neue Trades in letzten 7 Tagen?
- [ ] Winrate > 45%?
- [ ] Rolling Sharpe (30d) > 0.5?
- [ ] Max Drawdown < 15%?
- [ ] Keine kritischen Fehler in Logs?

### Monatliche Deep-Dive

- [ ] Threshold-KPI-Analyse durchgeführt?
- [ ] Regime-Performance-Matrix aktualisiert?
- [ ] Walk-Forward-Validierung durchgeführt?
- [ ] Modell-Retraining falls Sharpe < 0.5?
- [ ] Feature Importance geprüft?
- [ ] Backups erstellt?
- [ ] Logs archiviert?

### Vor Live-Deployment (3 Monate Paper)

- [ ] Mindestens 3 Monate Paper-Trading?
- [ ] 12 GO-Wochen erreicht?
- [ ] Sharpe Ratio > 1.5?
- [ ] Max Drawdown < 20%?
- [ ] Profit Factor > 1.8?
- [ ] Winrate > 50%?
- [ ] Reality-Check durchgeführt?
- [ ] Kill-Switch getestet?
- [ ] Backup-Strategie steht?

---

## 🎯 NÄCHSTE SCHRITTE

Nach Durcharbeiten dieses Fragebogens solltest du:

1. **Schwachstellen identifiziert haben** (z.B. zu niedrige Winrate in Regime 0)
2. **Optimierungspotenziale erkannt haben** (z.B. Threshold von 0.50 auf 0.55 erhöhen)
3. **Monitoring-Lücken geschlossen haben** (z.B. Telegram-Bot einrichten)
4. **Klare Aktionspunkte haben** (z.B. Retraining planen, Logs archivieren)

### Empfohlene Reihenfolge für Erstkonfiguration

1. **Datenqualität sicherstellen** (Abschnitt 2)
2. **Modellperformance validieren** (Abschnitt 3)
3. **Signalschwellen optimieren** (Abschnitt 4)
4. **Risikomanagement konfigurieren** (Abschnitt 5)
5. **Monitoring aufsetzen** (Abschnitt 10)
6. **Paper-Trading starten** (Abschnitt 6)
7. **3 Monate beobachten & optimieren** (Abschnitt 10–12)
8. **Live-Deployment nach GO-Kriterien** (Checkliste oben)

---

**Viel Erfolg beim Feintuning deines Systems! 🚀**

Bei Fragen zu spezifischen Bereichen kannst du mich jederzeit fragen.
