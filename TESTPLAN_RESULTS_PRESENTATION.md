# 📊 Backtest-Ergebnisse: 30 Konfigurationen (USDCAD + USDJPY)

**Testlauf durchgeführt:** 9. März 2026, 12:10-12:15 Uhr  
**Testzeitraum (Daten):** Oktober 2025 – März 2026 (~5 Monate)  
**Modell:** Two-Stage v4 (HTF=H1 / LTF=M5)  
**Symbole:** USDCAD + USDJPY  

---

## 🏆 TOP-10 KONFIGURATIONEN (sortiert nach Ø Sharpe Ratio)

| # | Test | Ø Sharpe | Ø Return | Ø MaxDD | Total Trades | Ø PF | Beschreibung |
|---|------|----------|----------|---------|--------------|------|--------------|
| **1** | **15** | **7.60** | **20.01%** | **-1.74%** | **1133** | **3.19** | **Hohe Schwelle (0.60)** |
| **2** | **14** | **7.41** | **24.30%** | **-2.08%** | **1396** | **3.16** | **Schwelle 58%** |
| **3** | **27** | **7.19** | **19.96%** | **-1.85%** | **1230** | **3.15** | **Konservativ (0.58 + cooldown 18)** |
| 4 | 16 | 7.02 | 25.73% | -3.01% | 1914 | 2.90 | RRR 1.5:1 |
| 5 | 13 | 6.56 | 26.20% | -2.67% | 1660 | 2.75 | Schwelle 56% |
| 6 | 20 | 6.48 | 17.39% | -3.12% | 2026 | 2.99 | Symmetrisch 0.5% (TP=SL) |
| 7 | 19 | 6.48 | 17.39% | -3.12% | 2026 | 2.99 | Symmetrisch 0.4% (TP=SL) |
| 8 | **2** | 6.36 | 27.65% | -2.55% | 1797 | 2.63 | **Top-Config (neu empfohlen, schwelle=0.55)** |
| 9 | 25 | 6.35 | 12.83% | -1.40% | 790 | 2.69 | Schwelle 56% + Trending |
| 10 | 6 | 6.31 | 22.00% | -2.62% | 1865 | 2.48 | Top + ATR 1.0x |

---

## 📌 DETAILANALYSE: TOP-3 KONFIGURATIONEN

### 🥇 #1: Test 15 – Hohe Schwelle (0.60)

**Parameter:**
- `--schwelle 0.60`
- `--tp_pct 0.006, --sl_pct 0.003` (RRR 2:1)
- `--cooldown_bars 12`
- `--regime_filter 0,1,2` (alle außer High-Vola)
- `--atr_faktor 1.5`
- `--horizon 24`

**Performance pro Symbol:**
|  | USDCAD | USDJPY | Ø |
|--|--------|--------|---|
| **Sharpe Ratio** | 4.21 | 10.99 | **7.60** |
| **Return** | 7.60% | 32.42% | **20.01%** |
| **Max Drawdown** | -2.12% | -1.36% | **-1.74%** |
| **Trades** | 437 | 696 | 1133 total |

**Warum ist diese Config besser?**
- ✅ **Sehr niedriger Drawdown** (-1.74% vs. -2.55% bei Test 2)
- ✅ **Hohe Sharpe Ratio** → bessere risikoadjustierte Rendite
- ✅ **USDJPY performa exzellent** (Sharpe 10.99!)
- ⚠️ Weniger Trades (1133) → mehr Geduld benötigt

---

### 🥈 #2: Test 14 – Schwelle 58%

**Parameter:**
- `--schwelle 0.58`
- `--tp_pct 0.006, --sl_pct 0.003` (RRR 2:1)
- `--cooldown_bars 12`
- `--regime_filter 0,1,2`
- `--atr_faktor 1.5`
- `--horizon 24`

**Performance pro Symbol:**
|  | USDCAD | USDJPY | Ø |
|--|--------|--------|---|
| **Sharpe Ratio** | 3.70 | 11.13 | **7.41** |
| **Return** | 8.39% | 40.22% | **24.30%** |
| **Max Drawdown** | -2.58% | -1.58% | **-2.08%** |
| **Trades** | 527 | 869 | 1396 total |

**Warum ist diese Config besser?**
- ✅ **Höchste Return** (24.30%) von allen Top-3
- ✅ **USDJPY Sharpe** noch besser als Test 15 (11.13)
- ✅ **Gute Balance** zwischen Trades (1396) und Selektivität
- ⚠️ Etwas höherer Drawdown als Test 15

---

### 🥉 #3: Test 27 – Konservativ (Hohe Schwelle + Mehr Cooldown)

**Parameter:**
- `--schwelle 0.58`
- `--tp_pct 0.006, --sl_pct 0.003` (RRR 2:1)
- `--cooldown_bars 18` ← mehr als Standard (12)
- `--regime_filter 0,1,2`
- `--atr_faktor 1.5`
- `--horizon 24`

**Performance pro Symbol:**
|  | USDCAD | USDJPY | Ø |
|--|--------|--------|---|
| **Sharpe Ratio** | 3.14 | 11.25 | **7.19** |
| **Return** | 6.65% | 33.27% | **19.96%** |
| **Max Drawdown** | -2.53% | -1.17% | **-1.85%** |
| **Trades** | 451 | 779 | 1230 total |

**Warum ist diese Config besser?**
- ✅ **Höchste USDJPY Sharpe** (11.25) aller Configs
- ✅ **Sehr niedriger USDJPY Drawdown** (-1.17%)
- ✅ **Cooldown 18** verhindert Overtrading
- ⚠️ USDCAD Return niedriger (6.65%)

---

## 🔍 WICHTIGE ERKENNTNISSE

### 1. **Höhere Schwellen sind besser**
Die Tests zeigen klar: **schwelle=0.58-0.60** schlägt alle niedrigeren Schwellen (0.50-0.55).

| Schwelle | Platzierung | Ø Sharpe |
|----------|-------------|----------|
| 0.60 | #1 | 7.60 |
| 0.58 | #2 | 7.41 |
| 0.58 + cd18 | #3 | 7.19 |
| 0.56 | #5 | 6.56 |
| **0.55** | **#8** | **6.36** |
| 0.54 | #12 | 5.87 |
| 0.52 | #11 | 5.63 |

**→ Die ursprünglich empfohlene schwelle=0.55 (Test 2) ist nur #8 im Ranking!**

---

### 2. **RRR 2:1 ist optimal**

Tests mit anderen RRR (1:1, 1.5:1, 2.5:1, 3:1) performen schlechter:

| RRR | Test | Platzierung | Ø Sharpe |
|-----|------|-------------|----------|
| **2:1** | **15** | **#1** | **7.60** |
| 1.5:1 | 16 | #4 | 7.02 |
| 2.5:1 | 17 | #16 | 5.47 |
| 3.0:1 | 18 | #24 | 4.13 |
| 1:1 | 1 | #11 | 5.63 |

**→ RRR 2:1 (TP=0.6%, SL=0.3%) ist der Sweet-Spot!**

---

### 3. **Cooldown 12 ist Standard, Cooldown 18 noch besser für Konservative**

| Cooldown | Test | Platzierung | Ø Sharpe | Ø Trades |
|----------|------|-------------|----------|----------|
| 18 | 27 | #3 | 7.19 | 1230 |
| 12 | 2 | #8 | 6.36 | 1797 |
| 6 | 3 | #17 | 5.38 | 2416 |

**→ Mehr Cooldown = höhere Sharpe, aber weniger Trades**

---

### 4. **Regime-Filter: Alle (0,1,2) besser als nur Trending (1,2)**

Tests mit `--regime_filter 1,2` (nur Trending) performen deutlich schlechter:

| Regime-Filter | Test | Platzierung | Ø Sharpe | Ø Trades |
|---------------|------|-------------|----------|----------|
| **0,1,2 (alle)** | **15** | **#1** | **7.60** | **1133** |
| 1,2 (trending) | 5 | #13 | 5.85 | 690 |
| 0,1,2,3 (inkl. Vola) | 30 | #21 | 4.81 | 1992 |

**→ Alle Regime außer High-Vola (0,1,2) ist optimal!**

---

### 5. **ATR-Faktor 1.5x ist optimal**

| ATR-Faktor | Test | Platzierung | Ø Sharpe |
|------------|------|-------------|----------|
| **1.5x** | **15** | **#1** | **7.60** |
| 1.0x | 6 | #10 | 6.31 |
| 2.0x | 7 | #18 | 5.20 |

**→ ATR 1.5x ist die beste Balance zwischen Schutz und Profit-Capture!**

---

## 🎯 EMPFEHLUNG

### **Beste Konfiguration für Paper-Trading:**

```batch
--symbol USDCAD/USDJPY
--version v4
--schwelle 0.60
--tp_pct 0.006
--sl_pct 0.003
--cooldown_bars 12
--regime_filter 0,1,2
--atr_sl 1
--atr_faktor 1.5
--horizon 24
--two_stage_enable 1
--two_stage_ltf_timeframe M5
```

### **Alternative: Konservativ (besserer DD)**

Für Trader, die noch niedrigeren Drawdown bevorzugen:

```batch
--schwelle 0.58
--cooldown_bars 18
(alle anderen Parameter identisch)
```

---

## 📊 VOLLSTÄNDIGE ERGEBNISSE

Alle 30 Test-Ergebnisse mit Details findest du in:
- **[backtest/testplan_results_latest.csv](backtest/testplan_results_latest.csv)** (alle Symbole+Metriken)
- **[backtest/testplan_ranking_latest.csv](backtest/testplan_ranking_latest.csv)** (aggregiertes Ranking)
- **[backtest/testplan_30configs.csv](backtest/testplan_30configs.csv)** (Testplan-Übersicht)

---

## ⚠️ WICHTIG: Nächste Schritte

### 1. **BAT-Datei aktualisieren**
Die aktuelle `start_testphase_topconfig.bat` nutzt `--schwelle 0.55` (Platz #8).  
**→ Aktualisieren auf `--schwelle 0.60` (Platz #1)!**

### 2. **Paper-Trading starten**
Nach BAT-Update:
1. BAT-Datei auf Windows-Laptop kopieren
2. `start_testphase_topconfig.bat` ausführen
3. Mindestens **2 Wochen Paper-Trading** durchlaufen
4. KPI-Gates prüfen (Sharpe > 0.8, PF > 1.3, MaxDD < 10%)

### 3. **Monitoring**
- Täglich: MT5-Dashboard prüfen
- Wöchentlich: Logs analysieren (`verify_live_log_sync.py`)
- Nach 2 Wochen: Walk-Forward-Validierung

---

**Ende der Präsentation** | Stand: 9. März 2026  
**Testlauf-Dauer:** ~5 Minuten (30 Configs × ~10s pro Test)  
**Getesteter Zeitraum:** 5 Monate (Okt 2025 – März 2026)
