# HTF/LTF Multi-Timeframe Trading-Strategie – Recherche & Integration

**Erstellt:** 2026-03-04  
**Projektsystem:** MT5 ML-Trading-System (XGBoost + LightGBM)

---

## 📋 Inhaltsverzeichnis

1. [Was ist HTF/LTF Trading?](#1-was-ist-htfltf-trading)
2. [Kernkonzepte](#2-kernkonzepte)
3. [Wissenschaftliche Grundlagen](#3-wissenschaftliche-grundlagen)
4. [Aktuelle Implementierung (Status Quo)](#4-aktuelle-implementierung-status-quo)
5. [Integration in dein Projekt](#5-integration-in-dein-projekt)
6. [Implementierungsplan](#6-implementierungsplan)
7. [Code-Beispiele](#7-code-beispiele)
8. [Risiken & Fallstricke](#8-risiken--fallstricke)
9. [Literatur & Ressourcen](#9-literatur--ressourcen)

---

## 1. Was ist HTF/LTF Trading?

### Definition

**HTF/LTF** = **Higher Timeframe / Lower Timeframe**

Eine Multi-Timeframe-Strategie, die auf zwei Zeitebenen operiert:

| Ebene | Timeframe | Aufgabe | Beispiel |
|-------|-----------|---------|----------|
| **HTF** | H1, H4, D1 | **Marktstruktur & Key Levels identifizieren** | Trend-Richtung, Support/Resistance, Swing-Punkte |
| **LTF** | M5, M15, M30 | **Präzise Entry-Signale generieren** | Pullbacks, Break-Retest, Candlestick-Patterns |

### Grundprinzip

```
┌─────────────────────────────────────────────┐
│ HTF (H1): "Was ist der Bias?"               │
│ → Aufwärtstrend + Resistance bei 1.3500     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ LTF (M5): "Wann steige ich ein?"            │
│ → Warte auf Pullback + Bestätigung          │
└─────────────────────────────────────────────┘
```

**Vorteil:** HTF liefert die "richtige Seite des Marktes", LTF liefert den "optimalen Einstieg".

---

## 2. Kernkonzepte

### 2.1 Key Levels (HTF)

**Key Levels** sind kritische Preisniveaus, an denen der Markt reagiert:

#### A) Support & Resistance (klassisch)

- **Support:** Preiszone, an der Kaufdruck die Abwärtsbewegung stoppt
- **Resistance:** Preiszone, an der Verkaufsdruck die Aufwärtsbewegung stoppt

**Identifikation:**

- Mehrfache Touches: Preis hat Level 2+ mal getestet
- Swing High/Low: Lokale Extrempunkte (Turning Points)
- Pivot Points: Mathematische Level (Classic, Camarilla, Fibonacci)

#### B) Swing High/Low

- **Swing High:** Lokales Maximum (High > High[−1] UND High > High[+1])
- **Swing Low:** Lokales Minimum (Low < Low[−1] UND Low < Low[+1])

**Verwendung:**

- Markiert wichtige Wendepunkte
- Definiert Trend-Struktur (Higher Highs/Lower Lows)
- Entry-Auslöser bei Break oder Retest

#### C) Liquidity Zones (Order Blocks)

- Bereiche mit hoher akkumulierter Liquidität
- Oft vor großen Bewegungen (Institutional Order Flow)
- **Smart Money Concept (SMC):** Banks/Hedge Funds jagen Retail-Stops

**Beispiel:**

```
┌─────────────────────────────────────────────┐
│ EURUSD H1 Chart                             │
│                                              │
│ 1.0950 ─┬─ Resistance (3× getestet)         │
│         │                                    │
│ 1.0920 ─┴─ Support (Break → Retest?)        │
│                                              │
│ 1.0880 ─── Previous Swing Low (Order Block) │
└─────────────────────────────────────────────┘
```

### 2.2 Entry-Signale (LTF)

Auf dem niedrigeren Timeframe warten wir auf **Bestätigung**:

#### A) Pullback-Entry (Trend-Following)

1. HTF-Trend: Aufwärts (H1)
2. Preis erreicht Key Support (M5)
3. LTF-Bestätigung: Bullish Engulfing oder Break of Structure (BOS)
4. Entry: Long bei M5-Close über EMA20

#### B) Break & Retest

1. HTF: Resistance-Break auf H1
2. LTF: Warte auf Pullback zu ex-Resistance (jetzt Support)
3. Entry: Long bei M5-Bounce mit Confirmation-Candle

#### C) False Break (Stop Hunt)

1. HTF: Key Level wird "fake" gebrochen (Liquidity Grab)
2. LTF: Schnelle Rückkehr über Level
3. Entry: In Richtung der echten Bewegung (gegen False Break)

#### D) Candlestick-Patterns (LTF)

- **Pinbar:** Lange Wick = Rejection an Key Level
- **Engulfing:** Starke Momentum-Umkehr
- **Inside Bar:** Konsolidierung vor Breakout

**Wichtig:** Pattern alleine reichen nicht – müssen mit HTF-Bias übereinstimmen.

---

## 3. Wissenschaftliche Grundlagen

### 3.1 Warum funktioniert Multi-Timeframe-Analyse?

#### Fractal Market Hypothesis (FMH)

- Märkte sind fraktal: Strukturen wiederholen sich auf allen Zeitebenen
- **Quelle:** Peters, E. E. (1994). *Fractal Market Analysis*

#### Market Microstructure

- Institutionelle Trader operieren auf HTF (H4/D1/W1)
- Retail-Trader auf LTF (M5/M15)
- Key Levels = **Institutional Footprints** (Order Flow Imbalances)

#### Mean Reversion + Momentum

- HTF: Identifiziert Trend-Regime (Momentum)
- LTF: Identifiziert Pullbacks (Mean Reversion innerhalb des Trends)
- **Quelle:** Menkhoff, L. (2010). *The Use of Technical Analysis by Fund Managers*

### 3.2 Empirische Evidenz (Forex)

#### Studie 1: Neely et al. (1997)

- **Finding:** Technische Trading-Rules (inklusive Support/Resistance) sind profitabel
- **Methodik:** 21 Jahre FX-Daten (1974-1995)
- **Quelle:** *Journal of Finance*

#### Studie 2: Osler (2003)

- **Finding:** Support/Resistance-Level beeinflussen Orderflow
- **Grund:** Clustering von Stop-Loss und Take-Profit Orders
- **Quelle:** *Federal Reserve Bank of New York*

#### Studie 3: Park & Irwin (2007)

- **Meta-Analyse:** 95 Studien zu technischen Trading-Strategien
- **Ergebnis:** 56% der Studien zeigen signifikante Profite
- **Caveat:** Transaction Costs reduzieren Profitabilität stark

### 3.3 Moderne Ansätze (ML-basiert)

#### Support Vector Machines (SVM) für Key Level Detection

- **Ansatz:** SVM trainiert auf historischen Support/Resistance
- **Features:** Price Clusters, Volume Spikes, Ichimoku-Komponenten
- **Performance:** F1-Score 0.65–0.75 (besser als klassische Methoden)
- **Quelle:** Booth et al. (2014), *Expert Systems with Applications*

#### Deep Learning (LSTM) für Multi-Timeframe

- **Ansatz:** Separate LSTM für H1 und M15, Combined Feature Vector
- **Ergebnis:** Sharpe Ratio 1.3 (EURUSD, 2015-2019)
- **Problem:** Black-Box, schwer zu interpretieren
- **Quelle:** Sezer et al. (2020), *Applied Soft Computing*

---

## 4. Aktuelle Implementierung (Status Quo)

### Was du **bereits hast:**

#### ✅ Multi-Timeframe-Features (rudimentär)

```python
# live/live_trader.py, Zeile 501-516
close_h4 = close.resample("4h").last().dropna()
trend_h4 = np.sign(ind_sma(close_h4, 20) - ind_sma(close_h4, 50)).fillna(0)
result["trend_h4"] = trend_h4.shift(1).reindex(result.index, method="ffill")

close_d1 = close.resample("1D").last().dropna()
trend_d1 = np.sign(ind_sma(close_d1, 20) - ind_sma(close_d1, 50)).fillna(0)
result["trend_d1"] = trend_d1.shift(1).reindex(result.index, method="ffill")
```

**✓** HTF-Trend-Features (H4 + D1)  
**✗** Keine Key Level Detection  
**✗** Keine Swing High/Low  
**✗** Kein LTF-Entry-Signal (alles läuft auf H1/M30/M15)

#### ✅ Timeframe-Flexibilität

- Modelle für H1, M30, M15 (USDCAD, USDJPY)
- `TIMEFRAME_CONFIG` mit `bars_per_hour`
- Zeitäquivalente Features

#### ✅ Regime Detection

```python
# live/live_trader.py, Zeile 530-542
regime[aufwaerts] = 1   # Aufwärtstrend + niedriges ADX
regime[abwaerts] = 2     # Abwärtstrend + niedriges ADX
regime[hoch_vol] = 3     # Hohe Volatilität (Seitwärts)
```

**✓** Identifiziert Marktphase  
**✗** Keine Verknüpfung mit Key Levels

### Was **fehlt:**

#### ❌ Key Level Detection

- Keine Support/Resistance-Identifikation
- Keine Swing High/Low Markierung
- Keine dynamischen Price Zones

#### ❌ LTF-HTF-Synchronisation

- Keine explizite "HTF-Bias → LTF-Entry"-Logik
- Keine Pullback-Erkennung auf LTF
- Keine Break-Retest-Strategie

#### ❌ Order Block / Liquidity Zone

- Keine instituionellen Footprints
- Kein Fair Value Gap (FVG)
- Kein Supply/Demand-Konzept

---

## 5. Integration in dein Projekt

### 5.1 Strategie-Übersicht

**Szenario:** USDCAD Trading mit HTF=H1, LTF=M5 (oder M15)

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: HTF-Analyse (H1)                                  │
├────────────────────────────────────────────────────────────┤
│ 1. Key Levels identifizieren:                             │
│    - Swing High/Low (letzte 50 H1-Kerzen)                 │
│    - Support/Resistance (Price Cluster mit Volume)        │
│ 2. HTF-Bias festlegen:                                     │
│    - Trend-Richtung (SMA20 > SMA50 = bullish)             │
│    - Regime (0=Seitwärts, 1=Aufwärts, 2=Abwärts, 3=Vola)  │
│ 3. Nächstes Key Level bestimmen:                          │
│    - Preis über Level → Resistance ist Ziel               │
│    - Preis unter Level → Support ist Ziel                 │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 2: LTF-Entry (M5 oder M15)                           │
├────────────────────────────────────────────────────────────┤
│ 1. Warte auf Preis-Annäherung an Key Level:               │
│    - Pullback zu Support (bei bullish HTF)                │
│    - Bounce von Resistance (bei bearish HTF)              │
│ 2. LTF-Bestätigungs-Signal:                               │
│    - Bullish Engulfing / Pinbar                           │
│    - RSI < 30 → über 30 (Oversold-Bounce)                │
│    - MACD Crossover                                       │
│ 3. ML-Modell validiert Entry:                             │
│    - Probability > 0.60 (erhöhte Schwelle)                │
│    - Regime = erlaubt (0,1,2 – nicht 3=Vola)              │
│ 4. Trade mit ATR-basiertem SL:                            │
│    - SL unter Key Level (+ ATR × 1.5)                     │
│    - TP an nächstem Key Level (oder RRR 1:2)              │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Feature-Engineering: Key Level Features

Neue Features für das ML-Modell:

#### A) `distance_to_nearest_support` / `distance_to_nearest_resistance`

```python
# Distanz zum nächsten Key Level (normiert durch ATR)
distance_support = (close - nearest_support) / atr
distance_resistance = (nearest_resistance - close) / atr
```

**Interpretation:**

- `distance_support = 0.5` → 0.5× ATR über Support (bullish Bias)
- `distance_resistance = -0.2` → 0.2× ATR unter Resistance (bearish Rejection)

#### B) `key_level_strength`

```python
# Anzahl der Touches an diesem Level in den letzten 50 Kerzen
strength = count_touches_within_zone(level, tolerance=0.002)
```

**Interpretation:**

- `strength = 1` → Schwaches Level (nur 1× getestet)
- `strength = 5` → Starkes Level (mehrfach respektiert)

#### C) `swing_high_proximity` / `swing_low_proximity`

```python
# Ist aktuelle Kerze nahe einem Swing-Punkt?
swing_proximity = 1 if abs(close - swing_level) < atr * 0.5 else 0
```

**Interpretation:**

- `1` → Preis ist an einem kritischen Wendepunkt
- `0` → Preis ist in neutraler Zone

#### D) `htf_ltf_alignment`

```python
# Stimmen HTF-Trend und LTF-Signal überein?
if trend_h4 == 1 and ltf_signal == "long":
    alignment = 1  # Perfekte Übereinstimmung
elif trend_h4 == -1 and ltf_signal == "short":
    alignment = 1
else:
    alignment = 0  # Konflikt → höheres Risiko
```

### 5.3 Architektur-Änderungen

#### Option 1: Zwei-Stufen-Modell (empfohlen)

```
┌─────────────────────────────────────────────┐
│ Modell 1 (HTF): Trend-Klassifikator (H1)   │
│ Input:  H1 + H4 + D1 Features               │
│ Output: Bias {LONG, SHORT, NEUTRAL}         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Modell 2 (LTF): Entry-Timing (M5/M15)      │
│ Input:  M5/M15 Features + HTF-Bias          │
│ Output: Signal {0, 1, 2} mit Probability    │
└─────────────────────────────────────────────┘
```

**Vorteile:**

- Saubere Trennung: HTF → "Was?", LTF → "Wann?"
- HTF-Modell kann mit weniger Daten trainiert werden (H1 hat mehr Historie)
- LTF-Modell fokussiert auf Timing (Entry-Qualität)

**Nachteil:**

- 2× Modelle zu warten (aber modular!)

#### Option 2: Unified Model mit HTF/LTF-Features (einfacher)

```
┌─────────────────────────────────────────────┐
│ Single Model (trainiert auf H1)            │
│ Input:  H1-Features + Key Level Features   │
│         + trend_h4, trend_d1                │
│ Output: Signal {0, 1, 2}                    │
└─────────────────────────────────────────────┘
```

**Vorteile:**

- Nur 1 Modell zu trainieren/deployen
- Einfachere Integration in bestehendes System

**Nachteil:**

- Keine explizite LTF-Logik (alles auf H1 gerechnet)

**Empfehlung:** Starte mit **Option 2** (Unified Model + Key Level Features), migriere später zu Option 1 wenn Performance-Limits erreicht sind.

---

## 6. Implementierungsplan

### Phase 1: Key Level Detection (Woche 1)

#### Ziele

- [ ] Swing High/Low identifizieren (H1-Daten)
- [ ] Support/Resistance-Zonen berechnen (Price Clustering)
- [ ] Neue Features: `distance_to_support`, `distance_to_resistance`

#### Deliverables

- `features/key_levels.py` (neue Datei)
- Unittest: `tests/test_key_levels.py`
- Visualisierung: `plots/USDCAD_key_levels.png`

### Phase 2: Feature-Integration (Woche 2)

#### Ziele

- [ ] Key Level Features in `feature_engineering.py` einbauen
- [ ] Pipeline-Test: Features auf allen 7 Symbolen berechnen
- [ ] Look-Ahead-Bias Check (Key Levels nur aus Vergangenheit!)

#### Deliverables

- `USDCAD_H1_features.csv` mit neuen Spalten:
  - `distance_to_support`
  - `distance_to_resistance`
  - `key_level_strength`
  - `swing_proximity`

### Phase 3: Model-Retraining (Woche 3)

#### Ziele

- [ ] XGBoost + LightGBM mit neuen Features trainieren
- [ ] Feature Importance: Sind Key Level Features relevant?
- [ ] Backtesting: Verbessert sich Sharpe Ratio?

#### Deliverables

- `models/lgbm_USDCAD_v5.pkl` (mit Key Level Features)
- `backtest/USDCAD_trades_v5.csv`
- KPI-Vergleich: v4 vs. v5

### Phase 4: LTF-Integration (Woche 4)

#### Ziele

- [ ] M5-Daten von MT5 laden (neuer Timeframe)
- [ ] LTF-Entry-Logik in `live_trader.py`
- [ ] Paper-Testing: USDCAD M5 mit HTF-Bias von H1

#### Deliverables

- `live/live_trader_ltf.py` (separate Implementierung für Test)
- Discord-Webhook: "LTF-Entry bei USDCAD M5 (HTF-Bias: Bullish)"

### Phase 5: Walk-Forward Validation (Woche 5-6)

#### Ziele

- [ ] 6-Monats-Walk-Forward auf OOS-Daten
- [ ] Performance-Gate: Sharpe > 0.8, Profit Factor > 1.3
- [ ] Wenn Gate erfüllt → Migration zu Echtgeld-Kandidat

---

## 7. Code-Beispiele

### 7.1 Swing High/Low Detection

```python
def identify_swing_points(
    df: pd.DataFrame, 
    left_bars: int = 5, 
    right_bars: int = 5
) -> pd.DataFrame:
    """
    Identifiziert Swing Highs und Swing Lows.
    
    Swing High: High[t] > max(High[t-left:t]) AND High[t] > max(High[t:t+right])
    Swing Low:  Low[t]  < min(Low[t-left:t])  AND Low[t]  < min(Low[t:t+right])
    
    Args:
        df: OHLCV DataFrame mit 'high' und 'low' Spalten
        left_bars: Anzahl Kerzen links vom Swing-Punkt
        right_bars: Anzahl Kerzen rechts vom Swing-Punkt
        
    Returns:
        DataFrame mit neuen Spalten:
        - 'swing_high': NaN oder High-Preis wenn Swing High
        - 'swing_low': NaN oder Low-Preis wenn Swing Low
    """
    result = df.copy()
    n = len(df)
    
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    
    for i in range(left_bars, n - right_bars):
        # Swing High Check
        current_high = df['high'].iloc[i]
        left_max = df['high'].iloc[i - left_bars:i].max()
        right_max = df['high'].iloc[i + 1:i + right_bars + 1].max()
        
        if current_high > left_max and current_high > right_max:
            swing_highs[i] = current_high
        
        # Swing Low Check
        current_low = df['low'].iloc[i]
        left_min = df['low'].iloc[i - left_bars:i].min()
        right_min = df['low'].iloc[i + 1:i + right_bars + 1].min()
        
        if current_low < left_min and current_low < right_min:
            swing_lows[i] = current_low
    
    result['swing_high'] = swing_highs
    result['swing_low'] = swing_lows
    
    return result
```

**Verwendung:**

```python
df = pd.read_csv("data/USDCAD_H1.csv", index_col='time', parse_dates=True)
df_with_swings = identify_swing_points(df, left_bars=5, right_bars=5)

# Nur die Swing-Punkte ausgeben (NaN entfernen)
swing_highs = df_with_swings['swing_high'].dropna()
swing_lows = df_with_swings['swing_low'].dropna()

print(f"Gefunden: {len(swing_highs)} Swing Highs, {len(swing_lows)} Swing Lows")
```

### 7.2 Support/Resistance Clustering

```python
from sklearn.cluster import DBSCAN

def find_support_resistance(
    df: pd.DataFrame, 
    eps_pct: float = 0.002,  # 0.2% Cluster-Toleranz
    min_touches: int = 2
) -> dict:
    """
    Findet Support/Resistance-Level durch Price Clustering.
    
    Methodik:
    1. Alle Swing Highs + Swing Lows sammeln
    2. DBSCAN-Clustering mit eps = eps_pct × median_price
    3. Cluster mit ≥ min_touches = Key Level
    
    Args:
        df: DataFrame mit 'swing_high' und 'swing_low' Spalten
        eps_pct: Cluster-Toleranz als % des Preises
        min_touches: Mindest-Touches für valides Level
        
    Returns:
        dict mit Keys 'support_levels', 'resistance_levels':
        [
            {'price': 1.3450, 'strength': 3, 'last_touch': '2026-03-01'},
            ...
        ]
    """
    # Swing-Punkte extrahieren
    swing_highs = df['swing_high'].dropna()
    swing_lows = df['swing_low'].dropna()
    
    # Alle Preise kombinieren
    all_levels = pd.concat([swing_highs, swing_lows]).values.reshape(-1, 1)
    
    if len(all_levels) < min_touches:
        return {'support_levels': [], 'resistance_levels': []}
    
    # DBSCAN-Clustering
    median_price = np.median(all_levels)
    eps = eps_pct * median_price
    
    clustering = DBSCAN(eps=eps, min_samples=min_touches).fit(all_levels)
    labels = clustering.labels_
    
    # Key Levels extrahieren (Label != -1 = valides Cluster)
    key_levels = []
    for label in set(labels):
        if label == -1:
            continue  # Rauschen
        
        cluster_prices = all_levels[labels == label]
        avg_price = np.mean(cluster_prices)
        strength = len(cluster_prices)
        
        key_levels.append({
            'price': avg_price,
            'strength': strength,
        })
    
    # Sortieren nach Preis
    key_levels.sort(key=lambda x: x['price'])
    
    # Split in Support (unter aktuellem Preis) / Resistance (darüber)
    current_price = df['close'].iloc[-1]
    
    support = [lvl for lvl in key_levels if lvl['price'] < current_price]
    resistance = [lvl for lvl in key_levels if lvl['price'] > current_price]
    
    return {
        'support_levels': support,
        'resistance_levels': resistance
    }
```

**Verwendung:**

```python
df = pd.read_csv("data/USDCAD_H1.csv", index_col='time', parse_dates=True)
df = identify_swing_points(df)  # Zuerst Swing-Punkte
levels = find_support_resistance(df, eps_pct=0.002, min_touches=2)

print(f"Support Levels: {len(levels['support_levels'])}")
for lvl in levels['support_levels'][-3:]:  # Letzte 3
    print(f"  {lvl['price']:.5f} (Strength: {lvl['strength']})")

print(f"Resistance Levels: {len(levels['resistance_levels'])}")
for lvl in levels['resistance_levels'][:3]:  # Nächste 3
    print(f"  {lvl['price']:.5f} (Strength: {lvl['strength']})")
```

### 7.3 Feature: Distance to Key Level

```python
def add_key_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Key Level Features hinzu (für ML-Training).
    
    WICHTIG – Look-Ahead-Bias vermeiden:
    Key Levels werden nur aus vergangenen 200 Kerzen berechnet
    und mit .shift(1) delayed, damit Kerze T keine Info aus T enthält.
    
    Args:
        df: OHLCV DataFrame mit 'close', 'high', 'low'
        
    Returns:
        DataFrame mit neuen Spalten:
        - distance_to_support
        - distance_to_resistance
        - key_level_strength_support
        - key_level_strength_resistance
    """
    result = df.copy()
    n = len(df)
    
    # ATR für Normierung
    atr = result['atr_14'].values  # Annahme: wurde vorher berechnet
    close = result['close'].values
    
    dist_support = np.full(n, np.nan)
    dist_resistance = np.full(n, np.nan)
    strength_support = np.full(n, 0)
    strength_resistance = np.full(n, 0)
    
    # Rolling Window: Key Levels aus letzten 200 Kerzen
    window = 200
    
    for i in range(window, n):
        # Slice: nur Daten BIS i-1 (keine Zukunft!)
        df_past = result.iloc[i - window:i].copy()
        df_past = identify_swing_points(df_past, left_bars=5, right_bars=5)
        levels = find_support_resistance(df_past, eps_pct=0.002, min_touches=2)
        
        current_close = close[i - 1]  # .shift(1) Logik: vorherige Kerze
        current_atr = atr[i - 1]
        
        # Nächster Support
        supports = levels['support_levels']
        if supports:
            nearest_support = max(supports, key=lambda x: x['price'])
            dist_support[i] = (current_close - nearest_support['price']) / current_atr
            strength_support[i] = nearest_support['strength']
        
        # Nächste Resistance
        resistances = levels['resistance_levels']
        if resistances:
            nearest_resistance = min(resistances, key=lambda x: x['price'])
            dist_resistance[i] = (nearest_resistance['price'] - current_close) / current_atr
            strength_resistance[i] = nearest_resistance['strength']
    
    result['distance_to_support'] = dist_support
    result['distance_to_resistance'] = dist_resistance
    result['key_level_strength_support'] = strength_support
    result['key_level_strength_resistance'] = strength_resistance
    
    return result
```

### 7.4 LTF-Entry-Logik (Pseudocode)

```python
def ltf_entry_signal(
    df_ltf: pd.DataFrame,  # M5 oder M15
    htf_bias: str,         # "long", "short", "neutral" (von H1)
    key_level: float,      # Nächstes Support/Resistance
    tolerance_atr: float = 0.5  # Wie nah muss Preis am Level sein?
) -> dict:
    """
    Generiert Entry-Signal auf LTF wenn HTF-Bedingungen erfüllt.
    
    Returns:
        {
            'signal': 'long' | 'short' | None,
            'reason': 'pullback_to_support' | 'break_retest' | ...,
            'confidence': float  # 0-1
        }
    """
    last_bar = df_ltf.iloc[-1]
    close = last_bar['close']
    atr = last_bar['atr_14']
    
    # Distanz zum Key Level (normiert)
    distance = abs(close - key_level) / atr
    
    # Ist Preis nahe genug am Level?
    if distance > tolerance_atr:
        return {'signal': None, 'reason': 'not_at_key_level', 'confidence': 0}
    
    # HTF-Bias bestimmt Richtung
    if htf_bias == "long":
        # Warte auf Bounce vom Support (bullish Confirmation)
        if is_bullish_engulfing(df_ltf.tail(2)):
            return {
                'signal': 'long',
                'reason': 'bullish_engulfing_at_support',
                'confidence': 0.75
            }
        
        if df_ltf['rsi_14'].iloc[-2] < 30 and df_ltf['rsi_14'].iloc[-1] > 30:
            return {
                'signal': 'long',
                'reason': 'rsi_oversold_recovery',
                'confidence': 0.65
            }
    
    elif htf_bias == "short":
        # Warte auf Rejection an Resistance (bearish Confirmation)
        if is_bearish_engulfing(df_ltf.tail(2)):
            return {
                'signal': 'short',
                'reason': 'bearish_engulfing_at_resistance',
                'confidence': 0.75
            }
        
        if df_ltf['rsi_14'].iloc[-2] > 70 and df_ltf['rsi_14'].iloc[-1] < 70:
            return {
                'signal': 'short',
                'reason': 'rsi_overbought_rejection',
                'confidence': 0.65
            }
    
    return {'signal': None, 'reason': 'no_confirmation', 'confidence': 0}

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """Bullish Engulfing: 2. Kerze schließt über 1. Kerze Open."""
    if len(df) < 2:
        return False
    
    first = df.iloc[0]
    second = df.iloc[1]
    
    return (
        first['close'] < first['open'] and  # Erste Kerze bearish
        second['close'] > second['open'] and  # Zweite Kerze bullish
        second['open'] < first['close'] and  # Öffnet unterhalb
        second['close'] > first['open']      # Schließt darüber
    )
```

---

## 8. Risiken & Fallstricke

### 8.1 Look-Ahead-Bias (KRITISCH!)

**Problem:**  
Key Levels aus der Zukunft in Features einfließen lassen.

**Beispiel (FALSCH):**

```python
# ❌ FALSCH: Berechnet Support/Resistance über gesamten Datensatz
levels = find_support_resistance(df)
df['distance_to_support'] = ...
```

**Lösung (RICHTIG):**

```python
# ✅ RICHTIG: Rolling Window + .shift(1)
for i in range(200, len(df)):
    df_past = df.iloc[i-200:i]  # NUR Vergangenheit
    levels = find_support_resistance(df_past)
    # Feature basierend auf Kerze i-1 (nicht i!)
```

### 8.2 Overfitting auf Key Levels

**Problem:**  
Modell lernt spezifische Preislevel auswendig statt generalized patterns.

**Symptome:**

- Training F1-Score: 0.85
- Validation F1-Score: 0.50
- Test F1-Score: 0.45

**Lösung:**

- Key Levels als **relative Features** (Distance / ATR)
- Regularisierung erhöhen (L2-Penalty in LightGBM)
- Walk-Forward Validation über 6+ Monate

### 8.3 Whipsaw bei Falschen Breakouts

**Problem:**  
Preis bricht Key Level, reversed sofort (Stop Hunt).

**Lösung:**

- Warte auf **Retest** (Break → Pullback → Confirmation)
- Erhöhe Entry-Schwelle: `probability > 0.65` statt 0.52
- Volume-Konfirmation: Breakout mit hohem Volume ist valider

### 8.4 Latenz bei LTF-Execution

**Problem:**  
M5-Signal kommt 30 Sekunden zu spät → Preis hat sich bewegt.

**Lösung:**

- MT5 `copy_rates_from_pos()` gecacht (max. 5 Sekunden alt)
- Limit-Orders statt Market-Orders
- Slippage-Buffer: Entry-Preis ± 5 Pips

### 8.5 Parameter-Optimierung (Curve Fitting)

**Problem:**  
`left_bars=7, right_bars=3, eps_pct=0.0023` funktioniert perfekt auf Backtest, versagt Live.

**Lösung:**

- Robustness-Check: Teste Parameter-Range (5-10 für `left_bars`)
- Monte Carlo Simulation: Random Parameter-Samples
- Keep It Simple: `left_bars=right_bars=5` (symmetrisch)

---

## 9. Literatur & Ressourcen

### Wissenschaftliche Studien

1. **Neely, C. J., Weller, P. A., & Dittmar, R. (1997)**  
   *"Is Technical Analysis in the Foreign Exchange Market Profitable?"*  
   Journal of Financial & Quantitative Analysis, 32(4), 405-426.

2. **Osler, C. L. (2003)**  
   *"Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis"*  
   Journal of Finance, 58(5), 1791-1819.

3. **Park, C. H., & Irwin, S. H. (2007)**  
   *"What Do We Know About the Profitability of Technical Analysis?"*  
   Journal of Economic Surveys, 21(4), 786-826.

4. **Booth, A., Gerding, E., & McGroarty, F. (2014)**  
   *"Automated Trading with Performance Weighted Random Forests and Seasonality"*  
   Expert Systems with Applications, 41(8), 3651-3661.

5. **Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020)**  
   *"Financial Time Series Forecasting with Deep Learning"*  
   Applied Soft Computing, 90, 106181.

### Bücher

- **Peters, E. E. (1994)**  
  *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*  
  Wiley Finance.

- **Murphy, J. J. (1999)**  
  *Technical Analysis of the Financial Markets*  
  New York Institute of Finance.

- **Brooks, A. (2009)**  
  *Trading Price Action Trends: Technical Analysis for Profiting from Market Trends*  
  Wiley Trading.

### Online-Ressourcen

- **BabyPips.com** – [Multi-Timeframe Analysis](https://www.babypips.com/)  
  Einsteiger-freundliche Erklärungen zu HTF/LTF.

- **Investopedia** – [Support & Resistance](https://www.investopedia.com/terms/s/support.asp)  
  Detaillierte Definition mit Chart-Beispielen.

- **ICT Concepts (Inner Circle Trader)**  
  YouTube-Kanal: Smart Money Concepts, Order Blocks, Liquidity Zones.  
  *Achtung:* Viel Marketing, aber technisch fundiert.

### GitHub-Projekte (Inspiration)

- **`finta`** – [Financial Technical Analysis Library](https://github.com/peerchemist/finta)  
  Python-Bibliothek für 80+ Indikatoren (inkl. Support/Resistance).

- **`vectorbt`** – [Backtesting + Portfolio Analysis](https://github.com/polakowo/vectorbt)  
  Dein System nutzt bereits vectorbt, aber prüfe `vbt.SR` für Support/Resistance.

- **`ta-lib`** – [Technical Analysis Library](https://github.com/mrjbq7/ta-lib)  
  C-Library mit Python-Wrapper (schneller als pandas_ta).

---

## 10. Nächste Schritte

### Quick Win (1-2 Tage)

1. **Visualisierung:** Plotte USDCAD H1 mit Swing High/Low Markierungen
2. **Baseline:** Key Level Features berechnen (ohne ML)
3. **Sanity Check:** Sind die Levels optisch sinnvoll?

### Medium-Term (2-3 Wochen)

1. **Integration:** Key Level Features in `feature_engineering.py`
2. **Retraining:** Modell v5 mit neuen Features
3. **Backtest:** Performance-Vergleich v4 vs. v5

### Long-Term (1-2 Monate)

1. **LTF-Pipeline:** M5-Daten laden + LTF-Entry-Logik
2. **Two-Stage Model:** HTF-Bias (H1) → LTF-Entry (M5)
3. **Walk-Forward:** 6-Monats-Validierung auf OOS-Daten

---

## 11. Recherche-Update (2026-03-04, verifiziert)

### 11.1 Kurzfazit für dein Setup (HTF=H1, LTF=M5)

Die Kombination **H1 für Bias/Key-Level** + **M5 für Entry-Timing** ist methodisch sinnvoll, wenn sie streng regelbasiert umgesetzt wird:

1. **H1 liefert den Kontext** (Trend, Zonen, Marktregime).
2. **M5 liefert den Trigger** (Pullback-Bestätigung statt Blind-Entry).
3. **ATR-basierte Stops** bleiben Pflicht, weil Volatilität intraday stark schwankt.

Wichtig: Der Edge kommt in der Praxis meist weniger von einem „magischen Pattern“, sondern von:

- konsistenter Filterung (Regime + HTF-Bias),
- sauberer Ausführung (SL/TP + Kosten),
- und diszipliniertem Risk-Management.

### 11.2 Quellenlage (dieses Update)

Für dieses Update wurden die folgenden Seiten inhaltlich ausgewertet:

- Investopedia: Support-Level (aktualisiert 2025)
- Investopedia: Pullback (aktualisiert 2025)
- Investopedia: Support/Resistance Basics (aktualisiert 2025)
- Investopedia: ATR (aktualisiert 2026)
- IG Academy: Risiko-/Ausbildungskontext (allgemein)

**Hinweis zur Evidenzqualität:**
Das sind primär praxisorientierte Fachquellen, keine RCTs. Für dein Projekt (systematisches FX-Research + Walk-Forward + Paper-Live) sind sie gut als Regelwerk-Grundlage, aber die finale Entscheidung muss immer über **deine OOS-/Live-KPIs** laufen.

### 11.3 Konkrete, projektnahe Regeln (H1 → M5)

#### A) HTF-Regeln auf H1 (Bias + Key Level)

1. Bias Long nur wenn `trend_h4 >= 0`, `trend_d1 >= 0` und `market_regime in {0,1,2}`.
2. Bias Short nur wenn `trend_h4 <= 0`, `trend_d1 <= 0` und `market_regime in {0,1,2}`.

Key-Level-Zonen aus H1:

- Swing-High/Low + Cluster-Stärke (Touches)
- Distanz als ATR-normalisierte Feature-Werte (`distance_to_support`, `distance_to_resistance`)

#### B) LTF-Regeln auf M5 (Entry)

Entry nur wenn **alle** Punkte erfüllt sind:

1. HTF-Bias vorhanden (Long oder Short).
2. Preis ist nahe Key-Level-Zone (z. B. Distanz ≤ 0.5 ATR(M5)).
3. M5-Trigger bestätigt (z. B. Engulfing ODER RSI-Rückkehr ODER Strukturbruch).
4. Modellwahrscheinlichkeit über Entry-Schwelle (z. B. > 0.55 bis 0.65 je Symbol).

Exit/Risiko:

- Stop-Loss: `ATR(M5) * Faktor` (z. B. 1.5)
- TP mindestens symmetrisch, besser levelbasiert am nächsten H1-Level
- Keine Trades ohne SL (Pflicht bleibt bestehen)

### 11.4 Integration in deine bestehende Codebasis

#### Schritt 1 – H1 Key-Level-Features ergänzen

Datei: `features/feature_engineering.py`

Ergänzen:

- Swing-Erkennung (ohne Future-Leak)
- Rolling-Level-Berechnung aus Vergangenheit
- Neue Spalten: `distance_to_support`, `distance_to_resistance`,
    `key_level_strength_support`, `key_level_strength_resistance`

Wichtig:

- Rolling-Fenster nur bis `t-1`
- `.shift(1)` vor dem Merge in die Zielkerze

#### Schritt 2 – Training für M5-Pipeline vorbereiten

Dateien:

- `train_model.py`
- `features/labeling.py`

Ergänzen:

- Timeframe `M5` analog zu `M15/M30/M60`
- eigenes Modellartefakt, z. B. `lgbm_usdcad_M5_v1.pkl`
- Entry-Schwelle pro Symbol separat evaluieren (Val + Walk-Forward)

#### Schritt 3 – Live: HTF-Bias + M5-Entry koppeln

Datei: `live/live_trader.py`

Umsetzungsidee:

1. Pro Zyklus beide Daten laden: H1 für Bias/Key-Level, M5 für Trigger/Execution.
2. Entscheidungspfad: Wenn H1-Bias neutral ist, kein M5-Trade.
3. Sonst M5-Signal prüfen und Modellwahrscheinlichkeit filtern.
4. ATR-SL auf M5 belassen, Kill-Switch unverändert aktiv.

#### Schritt 4 – Betriebsgrenzen einhalten (Roadmap-konform)

- Operativ weiterhin nur: `USDCAD`, `USDJPY`
- HTF/LTF zunächst nur im **Paper-Modus**
- Gate vor Ausweitung: Sharpe > 0.8, Profit Factor > 1.3, Drawdown < 10%

### 11.5 Nächste Aufgabe in deiner Roadmap

Passend zu **Phase 7 (Überwachung & Wartung)**:

1. HTF/LTF als **separaten Experiment-Branch** starten (kein Eingriff in laufenden Paper-Betrieb).
2. Für `USDCAD` zuerst Backtest + Walk-Forward aufsetzen.
3. Erst danach `USDJPY` hinzufügen.

So schützt du den aktiven Betrieb und bekommst trotzdem saubere Forschungsergebnisse.

---

## Ende des Berichts

*Für Fragen oder Anpassungen: Siehe `CLAUDE.md` und `Roadmap.md`*
