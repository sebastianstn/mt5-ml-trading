"""
feature_engineering.py – Technische Indikatoren und Features berechnen.

Läuft auf: Linux-Server
Input:     data/EURUSD_H1.csv
Output:    data/EURUSD_H1_features.csv

Abhängigkeiten: NUR numpy + pandas (kein pandas_ta!)
    pip install numpy pandas

WICHTIG – Look-Ahead-Bias-Prävention:
    Alle Indikatoren verwenden NUR Daten bis zur aktuellen Kerze (t).
    Das bedeutet: Feature[t] darf KEINE Information aus Kerze t+1 enthalten.
    - Alle eigenen Indikatoren sind standardmäßig korrekt (kein Future-Leak)
    - Multi-Timeframe-Features werden mit .shift(1) gesichert

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/feature_engineering.py
"""
# pylint: disable=duplicate-code

# Standard-Bibliotheken
import logging
from pathlib import Path

# Datenverarbeitung
import numpy as np
import pandas as pd

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 0. Eigene Indikator-Implementierungen (ersetzt pandas_ta)
# ============================================================
#
# Alle Funktionen arbeiten auf pd.Series und geben pd.Series zurück.
# Sie nutzen ausschließlich numpy und pandas – keine externen
# Bibliotheken nötig. Kein Look-Ahead-Bias möglich, da rolling()
# standardmäßig nur vergangene Werte verwendet.


def ind_sma(series: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Berechnet den arithmetischen Mittelwert der letzten `length` Kerzen.
    Gibt NaN zurück, solange weniger als `length` Werte vorhanden sind.

    Args:
        series: Preisreihe (z.B. Close-Preise)
        length: Anzahl der Perioden

    Returns:
        pd.Series mit SMA-Werten.
    """
    return series.rolling(window=length, min_periods=length).mean()


def ind_ema(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Neuere Werte werden stärker gewichtet als ältere. Der Glättungsfaktor
    alpha = 2 / (length + 1). `adjust=False` sorgt für das klassische
    EMA-Verhalten (wie z.B. in MetaTrader).

    Args:
        series: Preisreihe
        length: Anzahl der Perioden

    Returns:
        pd.Series mit EMA-Werten.
    """
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def ind_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD – Moving Average Convergence/Divergence.

    macd_line   = EMA(fast) - EMA(slow)     → Momentum-Indikator
    macd_signal = EMA(signal) der MACD-Linie → Trigger-Linie
    macd_hist   = macd_line - macd_signal    → Stärke des Signals

    Args:
        series: Preisreihe (Close)
        fast:   Periode der schnellen EMA (Standard: 12)
        slow:   Periode der langsamen EMA (Standard: 26)
        signal: Periode der Signal-EMA (Standard: 9)

    Returns:
        DataFrame mit Spalten: macd_line, macd_signal, macd_hist
    """
    ema_fast = ind_ema(series, fast)
    ema_slow = ind_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    )


def ind_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI – Relative Strength Index (Wilder-Methode).

    Misst die Stärke der letzten Aufwärts- vs. Abwärtsbewegungen.
    - Werte > 70: überkauft (potenzielles Verkaufssignal)
    - Werte < 30: überverkauft (potenzielles Kaufsignal)

    Verwendet den Wilder-Smoothing-Algorithmus (EWM mit alpha=1/length),
    identisch zur MetaTrader-Berechnung.

    Args:
        series: Preisreihe (Close)
        length: RSI-Periode (Standard: 14)

    Returns:
        pd.Series mit RSI-Werten (0–100).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder-Smoothing: alpha = 1/length (entspricht EMA mit span=2*length-1)
    avg_gain = gain.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ind_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K und %D).

    %K misst, wo der aktuelle Close innerhalb der letzten `k` Kerzen liegt:
        %K = (Close - Low_k) / (High_k - Low_k) * 100

    %D ist der SMA(d) von %K (Signal-Linie, geglättet).
    Werte > 80: überkauft, Werte < 20: überverkauft.

    Args:
        high:  High-Preise
        low:   Low-Preise
        close: Close-Preise
        k:     %K-Periode (Standard: 14)
        d:     %D-Glättungsperiode (Standard: 3)

    Returns:
        DataFrame mit Spalten: stoch_k, stoch_d
    """
    low_min = low.rolling(window=k, min_periods=k).min()
    high_max = high.rolling(window=k, min_periods=k).max()

    band = (high_max - low_min).replace(0, np.nan)
    stoch_k = (close - low_min) / band * 100
    stoch_d = stoch_k.rolling(window=d, min_periods=d).mean()

    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def ind_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Williams %R.

    Ähnlich wie Stochastic, aber invertiert (Skala: -100 bis 0).
    - Werte > -20: überkauft
    - Werte < -80: überverkauft

    %R = (High_max - Close) / (High_max - Low_min) * -100

    Args:
        high:   High-Preise
        low:    Low-Preise
        close:  Close-Preise
        length: Perioden-Fenster (Standard: 14)

    Returns:
        pd.Series mit Williams-%R-Werten (-100 bis 0).
    """
    high_max = high.rolling(window=length, min_periods=length).max()
    low_min = low.rolling(window=length, min_periods=length).min()

    band = (high_max - low_min).replace(0, np.nan)
    return (high_max - close) / band * -100


def ind_roc(series: pd.Series, length: int = 10) -> pd.Series:
    """
    ROC – Rate of Change.

    Prozentuale Preisveränderung über `length` Perioden:
        ROC = (Close_t - Close_{t-length}) / Close_{t-length} * 100

    Positiv = Preis gestiegen, Negativ = Preis gefallen.

    Args:
        series: Preisreihe
        length: Rückblick-Perioden (Standard: 10)

    Returns:
        pd.Series mit ROC-Werten in Prozent.
    """
    return series.pct_change(periods=length) * 100


def ind_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    ATR – Average True Range (Wilder-Smoothing).

    True Range ist das Maximum aus:
        1. High - Low                  (normale Kerzenbreite)
        2. |High - Close_prev|         (Aufwärtslücke)
        3. |Low  - Close_prev|         (Abwärtslücke)

    ATR = Wilder-EMA(True Range, length)

    Args:
        high:   High-Preise
        low:    Low-Preise
        close:  Close-Preise
        length: Glättungsperiode (Standard: 14)

    Returns:
        pd.Series mit ATR-Werten (in Preiseinheiten, z.B. Pips * 0.0001).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder-Smoothing: identisch zu MetaTrader
    atr = true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    return atr


def ind_bbands(
    series: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Mittleres Band  = SMA(length)
    Oberes Band     = SMA + std * Standardabweichung
    Unteres Band    = SMA - std * Standardabweichung

    Args:
        series: Preisreihe (Close)
        length: SMA-Periode (Standard: 20)
        std:    Faktor für die Standardabweichung (Standard: 2.0)

    Returns:
        DataFrame mit Spalten: bb_upper, bb_mid, bb_lower
    """
    bb_mid = series.rolling(window=length, min_periods=length).mean()
    bb_std = series.rolling(window=length, min_periods=length).std()

    bb_upper = bb_mid + std * bb_std
    bb_lower = bb_mid - std * bb_std

    return pd.DataFrame(
        {
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
        }
    )


def ind_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV – On-Balance Volume.

    Addiert das Volumen wenn der Kurs steigt, subtrahiert es wenn er fällt.
    Trendbestätigung: OBV sollte dem Preistrend folgen.

    OBV_t = OBV_{t-1} + Volumen  wenn Close_t > Close_{t-1}
    OBV_t = OBV_{t-1} - Volumen  wenn Close_t < Close_{t-1}
    OBV_t = OBV_{t-1}             wenn Close_t == Close_{t-1}

    Args:
        close:  Close-Preise
        volume: Volumen-Daten (Tick-Volumen bei MT5)

    Returns:
        pd.Series mit kumulativem OBV.
    """
    direction = np.sign(close.diff())  # +1, -1 oder 0
    return (direction * volume).fillna(0).cumsum()


# ============================================================
# 1. Daten laden
# ============================================================


def daten_laden(dateipfad: Path) -> pd.DataFrame:
    """
    Lädt OHLCV-Rohdaten aus CSV.

    Args:
        dateipfad: Pfad zur EURUSD_H1.csv

    Returns:
        DataFrame mit DatetimeIndex (UTC), sortiert aufsteigend.
    """
    df = pd.read_csv(dateipfad, index_col="time", parse_dates=True)
    df.sort_index(inplace=True)
    logger.info(
        "Rohdaten geladen: %s Kerzen (%s – %s)",
        f"{len(df):,}",
        df.index[0],
        df.index[-1],
    )
    return df


# ============================================================
# 2. Trend-Features
# ============================================================


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Trend-Indikatoren.

    SMA/EMA zeigen die Marktrichtung, MACD misst die Dynamik des Trends.
    Price-to-SMA-Ratio: Wie weit ist der Preis vom Durchschnitt entfernt?

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit neuen Trend-Spalten.
    """
    result = df.copy()

    # --- Simple Moving Averages ---
    result["sma_20"] = ind_sma(result["close"], 20)
    result["sma_50"] = ind_sma(result["close"], 50)
    result["sma_200"] = ind_sma(result["close"], 200)

    # Preis-zu-SMA Ratio: positive Werte = Preis über dem SMA (bullish)
    result["price_sma20_ratio"] = (
        result["close"] - result["sma_20"]
    ) / result["sma_20"]
    result["price_sma50_ratio"] = (
        result["close"] - result["sma_50"]
    ) / result["sma_50"]
    result["price_sma200_ratio"] = (
        result["close"] - result["sma_200"]
    ) / result["sma_200"]

    # SMA-Crossover: 1 = SMA20 über SMA50 (bullish), -1 = bearish
    result["sma_20_50_cross"] = np.sign(
        result["sma_20"] - result["sma_50"]
    ).fillna(0)
    result["sma_50_200_cross"] = np.sign(
        result["sma_50"] - result["sma_200"]
    ).fillna(0)

    # --- Exponential Moving Averages ---
    result["ema_12"] = ind_ema(result["close"], 12)
    result["ema_26"] = ind_ema(result["close"], 26)

    # EMA Crossover: 1 = EMA12 über EMA26 (bullish Signal)
    result["ema_cross"] = np.sign(result["ema_12"] - result["ema_26"]).fillna(
        0
    )

    # --- MACD (12, 26, 9) ---
    macd = ind_macd(result["close"], fast=12, slow=26, signal=9)
    result["macd_line"] = macd["macd_line"]
    result["macd_signal"] = macd["macd_signal"]
    result["macd_hist"] = macd["macd_hist"]

    logger.info("Trend-Features: SMA(20,50,200), EMA(12,26), MACD(12,26,9) ✓")
    return result


# ============================================================
# 3. Momentum-Features
# ============================================================


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Momentum-Indikatoren.

    RSI und Stochastic messen Überverkauft/Überkauft-Zustände.
    Williams %R ist ein weiterer Überkauft-Indikator.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Momentum-Spalten.
    """
    result = df.copy()

    # --- RSI (14) ---
    # >70 = überkauft (Verkaufssignal), <30 = überverkauft (Kaufsignal)
    result["rsi_14"] = ind_rsi(result["close"], length=14)

    # RSI zentriert um 50 (für lineares Modell besser)
    result["rsi_centered"] = result["rsi_14"] - 50

    # --- Stochastic Oscillator (14, 3) ---
    stoch = ind_stoch(
        result["high"], result["low"], result["close"], k=14, d=3
    )
    result["stoch_k"] = stoch["stoch_k"]
    result["stoch_d"] = stoch["stoch_d"]
    result["stoch_cross"] = np.sign(
        result["stoch_k"] - result["stoch_d"]
    ).fillna(0)

    # --- Williams %R (14) ---
    result["williams_r"] = ind_williams_r(
        result["high"], result["low"], result["close"], length=14
    )

    # --- Rate of Change (10 Perioden) ---
    result["roc_10"] = ind_roc(result["close"], length=10)

    logger.info(
        "Momentum-Features: RSI(14), Stochastic(14,3), "
        "Williams %%%%R(14), ROC(10) ✓"
    )
    return result


# ============================================================
# 4. Volatilitäts-Features
# ============================================================


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Volatilitäts-Indikatoren.

    ATR misst die absolute Volatilität in Pips.
    Bollinger Bands zeigen relative Volatilität und Preis-Position.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Volatilitäts-Spalten.
    """
    result = df.copy()

    # --- ATR (14) – Average True Range ---
    result["atr_14"] = ind_atr(
        result["high"], result["low"], result["close"], length=14
    )

    # ATR normalisiert (% des Preises, für alle Instrumente vergleichbar)
    result["atr_pct"] = result["atr_14"] / result["close"]

    # --- Bollinger Bands (20, 2 Standardabweichungen) ---
    bbands = ind_bbands(result["close"], length=20, std=2.0)
    result["bb_upper"] = bbands["bb_upper"]
    result["bb_mid"] = bbands["bb_mid"]
    result["bb_lower"] = bbands["bb_lower"]

    # BB Width: Breite der Bänder (hoch = volatile Phase)
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result[
        "bb_mid"
    ]

    # BB %B: Wo liegt der Preis? 0 = am unteren Band, 1 = am oberen Band
    band_range = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
    result["bb_pct"] = (result["close"] - result["bb_lower"]) / band_range

    # --- Historische Volatilität (Rolling 20 Perioden) ---
    # Annualisierte Volatilität der Log-Returns (252 Tage * 24 Stunden)
    log_returns = np.log(result["close"] / result["close"].shift(1))
    result["hist_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252 * 24)

    logger.info(
        "Volatilitäts-Features: ATR(14), Bollinger Bands(20,2), Hist.Vol(20) ✓"
    )
    return result


# ============================================================
# 5. Volumen-Features
# ============================================================


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Volumen-basierte Indikatoren.

    Hinweis: MT5 liefert Tick-Volumen, kein echtes Forex-Volumen.
    Tick-Volumen korreliert jedoch gut mit der echten Handelsaktivität.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Volumen-Spalten.
    """
    result = df.copy()

    # --- OBV: On-Balance Volume ---
    result["obv"] = ind_obv(result["close"], result["volume"])

    # OBV normalisiert (Z-Score, Rolling 50) für stationären Wert
    obv_mean = result["obv"].rolling(50).mean()
    obv_std = result["obv"].rolling(50).std().replace(0, np.nan)
    result["obv_zscore"] = (result["obv"] - obv_mean) / obv_std

    # --- Volumen Rate of Change (14 Perioden) ---
    result["volume_roc"] = ind_roc(result["volume"], length=14)

    # --- Volumen-Ratio: aktuelles Volumen vs. 20-Perioden-Durchschnitt ---
    # >1 = überdurchschnittliche Aktivität (oft bei Ausbrüchen)
    vol_sma_20 = result["volume"].rolling(20).mean().replace(0, np.nan)
    result["volume_ratio"] = result["volume"] / vol_sma_20

    logger.info(
        "Volumen-Features: OBV, OBV Z-Score, Volume ROC(14), Volume Ratio ✓"
    )
    return result


# ============================================================
# 6. Kerzenmuster-Features
# ============================================================


def kerzenmuster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Features basierend auf der Kerzenstruktur.

    Alle Werte werden durch den ATR normalisiert, damit sie
    für verschiedene Volatilitätsphasen vergleichbar sind.

    WICHTIG: volatility_features() muss vorher aufgerufen werden,
    da atr_14 bereits im DataFrame vorhanden sein muss.

    Args:
        df: OHLCV DataFrame (muss atr_14 bereits enthalten!)

    Returns:
        DataFrame mit Kerzenstruktur-Spalten.

    Raises:
        ValueError: Wenn atr_14 nicht im DataFrame vorhanden ist.
    """
    # Bug #2 Fix: Explizite Abhängigkeitsprüfung
    if "atr_14" not in df.columns:
        raise ValueError(
            "kerzenmuster_features() benötigt 'atr_14'. "
            "Bitte zuerst volatility_features() aufrufen."
        )

    result = df.copy()
    atr = result["atr_14"].replace(0, np.nan)

    # --- Log-Returns ---
    result["return_1h"] = np.log(result["close"] / result["close"].shift(1))
    result["return_4h"] = np.log(result["close"] / result["close"].shift(4))
    result["return_24h"] = np.log(result["close"] / result["close"].shift(24))

    # --- Kerzenstruktur (normalisiert durch ATR) ---
    body_top = result[["close", "open"]].max(axis=1)
    body_bot = result[["close", "open"]].min(axis=1)

    result["candle_body"] = (body_top - body_bot) / atr
    result["upper_wick"] = (result["high"] - body_top) / atr
    result["lower_wick"] = (body_bot - result["low"]) / atr

    # Kerzenrichtung: 1 = bullish (Kurs gestiegen), -1 = bearish
    result["candle_dir"] = np.sign(result["close"] - result["open"]).fillna(0)

    # High-Low-Range (normalisiert durch Close-Preis)
    result["hl_range"] = (result["high"] - result["low"]) / result["close"]

    logger.info(
        "Kerzenmuster-Features: Returns(1h,4h,24h), Körper, Wicks, HL-Range ✓"
    )
    return result


# ============================================================
# 7. Multi-Timeframe-Features (H4 + D1)
# ============================================================


def multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Multi-Timeframe Trend-Features (H4 und D1).

    LOOK-AHEAD-BIAS PRÄVENTION:
    H4/D1-Kerzen werden mit .shift(1) verzögert, bevor sie auf H1
    forward-gefüllt werden. Eine H4-Kerze die um 08:00 schließt,
    wird erst ab der nächsten H1-Kerze (09:00) als Feature genutzt.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Multi-Timeframe-Spalten.
    """
    result = df.copy()
    close = df["close"]

    # --- H4 (4-Stunden) Trend ---
    close_h4 = close.resample("4h").last().dropna()
    sma_h4_20 = ind_sma(close_h4, 20)
    sma_h4_50 = ind_sma(close_h4, 50)

    # 1 = SMA20 > SMA50 (H4 bullish), -1 = bearish
    trend_h4 = np.sign(sma_h4_20 - sma_h4_50).fillna(0)
    # .shift(1): verhindert Look-Ahead
    trend_h4_delayed = trend_h4.shift(1)
    result["trend_h4"] = trend_h4_delayed.reindex(result.index, method="ffill")

    # H4 RSI als zusätzliches Momentum-Feature
    rsi_h4 = ind_rsi(close_h4, length=14)
    rsi_h4_delayed = rsi_h4.shift(1)
    result["rsi_h4"] = rsi_h4_delayed.reindex(result.index, method="ffill")

    # --- D1 (Täglich) Trend ---
    # Bug #6 Fix: "1d" statt "1D" (Deprecation-Warnung in pandas 3.x vermeiden)
    close_d1 = close.resample("1d").last().dropna()
    sma_d1_20 = ind_sma(close_d1, 20)
    sma_d1_50 = ind_sma(close_d1, 50)

    trend_d1 = np.sign(sma_d1_20 - sma_d1_50).fillna(0)
    trend_d1_delayed = trend_d1.shift(1)
    result["trend_d1"] = trend_d1_delayed.reindex(result.index, method="ffill")

    logger.info("Multi-Timeframe-Features: H4-Trend, H4-RSI, D1-Trend ✓")
    return result


# ============================================================
# 8. Zeitbasierte Features
# ============================================================


def zeitbasierte_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet zeitbasierte Features (Handelssitzungen, Wochentag).

    Forex-Märkte zeigen starke Intraday-Muster:
    - London-Session (08–17 UTC): Hohe Volatilität
    - New York-Session (13–22 UTC): Ebenfalls hohe Aktivität
    - London/NY-Overlap (13–17 UTC): Maximale Liquidität
    - Asien-Session (00–09 UTC): Geringe Volatilität

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Zeit-Spalten.
    """
    result = df.copy()

    # Stunde (0–23 UTC) und Wochentag (0=Montag, 4=Freitag)
    if isinstance(result.index, pd.DatetimeIndex):
        result["hour"] = result.index.hour
        result["day_of_week"] = result.index.dayofweek
    else:
        result["hour"] = pd.to_datetime(result.index).hour
        result["day_of_week"] = pd.to_datetime(result.index).dayofweek

    # Handelssitzungen (Binary: 1 = aktiv, 0 = inaktiv)
    h = result["hour"]
    result["session_london"] = ((h >= 8) & (h < 17)).astype(int)
    result["session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    result["session_asia"] = ((h >= 0) & (h < 9)).astype(int)
    result["session_overlap"] = ((h >= 13) & (h < 17)).astype(int)

    logger.info("Zeitbasierte Features: Stunde, Wochentag, 4 Sitzungen ✓")
    return result


# ============================================================
# 9. NaN-Bereinigung
# ============================================================


def nan_bereinigung(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Zeilen mit NaN-Werten (entstehen durch Warm-Up-Perioden).

    SMA(200) braucht 200 Kerzen → erste 199 Zeilen haben NaN.
    Diese werden entfernt, damit das Modell saubere Daten erhält.

    Args:
        df: DataFrame mit möglichen NaN-Werten

    Returns:
        DataFrame ohne NaN-Werte.
    """
    vor = len(df)
    df_clean = df.dropna()
    entfernt = vor - len(df_clean)
    logger.info(
        "NaN-Bereinigung: %s Zeilen entfernt, %s Kerzen übrig",
        entfernt,
        f"{len(df_clean):,}",
    )
    return df_clean


# ============================================================
# 10. Hauptfunktion
# ============================================================


def features_berechnen(symbol: str) -> bool:
    """
    Berechnet alle Features für ein Symbol und speichert das Ergebnis.

    Args:
        symbol: Währungspaar, z.B. "EURUSD"

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    eingabe = DATA_DIR / f"{symbol}_H1.csv"
    if not eingabe.exists():
        logger.error("%s: Datei nicht gefunden: %s", symbol, eingabe)
        return False

    try:
        # Daten laden und alle Feature-Gruppen berechnen.
        # WICHTIG: volatility_features() MUSS vor
        # kerzenmuster_features() stehen,
        # da kerzenmuster_features() atr_14 aus volatility_features() benötigt.
        df = daten_laden(eingabe)
        df = trend_features(df)
        df = momentum_features(df)
        df = volatility_features(df)  # ← erstellt atr_14
        df = volume_features(df)
        df = kerzenmuster_features(df)  # ← verwendet atr_14
        df = multitimeframe_features(df)
        df = zeitbasierte_features(df)
        df = nan_bereinigung(df)

    except ValueError as e:
        logger.error("%s: Fehler in Feature-Berechnung: %s", symbol, e)
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("%s: Unerwarteter Fehler: %s", symbol, e)
        return False

    # Ergebnis speichern
    ausgabe = DATA_DIR / f"{symbol}_H1_features.csv"
    df.to_csv(ausgabe)
    groesse_mb = ausgabe.stat().st_size / (1024 * 1024)
    logger.info(
        "%s: gespeichert → %s (%.1f MB)", symbol, ausgabe.name, groesse_mb
    )
    return True


def main() -> None:
    """Hauptablauf: Features für alle 7 Forex-Symbole berechnen."""
    symbole = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
    ]

    logger.info("=" * 60)
    logger.info("Feature Engineering – %s Symbole", len(symbole))
    logger.info("Gerät: Linux-Server")
    logger.info("=" * 60)

    ergebnisse = []
    for symbol in symbole:
        logger.info("\n%s", "─" * 40)
        logger.info("Verarbeite: %s", symbol)
        logger.info("%s", "─" * 40)
        ok = features_berechnen(symbol)
        ergebnisse.append((symbol, "OK" if ok else "FEHLER"))

    # Abschluss-Zusammenfassung
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING – ABGESCHLOSSEN")
    print("=" * 60)
    for symbol, status in ergebnisse:
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark} {symbol}_H1_features.csv")
    erfolge = sum(1 for _, s in ergebnisse if s == "OK")
    print(f"\n{erfolge}/{len(symbole)} Symbole erfolgreich verarbeitet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
