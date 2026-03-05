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

# Optional: HMM-Regime-Detection (falls installiert)
try:
    from hmmlearn.hmm import GaussianHMM

    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Zeitzonen-neutrale Timeframe-Konfiguration (für H1/M60/M30/M15/M5 Migration)
TIMEFRAME_CONFIG = {
    "H1": {"bars_per_hour": 1},
    "M60": {"bars_per_hour": 1},
    "M30": {"bars_per_hour": 2},
    "M15": {"bars_per_hour": 4},
    "M5": {"bars_per_hour": 12},
}


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
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
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
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

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
    atr = true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
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
    result["price_sma20_ratio"] = (result["close"] - result["sma_20"]) / result[
        "sma_20"
    ]
    result["price_sma50_ratio"] = (result["close"] - result["sma_50"]) / result[
        "sma_50"
    ]
    result["price_sma200_ratio"] = (result["close"] - result["sma_200"]) / result[
        "sma_200"
    ]

    # SMA-Crossover: 1 = SMA20 über SMA50 (bullish), -1 = bearish
    result["sma_20_50_cross"] = np.sign(result["sma_20"] - result["sma_50"]).fillna(0)
    result["sma_50_200_cross"] = np.sign(result["sma_50"] - result["sma_200"]).fillna(0)

    # --- Exponential Moving Averages ---
    result["ema_12"] = ind_ema(result["close"], 12)
    result["ema_26"] = ind_ema(result["close"], 26)

    # EMA Crossover: 1 = EMA12 über EMA26 (bullish Signal)
    result["ema_cross"] = np.sign(result["ema_12"] - result["ema_26"]).fillna(0)

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
    stoch = ind_stoch(result["high"], result["low"], result["close"], k=14, d=3)
    result["stoch_k"] = stoch["stoch_k"]
    result["stoch_d"] = stoch["stoch_d"]
    result["stoch_cross"] = np.sign(result["stoch_k"] - result["stoch_d"]).fillna(0)

    # --- Williams %R (14) ---
    result["williams_r"] = ind_williams_r(
        result["high"], result["low"], result["close"], length=14
    )

    # --- Rate of Change (10 Perioden) ---
    result["roc_10"] = ind_roc(result["close"], length=10)

    logger.info(
        "Momentum-Features: RSI(14), Stochastic(14,3), " "Williams %%%%R(14), ROC(10) ✓"
    )
    return result


# ============================================================
# 4. Volatilitäts-Features
# ============================================================


def volatility_features(df: pd.DataFrame, timeframe: str = "H1") -> pd.DataFrame:
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
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]

    # BB %B: Wo liegt der Preis? 0 = am unteren Band, 1 = am oberen Band
    band_range = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
    result["bb_pct"] = (result["close"] - result["bb_lower"]) / band_range

    # --- Historische Volatilität (Rolling 20 Perioden) ---
    # Annualisierte Volatilität der Log-Returns.
    # H1/M60: 252*24 Bars/Jahr, M30: 252*48 Bars/Jahr, M15: 252*96 Bars/Jahr.
    log_returns = np.log(result["close"] / result["close"].shift(1))
    bars_per_hour = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
        "bars_per_hour"
    ]
    bars_per_day = 24 * bars_per_hour
    result["hist_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252 * bars_per_day)

    logger.info("Volatilitäts-Features: ATR(14), Bollinger Bands(20,2), Hist.Vol(20) ✓")
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

    logger.info("Volumen-Features: OBV, OBV Z-Score, Volume ROC(14), Volume Ratio ✓")
    return result


# ============================================================
# 6. Kerzenmuster-Features
# ============================================================


def kerzenmuster_features(df: pd.DataFrame, timeframe: str = "H1") -> pd.DataFrame:
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

    # --- Log-Returns mit zeitäquivalenten Fenstern ---
    # H1/M60: 1/4/24 Bars, M30: 2/8/48 Bars, M15: 4/16/96 Bars.
    bars_per_hour = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
        "bars_per_hour"
    ]
    shift_1h = 1 * bars_per_hour
    shift_4h = 4 * bars_per_hour
    shift_24h = 24 * bars_per_hour
    result["return_1h"] = np.log(result["close"] / result["close"].shift(shift_1h))
    result["return_4h"] = np.log(result["close"] / result["close"].shift(shift_4h))
    result["return_24h"] = np.log(result["close"] / result["close"].shift(shift_24h))

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

    logger.info("Kerzenmuster-Features: Returns(1h,4h,24h), Körper, Wicks, HL-Range ✓")
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
# 8b. Kill-Zone Features (präzise Session-Fenster)
# ============================================================


def killzone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet präzisere Kill-Zone-Features für Intraday-Entries.

    Im Unterschied zu breiten Session-Flags markieren Kill Zones
    die typischen Hochaktivitäts-Fenster:
      - London Open: 07:00–09:00 UTC
      - New York Open: 13:00–15:00 UTC
      - Asia Open: 00:00–02:00 UTC

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Kill-Zone-Spalten.
    """
    result = df.copy()

    if isinstance(result.index, pd.DatetimeIndex):
        stunde = result.index.hour
    else:
        stunde = pd.to_datetime(result.index).hour

    # Präzise Handelsfenster für Entry-Timing
    result["killzone_london_open"] = ((stunde >= 7) & (stunde < 9)).astype(int)
    result["killzone_ny_open"] = ((stunde >= 13) & (stunde < 15)).astype(int)
    result["killzone_asia_open"] = ((stunde >= 0) & (stunde < 2)).astype(int)

    logger.info("Kill-Zone-Features: London/NY/Asia Open ✓")
    return result


# ============================================================
# 8c. Key-Level Features (PDH/PDL/PWH/PWL)
# ============================================================


def key_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Key-Level-Distanzen (Previous Day/Week High/Low).

    LOOK-AHEAD-BIAS PRÄVENTION:
    - Tages-/Wochenlevels werden immer mit .shift(1) verzögert.
    - Eine Kerze sieht also nie High/Low derselben laufenden Periode.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Key-Level-Spalten.
    """
    result = df.copy()

    # Previous Day Levels
    day_high = result["high"].resample("1d").max().shift(1)
    day_low = result["low"].resample("1d").min().shift(1)
    result["pdh"] = day_high.reindex(result.index, method="ffill")
    result["pdl"] = day_low.reindex(result.index, method="ffill")

    # Previous Week Levels (Woche startet Montag)
    week_high = result["high"].resample("W-MON").max().shift(1)
    week_low = result["low"].resample("W-MON").min().shift(1)
    result["pwh"] = week_high.reindex(result.index, method="ffill")
    result["pwl"] = week_low.reindex(result.index, method="ffill")

    close_safe = result["close"].replace(0, np.nan)
    # Distanz-Features (normalisiert) – für Modell besser als absolute Preisniveaus
    result["dist_pdh_pct"] = (result["close"] - result["pdh"]) / close_safe
    result["dist_pdl_pct"] = (result["close"] - result["pdl"]) / close_safe
    result["dist_pwh_pct"] = (result["close"] - result["pwh"]) / close_safe
    result["dist_pwl_pct"] = (result["close"] - result["pwl"]) / close_safe

    # Nähe zu einem Key-Level (0.15% Toleranzband)
    level_tol = 0.0015
    result["near_key_level"] = (
        (result["dist_pdh_pct"].abs() <= level_tol)
        | (result["dist_pdl_pct"].abs() <= level_tol)
        | (result["dist_pwh_pct"].abs() <= level_tol)
        | (result["dist_pwl_pct"].abs() <= level_tol)
    ).astype(int)

    logger.info("Key-Level-Features: PDH/PDL/PWH/PWL + Distanz ✓")
    return result


# ============================================================
# 8d. FVG-Features (Fair Value Gaps)
# ============================================================


def fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet einfache Fair-Value-Gap (FVG) Features mit 3-Kerzen-Logik.

    Bullish FVG: low[t] > high[t-2]
    Bearish FVG: high[t] < low[t-2]

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit FVG-Spalten.
    """
    result = df.copy()

    high_shift2 = result["high"].shift(2)
    low_shift2 = result["low"].shift(2)

    bull_fvg = result["low"] > high_shift2
    bear_fvg = result["high"] < low_shift2

    result["fvg_bullish"] = bull_fvg.astype(int)
    result["fvg_bearish"] = bear_fvg.astype(int)

    # Gap-Größe in % des Close (signiert: bullish positiv, bearish negativ)
    close_safe = result["close"].replace(0, np.nan)
    bull_gap = (result["low"] - high_shift2) / close_safe
    bear_gap = (low_shift2 - result["high"]) / close_safe
    result["fvg_gap_pct"] = np.where(
        bull_fvg,
        bull_gap,
        np.where(bear_fvg, -bear_gap, 0.0),
    )

    logger.info("FVG-Features: bullish/bearish + gap_pct ✓")
    return result


# ============================================================
# 8e. MSS/BOS Features (Marktstruktur)
# ============================================================


def mss_bos_features(df: pd.DataFrame, pivot_bars: int = 20) -> pd.DataFrame:
    """
    Berechnet Marktstruktur-Features (BOS und MSS) ohne Future-Leak.

    Methodik (leakage-sicher):
    - Struktur-Level basieren ausschließlich auf Vergangenheitsdaten via shift(1).
    - BOS bullish: close > rolling_max(high, pivot_bars) der Vergangenheit
    - BOS bearish: close < rolling_min(low, pivot_bars) der Vergangenheit
    - MSS: erster BOS in Gegenrichtung zum letzten dominanten BOS

    Args:
        df: OHLCV DataFrame
        pivot_bars: Rückblickfenster für Struktur-Level

    Returns:
        DataFrame mit MSS/BOS-Spalten.
    """
    result = df.copy()

    prev_swing_high = result["high"].shift(1).rolling(pivot_bars).max()
    prev_swing_low = result["low"].shift(1).rolling(pivot_bars).min()

    bos_bull = (result["close"] > prev_swing_high).astype(int)
    bos_bear = (result["close"] < prev_swing_low).astype(int)

    result["bos_bull"] = bos_bull
    result["bos_bear"] = bos_bear

    # Struktur-Bias: 1=bullish, -1=bearish, 0=neutral
    structure_bias = np.where(bos_bull == 1, 1, np.where(bos_bear == 1, -1, 0))
    result["structure_bias"] = (
        pd.Series(structure_bias, index=result.index).ffill().fillna(0)
    )

    prev_bias = result["structure_bias"].shift(1).fillna(0)
    result["mss_bull"] = ((bos_bull == 1) & (prev_bias < 0)).astype(int)
    result["mss_bear"] = ((bos_bear == 1) & (prev_bias > 0)).astype(int)

    logger.info("MSS/BOS-Features: BOS bull/bear, MSS bull/bear, structure_bias ✓")
    return result


# ============================================================
# 8f. HMM-Regime-Feature (probabilistisch)
# ============================================================


def hmm_regime_feature(
    df: pd.DataFrame,
    n_states: int = 4,
    min_train_bars: int = 400,
    refit_interval: int = 2000,
) -> pd.DataFrame:
    """
    Berechnet ein HMM-basiertes Regime-Feature mit Walk-Forward-Fit.

    Ablauf:
    - HMM wird auf einem wachsenden historischen Fenster trainiert.
    - Refit nur alle `refit_interval` Bars (Performance).
    - Für jede Bar wird der Regime-State ohne Nutzung zukünftiger Bars geschätzt.

    Falls `hmmlearn` nicht installiert ist, wird robust auf das bestehende
    regelbasierte `market_regime` zurückgefallen.

    Args:
        df: DataFrame mit mindestens close und atr_pct
        n_states: Anzahl HMM-Zustände
        min_train_bars: Mindesthistorie vor erstem Fit
        refit_interval: Refit-Frequenz

    Returns:
        DataFrame mit zusätzlicher Spalte `market_regime_hmm`.
    """
    result = df.copy()

    if not HAS_HMMLEARN:
        logger.warning(
            "hmmlearn nicht installiert – market_regime_hmm nutzt Fallback market_regime"
        )
        result["market_regime_hmm"] = result.get("market_regime", 0)
        return result

    log_ret = np.log(result["close"] / result["close"].shift(1)).fillna(0.0)
    atr_pct = result["atr_pct"].ffill().fillna(0.0)
    x = np.column_stack([log_ret.values, atr_pct.values])

    states = np.full(len(result), np.nan)
    hmm_model: GaussianHMM | None = None

    train_window = 5000

    for i in range(min_train_bars, len(result)):
        if hmm_model is None or (i - min_train_bars) % refit_interval == 0:
            try:
                start_idx = max(0, i - train_window)
                hmm_model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=150,
                    random_state=42,
                )
                hmm_model.fit(x[start_idx:i])
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("HMM-Fit fehlgeschlagen bei i=%s: %s", i, e)
                hmm_model = None

        if hmm_model is not None:
            try:
                states[i] = int(hmm_model.predict(x[i : i + 1])[0])
            except Exception:  # pylint: disable=broad-exception-caught
                states[i] = np.nan

    state_series = pd.Series(states, index=result.index).ffill().fillna(0).astype(int)

    # Zustände semantisch auf 4 Regime mappen (0=Seitwärts,1=Auf,2=Ab,3=HighVola)
    # Mapping basiert auf in-sample Statistik der bereits berechneten States.
    state_stats = (
        pd.DataFrame(
            {
                "state": state_series,
                "ret": log_ret,
                "vol": atr_pct,
            }
        )
        .groupby("state")
        .agg(ret_mean=("ret", "mean"), vol_mean=("vol", "mean"))
    )

    if len(state_stats) >= 2:
        high_vol_state = int(state_stats["vol_mean"].idxmax())
        ret_sorted = state_stats.drop(
            index=high_vol_state, errors="ignore"
        ).sort_values("ret_mean")
        bear_state = int(ret_sorted.index[0]) if len(ret_sorted) > 0 else high_vol_state
        bull_state = (
            int(ret_sorted.index[-1]) if len(ret_sorted) > 0 else high_vol_state
        )
        regime_map = {s: 0 for s in state_stats.index}
        regime_map[high_vol_state] = 3
        regime_map[bull_state] = 1
        regime_map[bear_state] = 2
        result["market_regime_hmm"] = state_series.map(regime_map).fillna(0).astype(int)
    else:
        result["market_regime_hmm"] = result.get("market_regime", 0)

    logger.info("HMM-Regime-Feature: market_regime_hmm ✓")
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


def features_berechnen(symbol: str, timeframe: str = "H1") -> bool:
    """
    Berechnet alle Features für ein Symbol und speichert das Ergebnis.

    Args:
        symbol: Währungspaar, z.B. "EURUSD"

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    eingabe = DATA_DIR / f"{symbol}_{timeframe}.csv"
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
        df = volatility_features(df, timeframe=timeframe)  # ← erstellt atr_14
        df = volume_features(df)
        df = kerzenmuster_features(df, timeframe=timeframe)  # ← verwendet atr_14
        df = multitimeframe_features(df)
        df = zeitbasierte_features(df)
        df = killzone_features(df)
        df = key_level_features(df)
        df = fvg_features(df)
        df = mss_bos_features(df)

        # Bestehendes regelbasiertes Regime bleibt erhalten; HMM-Regime kommt zusätzlich.
        # In Echtzeit ist HMM optional (Fallback aktiv wenn hmmlearn fehlt).
        # Das Training kann dann explizit beide Regime-Features nutzen.
        df = hmm_regime_feature(df)
        df = nan_bereinigung(df)

    except ValueError as e:
        logger.error("%s: Fehler in Feature-Berechnung: %s", symbol, e)
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("%s: Unerwarteter Fehler: %s", symbol, e)
        return False

    # Ergebnis speichern
    ausgabe = DATA_DIR / f"{symbol}_{timeframe}_features.csv"
    df.to_csv(ausgabe)
    groesse_mb = ausgabe.stat().st_size / (1024 * 1024)
    logger.info("%s: gespeichert → %s (%.1f MB)", symbol, ausgabe.name, groesse_mb)
    return True


def main() -> None:
    """Hauptablauf: Features für ein oder mehrere Symbole berechnen."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature Engineering für H1/M60/M30/M15/M5-Daten"
    )
    parser.add_argument(
        "--symbol",
        default="alle",
        help=("Handelssymbol oder 'alle' (Standard). " "Beispiel: --symbol USDCAD"),
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "M60", "M30", "M15", "M5"],
        help="Zeitrahmen der Eingabe-CSV (Standard: H1).",
    )
    args = parser.parse_args()

    symbole = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "USDCHF",
        "NZDUSD",
    ]

    if args.symbol.lower() == "alle":
        ziel_symbole = symbole
    elif args.symbol.upper() in symbole:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: {args.symbol}")
        print(f"Verfügbar: {', '.join(symbole)} oder 'alle'")
        return

    logger.info("=" * 60)
    logger.info("Feature Engineering – %s Symbole", len(ziel_symbole))
    logger.info("Zeitrahmen: %s", args.timeframe)
    logger.info("Gerät: Linux-Server")
    logger.info("=" * 60)

    ergebnisse = []
    for symbol in ziel_symbole:
        logger.info("\n%s", "─" * 40)
        logger.info("Verarbeite: %s", symbol)
        logger.info("%s", "─" * 40)
        ok = features_berechnen(symbol, timeframe=args.timeframe)
        ergebnisse.append((symbol, "OK" if ok else "FEHLER"))

    # Abschluss-Zusammenfassung
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING – ABGESCHLOSSEN")
    print("=" * 60)
    for symbol, status in ergebnisse:
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark} {symbol}_{args.timeframe}_features.csv")
    erfolge = sum(1 for _, s in ergebnisse if s == "OK")
    print(f"\n{erfolge}/{len(ziel_symbole)} Symbole erfolgreich verarbeitet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
