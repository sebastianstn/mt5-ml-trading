"""
live_trader.py – Automatisches Live-Trading mit MetaTrader 5 und LightGBM

Läuft auf: Windows 11 Laptop (MetaTrader5-Bibliothek NUR auf Windows!)

PAPER-TRADING-MODUS (Standard):
    PAPER_TRADING = True  → Signale werden NUR geloggt, KEIN echtes Geld!
    PAPER_TRADING = False → Echte Orders – erst nach 2 Wochen Testlauf!

Ablauf pro H1-Kerze:
    1. Neue H1-Daten von MT5 abrufen (letzten 500 Barren)
    2. Alle 45 Features berechnen (identisch mit feature_engineering.py)
    3. Externe Features holen (Fear & Greed Index, BTC Funding Rate)
    4. Marktregime erkennen (identisch mit regime_detection.py)
    5. LightGBM-Vorhersage + Wahrscheinlichkeits-Filter (Schwelle)
       - Optional: Shadow-Mode mit Two-Stage (HTF H1 + LTF M5) für USDCAD/USDJPY
    6. Regime-Filter anwenden (z.B. nur im Aufwärtstrend handeln)
    7. Order senden (Paper: nur loggen / Live: echte MT5-Order)

Shadow-Mode (Two-Stage):
    --two_stage_enable 1 aktiviert das Two-Stage-System (USDCAD + USDJPY):
        - HTF-Bias-Modell (H1): Bestimmt Marktrichtung (Short/Neutral/Long)
        - LTF-Entry-Modell (M5): Generiert Entry-Signal basierend auf HTF-Bias
        - Beide Signale (Single-Stage vs. Two-Stage) werden geloggt für Vergleich
        - Hard Fallback zu Single-Stage bei jedem Fehler

Modell-Übertragung vom Linux-Server auf den Windows Laptop:
    Auf dem Linux-Server ausführen:
        # Single-Stage (H1):
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_usdcad_v1.pkl USER@LAPTOP:./models/
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_usdjpy_v1.pkl USER@LAPTOP:./models/

        # Two-Stage (H1 + M5, für USDJPY):
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_htf_bias_usdjpy_H1_v4.pkl USER@LAPTOP:./models/
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_ltf_entry_usdjpy_M5_v4.pkl USER@LAPTOP:./models/

Verwendung (Windows, venv aktiviert):
    # Single-Stage (H1):
    python live_trader.py --symbol USDCAD --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1

    # Two-Stage Shadow-Mode (USDCAD / USDJPY):
    python live_trader.py --symbol USDJPY --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1 \
        --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4

    # Hilfe:
    python live_trader.py --help

Voraussetzungen:
    pip install MetaTrader5 pandas numpy pandas_ta joblib requests python-dotenv
"""

# pylint: disable=too-many-lines,logging-fstring-interpolation

# Standard-Bibliotheken
import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Datenverarbeitung
import numpy as np
import pandas as pd

# Modell laden
import joblib

# HTTP-Anfragen für externe APIs (Fear & Greed, BTC Funding Rate)
import requests

# Optional: HMM-Regime-Detection (falls installiert)
try:
    from hmmlearn.hmm import GaussianHMM

    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# MetaTrader5 – NUR auf Windows verfügbar!
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_VERFUEGBAR = True
except ImportError:
    MT5_VERFUEGBAR = False
    mt5 = None  # type: ignore

# pandas_ta – für ADX-Berechnung (identisch mit regime_detection.py)
try:
    import pandas_ta  # noqa: F401  # pylint: disable=unused-import

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# Two-Stage Signal (Shadow-Mode für Option 1)
try:
    from two_stage_signal import (
        modelle_laden as two_stage_modelle_laden,
        zwei_stufen_signal,
    )

    TWO_STAGE_VERFUEGBAR = True
except ImportError:
    TWO_STAGE_VERFUEGBAR = False

# ============================================================
# Konfiguration und Pfade
# ============================================================

# Pfade (relativ zum Skript-Verzeichnis)
BASE_DIR = Path(__file__).parent.parent  # Ordner mt5_ml_trading/
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging: Terminal + Datei (UTF-8 für deutsche Sonderzeichen)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "live_trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Trading-Parameter (müssen mit backtest.py übereinstimmen!)
# ============================================================

# Risikomanagement
TP_PCT = 0.003  # Take-Profit: 0.3% (identisch mit Labeling)
SL_PCT = 0.003  # Stop-Loss:   0.3% (Fallback, wenn ATR-SL deaktiviert)
LOT = 0.01  # Minimale Lot-Größe (0.01 = Micro-Lot, ~1€/Pip)
MAX_OFFENE_TRADES = 1  # Maximal 1 offene Position pro Symbol
MAGIC_NUMBER = 20260101  # Eindeutige Kennung für ML-Trades in MT5

# ATR-basiertes Stop-Loss (dynamisch, passt sich an Volatilität an)
# Backtest-Ergebnis: ATR-SL 1.5× verbessert Sharpe drastisch (+2.1 USDCAD, +1.3 USDJPY)
ATR_SL_ENABLED = True  # ATR-SL statt festem SL verwenden
ATR_SL_FAKTOR = 1.5  # SL = ATR_14 × 1.5 (optimaler Faktor aus Backtest)

# Kill-Switch – Harter Stopp bei zu hohem Drawdown (Review-Punkt 8)
KILL_SWITCH_DD_DEFAULT = 0.15  # Harter Stopp bei 15% Drawdown (Standard)

# Heartbeat-Logging: schreibt auch ohne Trade ein CSV-Update pro neuer Kerze.
# Dadurch bleibt das MT5-Dashboard frisch und springt nicht auf "STALE",
# wenn nur wegen Regime-Filter keine Trades entstehen.
HEARTBEAT_LOG_DEFAULT = True

# Feature-Berechnung: Mindest-Barren für Warm-Up
N_BARREN = 500  # SMA200 braucht 200, MTF braucht mehr → 500 als Buffer

# Zeitrahmen-Konfiguration (für Migration H1 → M30 → M15 → M5)
TIMEFRAME_CONFIG = {
    "H1": {
        "mt5_name": "TIMEFRAME_H1",
        "bars_per_hour": 1,
        "minutes_per_bar": 60,
    },
    "M30": {
        "mt5_name": "TIMEFRAME_M30",
        "bars_per_hour": 2,
        "minutes_per_bar": 30,
    },
    "M15": {
        "mt5_name": "TIMEFRAME_M15",
        "bars_per_hour": 4,
        "minutes_per_bar": 15,
    },
    "M5": {
        "mt5_name": "TIMEFRAME_M5",
        "bars_per_hour": 12,
        "minutes_per_bar": 5,
    },
}

# Verfügbare Symbole im Projekt (Forschung + Betrieb)
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# AKTIVE PRODUKTIONS-SYMBOLE (Policy): Nur diese 2 Paare werden operativ gehandelt.
# Alle anderen Paare bleiben Research-only, bis die KPI-Gates dauerhaft erfüllt sind.
AKTIVE_SYMBOLE = ["USDCAD", "USDJPY"]

# Regime-Namen (für Logging)
REGIME_NAMEN = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

# Spalten, die beim Modell-Input AUSGESCHLOSSEN werden
# (gleich wie in train_model.py und backtest.py!)
AUSSCHLUSS_SPALTEN = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "atr_14",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "obv",
    "pdh",
    "pdl",
    "pwh",
    "pwl",
    "label",
}

# Exakte Feature-Reihenfolge (identisch mit Trainings-CSV)
# Das Modell erwartet genau diese 45 Spalten
FEATURE_SPALTEN = [
    "price_sma20_ratio",
    "price_sma50_ratio",
    "price_sma200_ratio",
    "sma_20_50_cross",
    "sma_50_200_cross",
    "ema_cross",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "rsi_centered",
    "stoch_k",
    "stoch_d",
    "stoch_cross",
    "williams_r",
    "roc_10",
    "atr_pct",
    "bb_width",
    "bb_pct",
    "hist_vol_20",
    "obv_zscore",
    "volume_roc",
    "volume_ratio",
    "return_1h",
    "return_4h",
    "return_24h",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "candle_dir",
    "hl_range",
    "trend_h4",
    "rsi_h4",
    "trend_d1",
    "hour",
    "day_of_week",
    "session_london",
    "session_ny",
    "session_asia",
    "session_overlap",
    "killzone_london_open",
    "killzone_ny_open",
    "killzone_asia_open",
    "dist_pdh_pct",
    "dist_pdl_pct",
    "dist_pwh_pct",
    "dist_pwl_pct",
    "near_key_level",
    "fvg_bullish",
    "fvg_bearish",
    "fvg_gap_pct",
    "bos_bull",
    "bos_bear",
    "mss_bull",
    "mss_bear",
    "structure_bias",
    "adx_14",
    "market_regime",
    "market_regime_hmm",
    "fear_greed_value",
    "fear_greed_class",
    "btc_funding_rate",
]

# ============================================================
# 1. Indikator-Funktionen (identisch mit feature_engineering.py)
# ============================================================


def ind_sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()


def ind_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average (adjust=False = MetaTrader-kompatibel)."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def ind_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD: macd_line, macd_signal, macd_hist."""
    ema_fast = ind_ema(series, fast)
    ema_slow = ind_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal_line,
            "macd_hist": macd_line - macd_signal_line,
        }
    )


def ind_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI – Wilder-Methode (identisch mit MetaTrader)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / length
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ind_stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator %K und %D."""
    low_min = low.rolling(window=k, min_periods=k).min()
    high_max = high.rolling(window=k, min_periods=k).max()
    band = (high_max - low_min).replace(0, np.nan)
    stoch_k = (close - low_min) / band * 100
    return pd.DataFrame(
        {
            "stoch_k": stoch_k,
            "stoch_d": stoch_k.rolling(window=d, min_periods=d).mean(),
        }
    )


def ind_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Williams %R (-100 bis 0)."""
    high_max = high.rolling(window=length, min_periods=length).max()
    low_min = low.rolling(window=length, min_periods=length).min()
    band = (high_max - low_min).replace(0, np.nan)
    return (high_max - close) / band * -100


def ind_roc(series: pd.Series, length: int = 10) -> pd.Series:
    """Rate of Change in Prozent."""
    return series.pct_change(periods=length) * 100


def ind_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """ATR – Average True Range (Wilder-Smoothing)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def ind_bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: bb_upper, bb_mid, bb_lower."""
    bb_mid = series.rolling(window=length, min_periods=length).mean()
    bb_std = series.rolling(window=length, min_periods=length).std()
    return pd.DataFrame(
        {
            "bb_upper": bb_mid + std * bb_std,
            "bb_mid": bb_mid,
            "bb_lower": bb_mid - std * bb_std,
        }
    )


def ind_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def ind_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    ADX – Average Directional Index.

    Verwendet pandas_ta falls verfügbar (identisch mit regime_detection.py),
    sonst manuelle Wilder-Implementierung als Fallback.

    Args:
        df:     DataFrame mit 'high', 'low', 'close'
        length: Periode (Standard: 14)

    Returns:
        ADX-Serie (0–100)
    """
    if HAS_PANDAS_TA:
        # Identisch mit regime_detection.py → gleiche Werte wie beim Training
        adx_df = df.ta.adx(length=length)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
        return adx_df[adx_col]

    # Fallback: Manuelle Wilder-Implementierung
    alpha = 1.0 / length
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    cond_plus = (up_move > down_move) & (up_move > 0)
    cond_minus = (down_move > up_move) & (down_move > 0)
    plus_dm[cond_plus] = up_move[cond_plus]
    minus_dm[cond_minus] = down_move[cond_minus]

    atr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_di = (
        plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_s
    ) * 100
    minus_di = (
        minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_s
    ) * 100
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


# ============================================================
# 2. Feature-Berechnung (identisch mit Trainings-Pipeline)
# ============================================================


def features_berechnen(
    df: pd.DataFrame,
    timeframe: str = "H1",
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    Berechnet alle 45 Model-Features aus OHLCV-Rohdaten.

    Die Feature-Namen und Formeln sind IDENTISCH mit feature_engineering.py
    und regime_detection.py. Das ist kritisch: Das Modell muss dieselben
    Feature-Werte sehen wie beim Training, sonst sind Vorhersagen falsch.

    Args:
        df: OHLCV DataFrame mit Spalten open, high, low, close, volume
            und DatetimeIndex (UTC)

    Returns:
        DataFrame mit allen 45 Features (ohne NaN-Zeilen am Anfang)
    """
    result = df.copy()

    # Bars pro Stunde für zeitäquivalente Fenster bestimmen.
    # Beispiel: return_1h = shift(1) bei H1, aber shift(2) bei M30.
    bars_per_hour = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
        "bars_per_hour"
    ]

    # --- Trend-Features ---
    result["sma_20"] = ind_sma(result["close"], 20)
    result["sma_50"] = ind_sma(result["close"], 50)
    result["sma_200"] = ind_sma(result["close"], 200)

    result["price_sma20_ratio"] = (result["close"] - result["sma_20"]) / result[
        "sma_20"
    ]
    result["price_sma50_ratio"] = (result["close"] - result["sma_50"]) / result[
        "sma_50"
    ]
    result["price_sma200_ratio"] = (result["close"] - result["sma_200"]) / result[
        "sma_200"
    ]
    result["sma_20_50_cross"] = np.sign(result["sma_20"] - result["sma_50"]).fillna(0)
    result["sma_50_200_cross"] = np.sign(result["sma_50"] - result["sma_200"]).fillna(0)

    result["ema_12"] = ind_ema(result["close"], 12)
    result["ema_26"] = ind_ema(result["close"], 26)
    result["ema_cross"] = np.sign(result["ema_12"] - result["ema_26"]).fillna(0)

    macd = ind_macd(result["close"])
    result["macd_line"] = macd["macd_line"]
    result["macd_signal"] = macd["macd_signal"]
    result["macd_hist"] = macd["macd_hist"]

    # --- Momentum-Features ---
    result["rsi_14"] = ind_rsi(result["close"], 14)
    result["rsi_centered"] = result["rsi_14"] - 50

    stoch = ind_stoch(result["high"], result["low"], result["close"])
    result["stoch_k"] = stoch["stoch_k"]
    result["stoch_d"] = stoch["stoch_d"]
    result["stoch_cross"] = np.sign(result["stoch_k"] - result["stoch_d"]).fillna(0)

    result["williams_r"] = ind_williams_r(
        result["high"], result["low"], result["close"]
    )
    result["roc_10"] = ind_roc(result["close"], 10)

    # --- Volatilitäts-Features ---
    result["atr_14"] = ind_atr(result["high"], result["low"], result["close"])
    result["atr_pct"] = result["atr_14"] / result["close"]

    bb = ind_bbands(result["close"])
    result["bb_upper"] = bb["bb_upper"]
    result["bb_mid"] = bb["bb_mid"]
    result["bb_lower"] = bb["bb_lower"]
    band_range = (result["bb_upper"] - result["bb_lower"]).replace(0, np.nan)
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]
    result["bb_pct"] = (result["close"] - result["bb_lower"]) / band_range

    log_ret = np.log(result["close"] / result["close"].shift(1))
    bars_per_day = 24 * bars_per_hour
    result["hist_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252 * bars_per_day)

    # --- Volumen-Features ---
    result["obv"] = ind_obv(result["close"], result["volume"])
    obv_mean = result["obv"].rolling(50).mean()
    obv_std = result["obv"].rolling(50).std().replace(0, np.nan)
    result["obv_zscore"] = (result["obv"] - obv_mean) / obv_std
    result["volume_roc"] = ind_roc(result["volume"], 14)
    vol_sma = result["volume"].rolling(20).mean().replace(0, np.nan)
    result["volume_ratio"] = result["volume"] / vol_sma

    # --- Kerzenmuster-Features ---
    shift_1h = 1 * bars_per_hour
    shift_4h = 4 * bars_per_hour
    shift_24h = 24 * bars_per_hour
    result["return_1h"] = np.log(result["close"] / result["close"].shift(shift_1h))
    result["return_4h"] = np.log(result["close"] / result["close"].shift(shift_4h))
    result["return_24h"] = np.log(result["close"] / result["close"].shift(shift_24h))

    atr_safe = result["atr_14"].replace(0, np.nan)
    body_top = result[["close", "open"]].max(axis=1)
    body_bot = result[["close", "open"]].min(axis=1)
    result["candle_body"] = (body_top - body_bot) / atr_safe
    result["upper_wick"] = (result["high"] - body_top) / atr_safe
    result["lower_wick"] = (body_bot - result["low"]) / atr_safe
    result["candle_dir"] = np.sign(result["close"] - result["open"]).fillna(0)
    result["hl_range"] = (result["high"] - result["low"]) / result["close"]

    # --- Multi-Timeframe-Features (H4 + D1) ---
    # LOOK-AHEAD-BIAS: .shift(1) verhindert zukünftige Information
    close = result["close"]
    close_h4 = close.resample("4h").last().dropna()
    trend_h4 = np.sign(ind_sma(close_h4, 20) - ind_sma(close_h4, 50)).fillna(0)
    result["trend_h4"] = trend_h4.shift(1).reindex(result.index, method="ffill")
    result["rsi_h4"] = (
        ind_rsi(close_h4, 14).shift(1).reindex(result.index, method="ffill")
    )

    close_d1 = (
        close.resample("1D").last().dropna()
    )  # pandas 4.x erwartet Großbuchstabe "D"
    trend_d1 = np.sign(ind_sma(close_d1, 20) - ind_sma(close_d1, 50)).fillna(0)
    result["trend_d1"] = trend_d1.shift(1).reindex(result.index, method="ffill")

    # --- Zeitbasierte Features ---
    result["hour"] = result.index.hour
    result["day_of_week"] = result.index.dayofweek
    h = result["hour"]
    result["session_london"] = ((h >= 8) & (h < 17)).astype(int)
    result["session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    result["session_asia"] = ((h >= 0) & (h < 9)).astype(int)
    result["session_overlap"] = ((h >= 13) & (h < 17)).astype(int)

    # --- Kill-Zone Features (präzisere Entry-Fenster) ---
    result["killzone_london_open"] = ((h >= 7) & (h < 9)).astype(int)
    result["killzone_ny_open"] = ((h >= 13) & (h < 15)).astype(int)
    result["killzone_asia_open"] = ((h >= 0) & (h < 2)).astype(int)

    # --- Key Levels (PDH/PDL/PWH/PWL) ---
    # LOOK-AHEAD-SCHUTZ: Levels mit shift(1) aus abgeschlossenen Perioden
    day_high = result["high"].resample("1D").max().shift(1)
    day_low = result["low"].resample("1D").min().shift(1)
    week_high = result["high"].resample("W-MON").max().shift(1)
    week_low = result["low"].resample("W-MON").min().shift(1)

    result["pdh"] = day_high.reindex(result.index, method="ffill")
    result["pdl"] = day_low.reindex(result.index, method="ffill")
    result["pwh"] = week_high.reindex(result.index, method="ffill")
    result["pwl"] = week_low.reindex(result.index, method="ffill")

    close_safe = result["close"].replace(0, np.nan)
    result["dist_pdh_pct"] = (result["close"] - result["pdh"]) / close_safe
    result["dist_pdl_pct"] = (result["close"] - result["pdl"]) / close_safe
    result["dist_pwh_pct"] = (result["close"] - result["pwh"]) / close_safe
    result["dist_pwl_pct"] = (result["close"] - result["pwl"]) / close_safe

    key_tol = 0.0015
    result["near_key_level"] = (
        (result["dist_pdh_pct"].abs() <= key_tol)
        | (result["dist_pdl_pct"].abs() <= key_tol)
        | (result["dist_pwh_pct"].abs() <= key_tol)
        | (result["dist_pwl_pct"].abs() <= key_tol)
    ).astype(int)

    # --- Fair Value Gaps (3-Kerzen-Logik) ---
    high_shift2 = result["high"].shift(2)
    low_shift2 = result["low"].shift(2)
    bull_fvg = result["low"] > high_shift2
    bear_fvg = result["high"] < low_shift2
    result["fvg_bullish"] = bull_fvg.astype(int)
    result["fvg_bearish"] = bear_fvg.astype(int)

    bull_gap = (result["low"] - high_shift2) / close_safe
    bear_gap = (low_shift2 - result["high"]) / close_safe
    result["fvg_gap_pct"] = np.where(
        bull_fvg,
        bull_gap,
        np.where(bear_fvg, -bear_gap, 0.0),
    )

    # --- MSS/BOS (Marktstruktur) ---
    pivot_bars = 20
    prev_swing_high = result["high"].shift(1).rolling(pivot_bars).max()
    prev_swing_low = result["low"].shift(1).rolling(pivot_bars).min()
    result["bos_bull"] = (result["close"] > prev_swing_high).astype(int)
    result["bos_bear"] = (result["close"] < prev_swing_low).astype(int)

    structure_bias = np.where(
        result["bos_bull"] == 1,
        1,
        np.where(result["bos_bear"] == 1, -1, 0),
    )
    result["structure_bias"] = (
        pd.Series(structure_bias, index=result.index).ffill().fillna(0)
    )
    prev_bias = result["structure_bias"].shift(1).fillna(0)
    result["mss_bull"] = ((result["bos_bull"] == 1) & (prev_bias < 0)).astype(int)
    result["mss_bear"] = ((result["bos_bear"] == 1) & (prev_bias > 0)).astype(int)

    # --- ADX + Regime-Detection ---
    result["adx_14"] = ind_adx(result)

    # Volatilitätsschwelle (rollender Median 50 Perioden)
    atr_pct = result["atr_pct"]
    median_atr = atr_pct.rolling(window=50, min_periods=50).median()
    adx = result["adx_14"]

    regime = pd.Series(0, index=result.index, dtype=int)
    hoch_vol = atr_pct > (1.5 * median_atr)
    aufwaerts = (adx > 25.0) & (result["close"] > result["sma_50"]) & ~hoch_vol
    abwaerts = (adx > 25.0) & (result["close"] < result["sma_50"]) & ~hoch_vol
    regime[aufwaerts] = 1
    regime[abwaerts] = 2
    regime[hoch_vol] = 3
    result["market_regime"] = regime

    # --- HMM-Regime (optional, mit robustem Fallback) ---
    if HAS_HMMLEARN:
        try:
            hmm_input = np.column_stack(
                [
                    log_ret.fillna(0.0).values,
                    atr_pct.ffill().fillna(0.0).values,
                ]
            )

            hmm_regimes = np.full(len(result), np.nan)
            min_train_bars = min(400, max(120, len(result) // 3))
            refit_interval = 120
            hmm_model = None

            for i in range(min_train_bars, len(result)):
                if hmm_model is None or (i - min_train_bars) % refit_interval == 0:
                    hmm_model = GaussianHMM(
                        n_components=4,
                        covariance_type="diag",
                        n_iter=120,
                        random_state=42,
                    )
                    hmm_model.fit(hmm_input[:i])

                hmm_regimes[i] = int(hmm_model.predict(hmm_input[i : i + 1])[0])

            hmm_state_series = (
                pd.Series(hmm_regimes, index=result.index).ffill().fillna(0).astype(int)
            )
            stats = (
                pd.DataFrame(
                    {
                        "state": hmm_state_series,
                        "ret": log_ret.fillna(0.0),
                        "vol": atr_pct.fillna(0.0),
                    }
                )
                .groupby("state")
                .agg(ret_mean=("ret", "mean"), vol_mean=("vol", "mean"))
            )

            high_vol_state = int(stats["vol_mean"].idxmax()) if len(stats) > 0 else 0
            ret_sorted = stats.drop(index=high_vol_state, errors="ignore").sort_values(
                "ret_mean"
            )
            bear_state = (
                int(ret_sorted.index[0]) if len(ret_sorted) > 0 else high_vol_state
            )
            bull_state = (
                int(ret_sorted.index[-1]) if len(ret_sorted) > 0 else high_vol_state
            )
            regime_map = {int(s): 0 for s in stats.index}
            regime_map[high_vol_state] = 3
            regime_map[bear_state] = 2
            regime_map[bull_state] = 1
            result["market_regime_hmm"] = (
                hmm_state_series.map(regime_map).fillna(0).astype(int)
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("HMM-Regime-Fallback aktiv (Fehler: %s)", e)
            result["market_regime_hmm"] = result["market_regime"]
    else:
        result["market_regime_hmm"] = result["market_regime"]

    return result


# ============================================================
# 3. Externe Features (Fear & Greed + BTC Funding Rate)
# ============================================================


def fear_greed_holen() -> dict:
    """
    Holt den aktuellen Fear & Greed Index von alternative.me.

    Returns:
        Dict mit 'fear_greed_value' (0–100) und 'fear_greed_class' (0–3).
        Fallback-Werte 50 und 1 bei API-Fehler.
    """
    fallback = {"fear_greed_value": 50.0, "fear_greed_class": 1.0}
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=10,
        )
        resp.raise_for_status()
        daten = resp.json().get("data", [{}])[0]
        wert = float(daten.get("value", 50))
        # Klassifizierung: 0=Extreme Fear (0–24), 1=Fear (25–49),
        #                   2=Greed (50–74), 3=Extreme Greed (75–100)
        if wert < 25:
            klasse = 0.0
        elif wert < 50:
            klasse = 1.0
        elif wert < 75:
            klasse = 2.0
        else:
            klasse = 3.0
        logger.info(f"Fear & Greed: {wert:.0f} (Klasse {klasse:.0f})")
        return {"fear_greed_value": wert, "fear_greed_class": klasse}
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"Fear & Greed API Fehler: {e} – Fallback 50/1")
        return fallback


def btc_funding_holen() -> float:
    """
    Holt die aktuelle BTC Funding Rate von Binance Futures.

    Returns:
        Funding Rate als Float (z.B. 0.0001 = 0.01%).
        Fallback-Wert 0.0 bei API-Fehler.
    """
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        rate = float(resp.json().get("lastFundingRate", 0.0))
        logger.info(f"BTC Funding Rate: {rate:.6f}")
        return rate
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"BTC Funding Rate API Fehler: {e} – Fallback 0.0")
        return 0.0


def externe_features_holen(
    cache: Optional[dict] = None, max_age_seconds: int = 240
) -> dict:
    """
    Holt externe Features optional mit Cache, um API-Rauschen zu reduzieren.

    Args:
        cache: Optionales Cache-Dict, das zwischen Loop-Durchläufen weitergegeben wird.
        max_age_seconds: Maximales Alter der Cache-Werte in Sekunden.

    Returns:
        Dict mit fear_greed_value, fear_greed_class, btc_funding_rate.
    """
    now_ts = time.time()
    if cache is not None:
        cached_ts = float(cache.get("fetched_at", 0.0))
        if cached_ts > 0 and (now_ts - cached_ts) <= max_age_seconds:
            return {
                "fear_greed_value": float(cache.get("fear_greed_value", 50.0)),
                "fear_greed_class": float(cache.get("fear_greed_class", 1.0)),
                "btc_funding_rate": float(cache.get("btc_funding_rate", 0.0)),
            }

    fg = fear_greed_holen()
    btc_rate = btc_funding_holen()
    values = {
        "fear_greed_value": float(fg["fear_greed_value"]),
        "fear_greed_class": float(fg["fear_greed_class"]),
        "btc_funding_rate": float(btc_rate),
    }
    if cache is not None:
        cache.update(values)
        cache["fetched_at"] = now_ts
    return values


def externe_features_einfuegen(
    df: pd.DataFrame, external_features: Optional[dict] = None
) -> pd.DataFrame:
    """
    Fügt Fear & Greed und BTC Funding Rate als Features ein.

    Alle Zeilen erhalten denselben aktuellen Wert (für die letzte Kerze
    relevant – ältere Zeilen werden für die Vorhersage eh nicht verwendet).

    Args:
        df: Feature-DataFrame (bereits mit technischen Indikatoren)
        external_features: Optional bereits geholte externe Features (für Loop-Caching)

    Returns:
        DataFrame mit 3 zusätzlichen Spalten.
    """
    if external_features is None:
        external_features = externe_features_holen()

    df["fear_greed_value"] = float(external_features.get("fear_greed_value", 50.0))
    df["fear_greed_class"] = float(external_features.get("fear_greed_class", 1.0))
    df["btc_funding_rate"] = float(external_features.get("btc_funding_rate", 0.0))
    return df


# ============================================================
# 4. Signal-Generierung (Modell-Vorhersage)
# ============================================================


def signal_generieren(
    df: pd.DataFrame,
    modell: object,
    schwelle: float = 0.60,
    short_schwelle: Optional[float] = None,
    decision_mapping: str = "class",
    regime_spalte: str = "market_regime",
    regime_erlaubt: Optional[list] = None,
) -> Tuple[int, float, int, float]:
    """
    Generiert ein Trade-Signal für die letzte Kerze (die gerade geschlossen hat).

    Args:
        df:              Feature-DataFrame (alle 45 Features vorhanden)
        modell:          Geladenes LightGBM-Modell
        schwelle:        Mindest-Wahrscheinlichkeit für Trade-Ausführung
        short_schwelle:  Optionale Short-Schwelle (wenn None => wie schwelle bzw. 1-schwelle)
        decision_mapping:
            "class"     => Long wenn proba_long >= schwelle, Short wenn proba_short >= short_schwelle
            "long_prob" => Long wenn proba_long >= schwelle, Short wenn proba_long <= short_schwelle
        regime_spalte:   Welche Regime-Spalte genutzt wird ("market_regime" oder "market_regime_hmm")
        regime_erlaubt:  Erlaubte Regime-Nummern (None = alle)

    Returns:
        Tuple (signal, prob, regime, atr_pct):
            signal:  2=Long, -1=Short, 0=Kein Trade
            prob:    Wahrscheinlichkeit des Signals (0–1)
            regime:  Aktuelles Markt-Regime (0–3)
            atr_pct: ATR_14 als Prozent vom Close (für ATR-SL Berechnung)
    """
    # Letzte vollständige Kerze (Index -1 = aktuelle Kerze, -2 = letzte geschlossene)
    # Wir verwenden die letzte vollständige Kerze für das Signal
    letzte_kerze = df.iloc[[-2]]  # -2: letzte geschlossene Kerze (sicher!)

    # Aktuelles Regime aus konfigurierter Spalte lesen (mit sicherem Fallback)
    regime_spalte_eff = regime_spalte if regime_spalte in letzte_kerze.columns else "market_regime"
    if regime_spalte_eff != regime_spalte:
        logger.warning(
            f"Regime-Spalte '{regime_spalte}' nicht vorhanden – fallback auf '{regime_spalte_eff}'"
        )
    aktuelles_regime = int(letzte_kerze[regime_spalte_eff].iloc[0])

    # ATR als Prozent vom Close (für dynamisches Stop-Loss)
    atr_pct = 0.0
    if "atr_14" in letzte_kerze.columns and "close" in df.columns:
        atr_abs = float(letzte_kerze["atr_14"].iloc[0])
        close_preis = float(letzte_kerze["close"].iloc[0])
        if close_preis > 0:
            atr_pct = atr_abs / close_preis  # z.B. 0.0021 = 0.21%

    # Regime-Filter prüfen
    if regime_erlaubt is not None:
        if aktuelles_regime not in regime_erlaubt:
            regime_name = REGIME_NAMEN.get(aktuelles_regime, "?")
            logger.info(
                f"Signal übersprungen: Regime '{regime_name}' nicht in "
                f"{[REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]}"
            )
            return 0, 0.0, aktuelles_regime, atr_pct

    # Features für Modell vorbereiten (NaN-Werte mit Median auffüllen)
    # Nutze modell.feature_name_ wenn verfügbar (exakte Trainings-Features)
    if hasattr(modell, "feature_name_") and modell.feature_name_:
        modell_features = list(modell.feature_name_)
    else:
        modell_features = FEATURE_SPALTEN

    verfuegbare = [f for f in modell_features if f in df.columns]
    fehlende = [f for f in modell_features if f not in df.columns]
    if fehlende:
        logger.warning(f"Fehlende Features: {fehlende} – werden mit 0 gefüllt")
        for feat in fehlende:
            letzte_kerze[feat] = 0.0

    x_features = letzte_kerze[verfuegbare].copy()

    # NaN auffüllen (Sicherheitsnetz)
    if x_features.isna().any().any():
        logger.warning(
            "NaN-Werte in Features – werden mit Median der letzten 50 Kerzen gefüllt"
        )
        nan_fill = df[verfuegbare].iloc[-50:].median()
        x_features = x_features.fillna(nan_fill)

    # Modell-Vorhersage: Wahrscheinlichkeiten für alle 3 Klassen
    # proba[:,0] = Short (0→-1), proba[:,1] = Neutral, proba[:,2] = Long
    proba = modell.predict_proba(x_features)[0]
    raw_pred = int(np.argmax(proba))
    long_prob = float(proba[2])
    short_prob = float(proba[0])
    short_schwelle_eff = float(short_schwelle) if short_schwelle is not None else float(
        1.0 - schwelle if decision_mapping == "long_prob" else schwelle
    )

    # DEBUG: Detailliertes Logging der Wahrscheinlichkeiten
    logger.info(
        f"Modell-Output: Short={proba[0]:.1%}, Neutral={proba[1]:.1%}, Long={proba[2]:.1%} | "
        f"raw_pred={raw_pred} | Mapping={decision_mapping} | "
        f"Long-Schwelle={schwelle:.1%} | Short-Schwelle={short_schwelle_eff:.1%}"
    )

    # Signal mit Schwellenwert-Filter
    if decision_mapping == "long_prob":
        if long_prob >= schwelle:
            logger.info(
                f"→ Long-Signal ausgelöst (proba_long={long_prob:.1%} >= {schwelle:.1%})"
            )
            return 2, long_prob, aktuelles_regime, atr_pct
        if long_prob <= short_schwelle_eff:
            logger.info(
                f"→ Short-Signal ausgelöst (proba_long={long_prob:.1%} <= {short_schwelle_eff:.1%})"
            )
            return -1, 1.0 - long_prob, aktuelles_regime, atr_pct
    else:
        if raw_pred == 2 and long_prob >= schwelle:
            logger.info(
                f"→ Long-Signal ausgelöst (proba_long={long_prob:.1%} >= {schwelle:.1%})"
            )
            return 2, long_prob, aktuelles_regime, atr_pct
        if raw_pred == 0 and short_prob >= short_schwelle_eff:
            logger.info(
                f"→ Short-Signal ausgelöst (proba_short={short_prob:.1%} >= {short_schwelle_eff:.1%})"
            )
            return -1, short_prob, aktuelles_regime, atr_pct

    logger.info(
        f"→ Kein Signal (raw_pred={raw_pred}, höchste Prob={max(proba):.1%}, aber Schwelle nicht erfüllt)"
    )
    return 0, float(max(proba)), aktuelles_regime, atr_pct  # Kein Trade


def shadow_signal_generieren(
    symbol: str,
    df: pd.DataFrame,
    modell: object,
    schwelle: float = 0.60,
    short_schwelle: Optional[float] = None,
    decision_mapping: str = "class",
    regime_spalte: str = "market_regime",
    two_stage_kongruenz: bool = True,
    regime_erlaubt: Optional[list] = None,
    two_stage_config: Optional[dict] = None,
) -> Tuple[int, float, int, float]:
    """
    Shadow-Mode für Two-Stage-Integration (Phase 7A Step 5).

    Diese Funktion routet symbol-basiert zwischen Single-Stage und Two-Stage:
        - USDCAD/USDJPY mit v4-Modellen → Two-Stage (HTF H1 + LTF M5)
        - Alle anderen Symbole  → Single-Stage (bestehende Logik)

    Bei jedem Fehler im Two-Stage-Pfad: Hard Fallback zu Single-Stage.

    Args:
        symbol:           Handelssymbol (USDCAD, USDJPY, ...)
        df:               Feature-DataFrame (mit allen Features)
        modell:           Single-Stage-Modell (Fallback)
        schwelle:         Wahrscheinlichkeits-Schwelle
        short_schwelle:   Optionale Short-Schwelle
        decision_mapping: "class" oder "long_prob"
        regime_spalte:    Regime-Quelle ("market_regime" oder "market_regime_hmm")
        two_stage_kongruenz: True=Kongruenzfilter aktiv, False=deaktiviert (aggressiver)
        regime_erlaubt:   Erlaubte Regime oder None
        two_stage_config: Dict mit {
                             "enable": bool,
                             "ltf_timeframe": str (M5/M15),
                             "version": str (v4/v1/...),
                             "htf_features": list,
                             "ltf_features": list,
                             "htf_df": pd.DataFrame (H1-Daten),
                             "ltf_df": pd.DataFrame (M5/M15-Daten)
                          }

    Returns:
        Tuple (signal, prob, regime, atr_pct) – identisch zu signal_generieren()
    """
    # ---- Fallback-Strategie: Single-Stage als Baseline ----
    baseline_signal, baseline_prob, baseline_regime, baseline_atr = signal_generieren(
        df=df,
        modell=modell,
        schwelle=schwelle,
        short_schwelle=short_schwelle,
        decision_mapping=decision_mapping,
        regime_spalte=regime_spalte,
        regime_erlaubt=regime_erlaubt,
    )
    if decision_mapping == "long_prob":
        if baseline_signal == 2:
            baseline_prob_label = "proba_long"
        elif baseline_signal == -1:
            baseline_prob_label = "short_score(1-proba_long)"
        else:
            baseline_prob_label = "score"
    else:
        baseline_prob_label = "proba_class"

    # ---- Two-Stage nur wenn explizit enabled und für freigegebene Symbole ----
    TWO_STAGE_APPROVED = {"USDCAD", "USDJPY"}
    if not two_stage_config or not two_stage_config.get("enable", False):
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr

    if symbol.upper() not in TWO_STAGE_APPROVED:
        logger.debug(
            f"[{symbol}] Two-Stage deaktiviert (nur {TWO_STAGE_APPROVED} approved)"
        )
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr

    # ---- Two-Stage-Pfad mit Hard Fallback ----
    try:
        if not TWO_STAGE_VERFUEGBAR:
            logger.warning(
                f"[{symbol}] Two-Stage-Modul nicht verfügbar – Fallback Single-Stage"
            )
            return baseline_signal, baseline_prob, baseline_regime, baseline_atr

        # Modelle laden (lazy loading – nur beim ersten Aufruf)
        if "htf_model" not in two_stage_config or "ltf_model" not in two_stage_config:
            ltf_tf = two_stage_config.get("ltf_timeframe", "M5")
            version = two_stage_config.get("version", "v4")

            htf_model, ltf_model = two_stage_modelle_laden(
                models_dir=MODEL_DIR,
                symbol=symbol,
                ltf_timeframe=ltf_tf,
                version=version,
            )
            two_stage_config["htf_model"] = htf_model
            two_stage_config["ltf_model"] = ltf_model
            logger.info(
                f"[{symbol}] Two-Stage-Modelle geladen: H1 HTF + {ltf_tf} LTF ({version})"
            )

        # HTF und LTF DataFrames müssen bereitgestellt werden
        htf_df = two_stage_config.get("htf_df")
        ltf_df = two_stage_config.get("ltf_df")
        if htf_df is None or ltf_df is None:
            logger.warning(
                f"[{symbol}] HTF/LTF DataFrames fehlen – Fallback Single-Stage"
            )
            return baseline_signal, baseline_prob, baseline_regime, baseline_atr

        # Two-Stage-Signal generieren
        ts_signal = zwei_stufen_signal(
            htf_df=htf_df,
            ltf_df=ltf_df,
            htf_model=two_stage_config["htf_model"],
            ltf_model=two_stage_config["ltf_model"],
            htf_feature_spalten=two_stage_config.get("htf_features", FEATURE_SPALTEN),
            ltf_feature_spalten=two_stage_config.get("ltf_features", FEATURE_SPALTEN),
            schwelle=schwelle,
        )

        # ---- Kongruenz-Filter: HTF-Bias und LTF-Signal müssen übereinstimmen ----
        # HTF-Bias 0=Short, 1=Neutral, 2=Long | LTF-Signal -1=Short, 0=Neutral, 2=Long
        # Erlaubt: HTF-Short + LTF-Short, HTF-Long + LTF-Long
        # Blockiert: HTF-Bias widerspricht LTF-Signal, oder HTF=Neutral
        htf_bias = ts_signal.htf_bias_klasse
        ltf_signal = ts_signal.signal
        kongruent = True  # Standardannahme: neutral-Signale sind immer OK

        if ltf_signal != 0:  # Nur aktive Trades prüfen (Short/Long)
            if htf_bias == 1:
                # HTF sagt Neutral → kein aktiver Trade erlaubt
                kongruent = False
            elif htf_bias == 0 and ltf_signal != -1:
                # HTF sagt Short, aber LTF will Long → blockieren
                kongruent = False
            elif htf_bias == 2 and ltf_signal != 2:
                # HTF sagt Long, aber LTF will Short → blockieren
                kongruent = False

        if two_stage_kongruenz and not kongruent:
            logger.info(
                f"[{symbol}] ⛔ KONGRUENZ-FILTER | "
                f"LTF-Signal={ltf_signal} BLOCKIERT (HTF-Bias={htf_bias}) | "
                f"LTF-Prob={ts_signal.prob:.1%} | Baseline={baseline_signal} "
                f"({baseline_prob_label}={baseline_prob:.1%})"
            )
            # Signal auf Neutral setzen, Keep Prob für Logging
            return (0, ts_signal.prob, baseline_regime, baseline_atr)
        if not two_stage_kongruenz and not kongruent:
            logger.info(
                f"[{symbol}] ⚠️ KONGRUENZ-FILTER DEAKTIVIERT | "
                f"LTF-Signal={ltf_signal} wird trotz HTF-Bias={htf_bias} durchgelassen"
            )

        # Logging: Shadow vs. Baseline Vergleich
        if ts_signal.signal != baseline_signal:
            logger.info(
                f"[{symbol}] 🔀 SHADOW-DIVERGENZ | "
                f"Two-Stage={ts_signal.signal} (prob={ts_signal.prob:.1%}, HTF-bias={ts_signal.htf_bias_klasse}) | "
                f"Baseline={baseline_signal} ({baseline_prob_label}={baseline_prob:.1%})"
            )
        else:
            logger.info(
                f"[{symbol}] ✓ SHADOW-KONGRUENZ | Signal={ts_signal.signal} | "
                f"Two-Stage-Prob={ts_signal.prob:.1%}, Baseline-{baseline_prob_label}={baseline_prob:.1%}"
            )

        # Two-Stage-Signal verwenden (Shadow-Mode aktiv)
        return (
            ts_signal.signal,
            ts_signal.prob,
            baseline_regime,  # Regime aus Baseline (gleich)
            baseline_atr,  # ATR aus Baseline (gleich)
        )

    except FileNotFoundError as e:
        logger.warning(
            f"[{symbol}] Two-Stage-Modelle nicht gefunden: {e} – Fallback Single-Stage"
        )
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            f"[{symbol}] Two-Stage-Fehler: {e} – Fallback Single-Stage",
            exc_info=True,
        )
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr


# ============================================================
# 5. MetaTrader 5 Funktionen
# ============================================================


def mt5_verbinden(server: str, login: int, password: str, pfad: str = "") -> bool:
    """
    Verbindet mit dem MT5-Terminal.

    Das MT5-Terminal muss bereits geöffnet und eingeloggt sein.
    Das Skript verbindet sich mit der laufenden MT5-Instanz.

    Args:
        server:   Broker-Server (z.B. "ICMarkets-Demo")
        login:    Kontonummer
        password: Passwort
        pfad:     Optionaler Pfad zur terminal64.exe (z.B. für portable Installation)

    Returns:
        True bei Erfolg, False bei Fehler.
    """
    if not MT5_VERFUEGBAR:
        logger.warning(
            "MetaTrader5 nicht installiert – nur Paper-Trading möglich!\n"
            "Lösung: pip install MetaTrader5"
        )
        return False

    # MT5 initialisieren und verbinden
    if pfad:
        ok = mt5.initialize(path=pfad, server=server, login=login, password=password)
    else:
        ok = mt5.initialize(server=server, login=login, password=password)

    if not ok:
        logger.error(f"MT5 Verbindung fehlgeschlagen: {mt5.last_error()}")
        return False

    # Konto-Info ausgeben
    konto = mt5.account_info()
    logger.info(
        f"MT5 verbunden | Server: {server} | "
        f"Konto: {konto.login} | Saldo: {konto.balance:.2f} {konto.currency}"
    )
    return True


def mt5_timeframe_konstante(timeframe: str) -> Optional[int]:
    """
    Übersetzt den String-Zeitrahmen in die passende MT5-Konstante.

    Args:
        timeframe: Zeitrahmen als String ("H1", "M30" oder "M15")

    Returns:
        MT5-Timeframe-Konstante oder None bei unbekanntem Zeitrahmen.
    """
    if not MT5_VERFUEGBAR:
        return None

    cfg = TIMEFRAME_CONFIG.get(timeframe)
    if cfg is None:
        return None

    konst_name = cfg["mt5_name"]
    return getattr(mt5, konst_name, None)


def n_barren_fuer_timeframe(timeframe: str) -> int:
    """
    Berechnet die benötigte Barrenanzahl für denselben Zeit-Buffer je Zeitrahmen.

    H1 nutzt standardmäßig 500 Bars. Für kleinere Timeframes (M30/M15)
    werden entsprechend mehr Bars geladen, um denselben Zeitbereich abzudecken.

    Args:
        timeframe: Zeitrahmen ("H1", "M30" oder "M15")

    Returns:
        Empfohlene Barrenanzahl.
    """
    bars_per_hour = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
        "bars_per_hour"
    ]
    return N_BARREN * bars_per_hour


def mt5_daten_holen(
    symbol: str,
    timeframe: str = "H1",
    n_barren: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Holt die letzten Barren im gewählten Zeitrahmen von MT5.

    Args:
        symbol:   Handelssymbol (z.B. "USDCAD")
        n_barren: Anzahl der Barren (None = automatisch je Zeitrahmen)

    Returns:
        OHLCV DataFrame mit UTC-DatetimeIndex oder None bei Fehler.
    """
    if not MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – keine Live-Daten!")
        return None

    # Barren von Position 0 (aktuelle Kerze) bis n-1
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        logger.error(f"Unbekannter oder nicht verfügbarer Zeitrahmen: {timeframe}")
        return None

    n_bars_effektiv = (
        n_barren if n_barren is not None else n_barren_fuer_timeframe(timeframe)
    )

    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars_effektiv)
    if rates is None:
        logger.error(f"[{symbol}] Keine Daten von MT5: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    # MT5 liefert Unix-Timestamp in Sekunden (Broker-Zeitzone)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # Nur OHLCV-Spalten behalten
    return df[["open", "high", "low", "close", "volume"]]


def mt5_letzte_kerze_uhrzeit(symbol: str, timeframe: str = "H1") -> Optional[datetime]:
    """
    Gibt die Öffnungszeit der letzten geschlossenen Kerze zurück.

    Args:
        symbol: Handelssymbol

    Returns:
        datetime (UTC) der letzten Kerzen-Eröffnung oder None.
    """
    if not MT5_VERFUEGBAR:
        return None
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 2)
    if rates is None or len(rates) < 2:
        logger.warning(f"[{symbol}] MT5 liefert keine Kerzen-Daten: {mt5.last_error()}")
        return None
    # Index 1 = letzte geschlossene Kerze
    return datetime.fromtimestamp(int(rates[1]["time"]), tz=timezone.utc)


def mt5_offene_position(symbol: str) -> bool:
    """
    Prüft ob bereits eine offene Position für das Symbol existiert.

    Args:
        symbol: Handelssymbol

    Returns:
        True wenn eine offene Position existiert (ML-Trade mit MAGIC_NUMBER).
    """
    if not MT5_VERFUEGBAR:
        return False
    positionen = mt5.positions_get(symbol=symbol)
    if positionen is None:
        return False
    # Nur eigene ML-Positionen (erkennbar an der MAGIC_NUMBER)
    return any(p.magic == MAGIC_NUMBER for p in positionen)


def order_senden(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol: str,
    richtung: int,
    lot: float,
    tp_pct: float,
    sl_pct: float,
    paper_trading: bool = True,
) -> bool:
    """
    Sendet eine Market Order an MT5 (oder loggt sie im Paper-Modus).

    SICHERHEIT: Stop-Loss ist PFLICHT – keine Order ohne SL!

    Args:
        symbol:        Handelssymbol
        richtung:      2=Long, -1=Short
        lot:           Lot-Größe (Standard: 0.01 = Micro-Lot)
        tp_pct:        Take-Profit in Prozent (z.B. 0.003 = 0.3%)
        sl_pct:        Stop-Loss in Prozent   (z.B. 0.003 = 0.3%)
        paper_trading: True = nur loggen (kein echtes Geld!)

    Returns:
        True bei Erfolg, False bei Fehler.
    """
    richtung_str = "LONG (Kaufen)" if richtung == 2 else "SHORT (Verkaufen)"

    if paper_trading:
        logger.info(
            f"[PAPER] {symbol} {richtung_str} | "
            f"Lot={lot} | TP={tp_pct:.1%} | SL={sl_pct:.1%}"
        )
        return True

    # ====== ECHTE ORDER ======
    if not MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – Order nicht gesendet!")
        return False

    # Symbol aktivieren (falls nicht im Market Watch)
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} nicht verfügbar!")
        return False

    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        logger.error(f"Symbol-Info für {symbol} nicht abrufbar!")
        return False

    # Preis und TP/SL berechnen
    if richtung == 2:  # Long: Buy
        order_type = mt5.ORDER_TYPE_BUY
        preis = tick.ask
        sl_preis = round(preis * (1.0 - sl_pct), symbol_info.digits)
        tp_preis = round(preis * (1.0 + tp_pct), symbol_info.digits)
    else:  # Short: Sell
        order_type = mt5.ORDER_TYPE_SELL
        preis = tick.bid
        sl_preis = round(preis * (1.0 + sl_pct), symbol_info.digits)
        tp_preis = round(preis * (1.0 - tp_pct), symbol_info.digits)

    # Order-Request (PFLICHT: Stop-Loss!)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": preis,
        "sl": sl_preis,  # Stop-Loss ist PFLICHT!
        "tp": tp_preis,
        "deviation": 20,  # Max. Slippage in Punkte
        "magic": MAGIC_NUMBER,  # Eindeutige ID für diesen Bot
        "comment": "ML-Phase6",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order fehlgeschlagen: Code={result.retcode} | {result.comment}")
        return False

    logger.info(
        f"Order ausgeführt: {symbol} {richtung_str} | "
        f"{lot} Lot @ {preis:.5f} | SL={sl_preis:.5f} | TP={tp_preis:.5f}"
    )
    return True


# ============================================================
# 6. Trade-Logging (CSV)
# ============================================================


def trade_loggen(
    symbol: str,
    richtung: int,
    prob: float,
    regime: int,
    paper_trading: bool,
    entry_price: float = 0.0,
    sl_price: float = 0.0,
    tp_price: float = 0.0,
) -> None:
    """
    Schreibt Signal-/Heartbeat-Events in eine CSV-Datei.

    Schreibt zusätzlich eine Kopie in den MT5 Common/Files-Ordner,
    damit das LiveSignalDashboard.mq5 die Daten lesen kann.

    Args:
        symbol:        Handelssymbol
        richtung:      2=Long, -1=Short, 0=Kein Trade (Heartbeat/No-Signal)
        prob:          Signal-Wahrscheinlichkeit
        regime:        Markt-Regime (0–3)
        paper_trading: True = Paper-Modus aktiv
        entry_price:   Einstiegspreis (0 bei Heartbeat/No-Signal)
        sl_price:      Stop-Loss-Preis (0 bei Heartbeat/No-Signal)
        tp_price:      Take-Profit-Preis (0 bei Heartbeat/No-Signal)
    """
    log_pfad = LOG_DIR / f"{symbol}_live_trades.csv"

    eintrag = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "richtung": "Long" if richtung == 2 else "Short" if richtung == -1 else "Kein",
        "signal": richtung,
        "prob": round(prob, 4),
        "regime": regime,
        "regime_name": REGIME_NAMEN.get(regime, "?"),
        "paper_trading": paper_trading,
        "modus": "PAPER" if paper_trading else "LIVE",
        "entry_price": round(entry_price, 5),
        "sl_price": round(sl_price, 5),
        "tp_price": round(tp_price, 5),
    }

    df_log = pd.DataFrame([eintrag])
    # CSV anhängen (header=False wenn Datei bereits existiert)
    df_log.to_csv(
        log_pfad,
        mode="a",
        header=not log_pfad.exists(),
        index=False,
    )

    # Kopie in MT5 Common/Files für LiveSignalDashboard.mq5
    try:
        mt5_common = (
            Path(os.environ.get("APPDATA", ""))
            / "MetaQuotes"
            / "Terminal"
            / "Common"
            / "Files"
        )
        if mt5_common.exists():
            mt5_csv = mt5_common / f"{symbol}_live_trades.csv"
            df_log.to_csv(
                mt5_csv,
                mode="a",
                header=not mt5_csv.exists(),
                index=False,
            )
    except Exception:
        pass  # Dashboard-Sync ist nice-to-have, kein Fehler


# ============================================================
# 7. Kill-Switch – Harter Stopp bei zu hohem Drawdown
# ============================================================


def kill_switch_pruefen(
    symbol: str,
    start_equity: float,
    aktuell_equity: float,
    max_dd_pct: float,
    paper_trading: bool,
) -> bool:
    """
    Prüft ob der Kill-Switch ausgelöst werden soll (Review-Punkt 8).

    Im LIVE-Modus: Vergleicht MT5-Kontostand mit Startkapital.
    Im PAPER-Modus: Vergleicht simuliertes Kapital (wird von trading_loop übergeben).

    Args:
        symbol:        Handelssymbol (für Logging)
        start_equity:  Startkapital der Session (z.B. 10000.0)
        aktuell_equity: Aktueller Kontostand oder simuliertes Kapital
        max_dd_pct:    Maximaler Drawdown in Dezimal (z.B. 0.15 = 15%)
        paper_trading: True = Paper-Modus

    Returns:
        True wenn Kill-Switch ausgelöst wird (Trader soll stoppen).
    """
    # Drawdown berechnen: wie viel % des Startkapitals wurde verloren?
    verlust = start_equity - aktuell_equity
    drawdown_pct = verlust / start_equity if start_equity > 0 else 0.0

    if drawdown_pct >= max_dd_pct:
        modus = "PAPER" if paper_trading else "LIVE"
        logger.critical("=" * 65)
        logger.critical(f"[{symbol}] ⛔ KILL-SWITCH AUSGELÖST! [{modus}]")
        logger.critical(
            f"[{symbol}] Drawdown: {drawdown_pct:.1%} " f"(Limit: {max_dd_pct:.1%})"
        )
        logger.critical(
            f"[{symbol}] Startkapital: {start_equity:.2f} | "
            f"Aktuell: {aktuell_equity:.2f} | Verlust: {verlust:.2f}"
        )
        logger.critical(f"[{symbol}] Trader wird automatisch gestoppt!")
        logger.critical("=" * 65)
        return True

    # Warnstufen (50% und 75% des Limits)
    if drawdown_pct >= max_dd_pct * 0.75:
        logger.warning(
            f"[{symbol}] ⚠️  Drawdown-WARNUNG: {drawdown_pct:.1%} "
            f"(Kill-Switch-Limit: {max_dd_pct:.1%})"
        )
    elif drawdown_pct >= max_dd_pct * 0.50:
        logger.warning(
            f"[{symbol}] Drawdown {drawdown_pct:.1%} – "
            f"Kill-Switch bei {max_dd_pct:.1%}"
        )

    return False


def alle_positionen_schliessen(symbol: str) -> None:
    """
    Schließt alle offenen MT5-Positionen für dieses Symbol (nach Kill-Switch).

    Nur im Live-Modus relevant – Paper-Modus hat keine echten Positionen.

    Args:
        symbol: Handelssymbol
    """
    if not MT5_VERFUEGBAR:
        return

    positionen = mt5.positions_get(symbol=symbol)  # type: ignore[union-attr]
    if not positionen:
        return

    logger.info(f"[{symbol}] Schließe {len(positionen)} offene Position(en) ...")
    for pos in positionen:
        # Gegenläufige Order zum Schließen der Position
        tick = mt5.symbol_info_tick(symbol)  # type: ignore[union-attr]
        if tick is None:
            continue

        # Long-Position → mit Sell schließen; Short-Position → mit Buy schließen
        if pos.type == mt5.ORDER_TYPE_BUY:  # type: ignore[union-attr]
            close_type = mt5.ORDER_TYPE_SELL  # type: ignore[union-attr]
            preis = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY  # type: ignore[union-attr]
            preis = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,  # type: ignore[union-attr]
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": preis,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Kill-Switch",
            "type_time": mt5.ORDER_TIME_GTC,  # type: ignore[union-attr]
            "type_filling": mt5.ORDER_FILLING_IOC,  # type: ignore[union-attr]
        }
        result = mt5.order_send(request)  # type: ignore[union-attr]
        if result.retcode == mt5.TRADE_RETCODE_DONE:  # type: ignore[union-attr]
            logger.info(f"[{symbol}] Position {pos.ticket} geschlossen ✓")
        else:
            logger.error(
                f"[{symbol}] Schließen fehlgeschlagen: "
                f"Code={result.retcode} | {result.comment}"
            )


# ============================================================
# 8. Haupt-Trading-Schleife
# ============================================================


def neue_kerze_abwarten(
    symbol: str,
    letzte_kerzen_zeit: Optional[datetime],
    timeframe: str = "H1",
) -> bool:
    """
    Prüft ob eine neue H1-Kerze geöffnet wurde.

    Args:
        symbol:            Handelssymbol
        letzte_kerzen_zeit: Zeitstempel der letzten verarbeiteten Kerze

    Returns:
        True wenn neue Kerze verfügbar.
    """
    aktuelle_kerze = mt5_letzte_kerze_uhrzeit(symbol, timeframe)
    if aktuelle_kerze is None:
        logger.debug(f"[{symbol}] mt5_letzte_kerze_uhrzeit() gibt None zurück")
        return False

    # Debug-Logging (alle 60 Sekunden, um Spam zu vermeiden)
    import time as time_module

    if not hasattr(neue_kerze_abwarten, "_last_debug"):
        neue_kerze_abwarten._last_debug = 0
    jetzt = time_module.time()
    if jetzt - neue_kerze_abwarten._last_debug > 60:
        logger.debug(
            f"[{symbol}] Kerzen-Check: Letzte={letzte_kerzen_zeit} | "
            f"Aktuell={aktuelle_kerze}"
        )
        neue_kerze_abwarten._last_debug = jetzt

    return letzte_kerzen_zeit is None or aktuelle_kerze != letzte_kerzen_zeit


def trading_loop(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    symbol: str,
    schwelle: float,
    short_schwelle: Optional[float],
    decision_mapping: str,
    regime_spalte: str,
    two_stage_kongruenz: bool,
    regime_erlaubt: Optional[list],
    paper_trading: bool,
    lot: float,
    modell: object,
    kill_switch_dd: float = KILL_SWITCH_DD_DEFAULT,
    kapital_start: float = 10000.0,
    heartbeat_log: bool = HEARTBEAT_LOG_DEFAULT,
    timeframe: str = "H1",
    atr_sl_aktiv: bool = True,
    atr_sl_faktor: float = 1.5,
    two_stage_config: Optional[dict] = None,
) -> None:
    """
    Haupt-Schleife: Läuft dauerhaft und wartet auf neue Kerzen im gewählten Zeitrahmen.

    Bei jeder neuen Kerze:
    1. MT5-Daten holen
    2. Features berechnen
    3. Signal generieren (Shadow-Mode: Single-Stage vs. Two-Stage)
    4. Kill-Switch prüfen (Drawdown-Limit!)
    5. Trade ausführen (falls Signal stark genug)

    Args:
        symbol:          Handelssymbol
        schwelle:        Wahrscheinlichkeits-Schwelle (z.B. 0.60)
        short_schwelle:  Optionale Short-Schwelle
        decision_mapping: Mapping-Modus ("class" oder "long_prob")
        regime_spalte:   Regime-Quelle ("market_regime" oder "market_regime_hmm")
        two_stage_kongruenz: True=Kongruenzfilter aktiv, False=deaktiviert
        regime_erlaubt:  Erlaubte Regime oder None für alle
        paper_trading:   True = Paper-Modus (kein echtes Geld!)
        lot:             Lot-Größe
        modell:          Geladenes LightGBM-Modell (Single-Stage)
        kill_switch_dd:  Max. Drawdown bis zum automatischen Stopp (Standard: 0.15 = 15%)
        kapital_start:   Startkapital für Paper-Tracking und Kill-Switch-Berechnung
        heartbeat_log:   True = schreibe pro neuer Kerze einen CSV-Heartbeat
        atr_sl_aktiv:    True = ATR-basiertes SL (dynamisch), False = festes SL
        atr_sl_faktor:   ATR-Multiplikator für SL (Standard: 1.5)
        two_stage_config: Two-Stage-Konfiguration für Shadow-Mode (optional)
    """
    modus_str = "PAPER-TRADING" if paper_trading else "⚠️  LIVE-TRADING MIT ECHTEM GELD!"
    regime_str = (
        [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        if regime_erlaubt
        else "alle"
    )

    # Two-Stage M5-Takt: Flag vorab setzen (wird für Logging + Loop gebraucht)
    ts_aktiv = two_stage_config is not None and two_stage_config.get("enable", False)

    logger.info("=" * 65)
    logger.info(f"LIVE-TRADER GESTARTET – {symbol}")
    logger.info(f"Modus:          {modus_str}")
    logger.info(f"Long-Schwelle:  {schwelle:.0%}")
    if short_schwelle is None:
        logger.info("Short-Schwelle: auto")
    else:
        logger.info(f"Short-Schwelle: {short_schwelle:.0%}")
    logger.info(f"Mapping-Modus:  {decision_mapping}")
    logger.info(f"Regime-Quelle:  {regime_spalte}")
    logger.info(
        f"Kongruenz-Filter: {'aktiv' if two_stage_kongruenz else 'deaktiviert (aggressiv)'}"
    )
    logger.info(f"Regime-Filter:  {regime_str}")
    if atr_sl_aktiv:
        logger.info(
            f"Stop-Loss:      ATR-SL aktiv ({atr_sl_faktor}× ATR_14, dynamisch)"
        )
    else:
        logger.info(
            f"TP/SL:          {TP_PCT:.1%} / {SL_PCT:.1%} (RRR={TP_PCT/SL_PCT:.1f}:1)"
        )
    logger.info(f"Lot-Größe:      {lot}")
    logger.info(
        f"Kill-Switch:    Drawdown > {kill_switch_dd:.0%} → automatischer Stopp"
    )
    logger.info(f"Startkapital:   {kapital_start:,.2f} (für Kill-Switch-Berechnung)")
    logger.info(
        f"Heartbeat-Log:  {'aktiv' if heartbeat_log else 'aus'} "
        "(CSV-Update pro Kerze)"
    )
    logger.info(f"Logs:           {LOG_DIR}")
    logger.info(f"Zeitrahmen:     {timeframe}")
    if ts_aktiv:
        logger.info(
            f"M5-Takt:        AKTIV → Loop auf {two_stage_config.get('ltf_timeframe', 'M5')}, "
            f"HTF-Bias auf H1 (gecached)"
        )
        logger.info(
            f"Warte auf neue {two_stage_config.get('ltf_timeframe', 'M5')}-Kerze ..."
        )
    else:
        logger.info(f"Warte auf neue {timeframe}-Kerze ...")
    logger.info("=" * 65)

    letzte_kerzen_zeit: Optional[datetime] = None
    n_signale = 0  # Gesamt-Signale
    n_trades = 0  # Ausgeführte Trades

    # ---- Kill-Switch: Startkapital ermitteln ----
    # Im Live-Modus: echtes Kontostand von MT5 lesen
    # Im Paper-Modus: übergebenes Startkapital verwenden
    if not paper_trading and MT5_VERFUEGBAR:
        account = mt5.account_info()  # type: ignore[union-attr]
        if account:
            # Echtes Startkapital aus MT5-Konto
            start_equity = account.equity
            logger.info(
                f"[{symbol}] MT5-Startkapital: {start_equity:,.2f} {account.currency}"
            )
        else:
            logger.warning(
                f"[{symbol}] MT5-Kontodaten nicht lesbar – Kill-Switch nutzt Startkapital {kapital_start}"
            )
            start_equity = kapital_start
    else:
        # Paper-Modus: konfiguriertes Startkapital verwenden
        start_equity = kapital_start
        logger.info(f"[{symbol}] Paper-Startkapital: {start_equity:,.2f} (simuliert)")

    # Simuliertes Kapital für Paper-Modus (wird nach jedem Trade aktualisiert)
    paper_kapital = start_equity

    # ---- Two-Stage M5-Takt: effektiver Zeitrahmen und HTF-Cache ----
    # Wenn Two-Stage aktiv: Loop läuft auf LTF (M5) statt H1
    # HTF-Bias (H1) wird gecached und nur bei neuer H1-Kerze aktualisiert
    if ts_aktiv:
        effektiver_tf = two_stage_config.get("ltf_timeframe", "M5")
        logger.info(
            f"[{symbol}] ⚡ M5-TAKT AKTIV: Loop-Intervall = {effektiver_tf} "
            f"(alle {TIMEFRAME_CONFIG[effektiver_tf]['minutes_per_bar']} Min) | "
            f"HTF-Bias = H1 (gecached, Update nur bei neuer H1-Kerze)"
        )
    else:
        effektiver_tf = timeframe

    # HTF-Cache Variablen (nur für Two-Stage M5-Takt)
    letzte_htf_kerzen_zeit: Optional[datetime] = None
    cached_h1_df_clean: Optional[pd.DataFrame] = None
    externe_feature_cache: dict = {}

    while True:
        try:
            # Neue Kerze abwarten (M5 bei Two-Stage, sonst H1)
            if not neue_kerze_abwarten(symbol, letzte_kerzen_zeit, effektiver_tf):
                time.sleep(15)
                continue

            # Neue Kerze erkannt!
            jetzt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\n[{symbol}] Neue {effektiver_tf}-Kerze | {jetzt} UTC")
            externe_features = externe_features_holen(cache=externe_feature_cache)

            # ---- Kill-Switch prüfen ----
            # Im Live-Modus: aktuellen MT5-Kontostand lesen
            # Im Paper-Modus: simuliertes Kapital prüfen
            if not paper_trading and MT5_VERFUEGBAR:
                account = mt5.account_info()  # type: ignore[union-attr]
                if account:
                    aktuell_equity = account.equity
                else:
                    aktuell_equity = start_equity  # Sicherheitshalber
            else:
                # Paper-Modus: simuliertes Kapital (konservativ: jeder Trade -SL beim Verlust)
                aktuell_equity = paper_kapital

            if kill_switch_pruefen(
                symbol, start_equity, aktuell_equity, kill_switch_dd, paper_trading
            ):
                # Kill-Switch ausgelöst: Positionen schließen und stoppen
                if not paper_trading:
                    alle_positionen_schliessen(symbol)
                break  # Trading-Schleife beenden

            # ==============================================================
            # DATEN LADEN – Unterschied je nach Modus:
            #   Two-Stage (M5-Takt): HTF gecached + LTF frisch pro M5-Kerze
            #   Single-Stage:        H1-Daten wie bisher
            # ==============================================================

            if ts_aktiv:
                # ── Two-Stage M5-Takt ──────────────────────────────────
                # A) HTF-Bias (H1) nur aktualisieren wenn neue H1-Kerze da
                neue_h1 = neue_kerze_abwarten(symbol, letzte_htf_kerzen_zeit, "H1")
                if neue_h1 or cached_h1_df_clean is None:
                    htf_df_raw = mt5_daten_holen(symbol, timeframe="H1")
                    if htf_df_raw is not None and len(htf_df_raw) >= 250:
                        htf_df = features_berechnen(htf_df_raw, timeframe="H1")
                        htf_df = externe_features_einfuegen(
                            htf_df, external_features=externe_features
                        )
                        cached_h1_df_clean = htf_df.dropna(subset=FEATURE_SPALTEN)
                        two_stage_config["htf_df"] = cached_h1_df_clean
                        letzte_htf_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, "H1")
                        logger.info(
                            f"[{symbol}] 📊 HTF-Bias aktualisiert (neue H1-Kerze) | "
                            f"HTF-Bars={len(cached_h1_df_clean)}"
                        )
                    else:
                        logger.warning(
                            f"[{symbol}] HTF-H1-Daten unzureichend ({len(htf_df_raw) if htf_df_raw is not None else 0} Bars)"
                        )
                        if cached_h1_df_clean is None:
                            # Erster Lauf und H1 nicht verfügbar → überspringen
                            letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(
                                symbol, effektiver_tf
                            )
                            time.sleep(30)
                            continue

                # B) LTF-Daten (M5) frisch laden bei jeder M5-Kerze
                ltf_tf = two_stage_config["ltf_timeframe"]
                ltf_df_raw = mt5_daten_holen(symbol, timeframe=ltf_tf)
                ltf_min_bars = (
                    250
                    * TIMEFRAME_CONFIG.get(ltf_tf, {"bars_per_hour": 1})[
                        "bars_per_hour"
                    ]
                )
                if ltf_df_raw is None or len(ltf_df_raw) < ltf_min_bars:
                    logger.warning(
                        f"[{symbol}] LTF-{ltf_tf}-Daten unzureichend "
                        f"({len(ltf_df_raw) if ltf_df_raw is not None else 0}) – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    time.sleep(30)
                    continue

                ltf_df = features_berechnen(ltf_df_raw, timeframe=ltf_tf)
                ltf_df = externe_features_einfuegen(
                    ltf_df, external_features=externe_features
                )
                ltf_df_clean = ltf_df.dropna(subset=FEATURE_SPALTEN)
                two_stage_config["ltf_df"] = ltf_df_clean

                if len(ltf_df_clean) < 10:
                    logger.warning(
                        f"[{symbol}] LTF nach NaN-Bereinigung zu wenig Zeilen – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    continue

                # df_clean = gecachte H1-Daten (wird von shadow_signal_generieren
                # für den Single-Stage Baseline-Vergleich benötigt)
                df_clean = cached_h1_df_clean

            else:
                # ── Single-Stage (H1, unverändert) ─────────────────────
                df = mt5_daten_holen(symbol, timeframe=timeframe)
                min_bars = (
                    250
                    * TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
                        "bars_per_hour"
                    ]
                )
                if df is None or len(df) < min_bars:
                    logger.warning(
                        f"[{symbol}] Zu wenige Daten ({len(df) if df is not None else 0}) – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    time.sleep(30)
                    continue

                df = features_berechnen(df, timeframe=timeframe)
                df = externe_features_einfuegen(df, external_features=externe_features)

                # NaN-Zeilen am Anfang (Warm-Up) entfernen
                if hasattr(modell, "feature_name_") and modell.feature_name_:
                    dropna_features = [
                        f for f in modell.feature_name_ if f in df.columns
                    ]
                else:
                    dropna_features = [f for f in FEATURE_SPALTEN if f in df.columns]
                df_clean = df.dropna(subset=dropna_features)
                if len(df_clean) < 10:
                    logger.warning(
                        f"[{symbol}] Nach NaN-Bereinigung zu wenige Zeilen – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    continue

            # ---- Schritt 4: Signal generieren (Shadow-Mode) ----
            signal, prob, regime, atr_pct = shadow_signal_generieren(
                symbol=symbol,
                df=df_clean,
                modell=modell,
                schwelle=schwelle,
                short_schwelle=short_schwelle,
                decision_mapping=decision_mapping,
                regime_spalte=regime_spalte,
                two_stage_kongruenz=two_stage_kongruenz,
                regime_erlaubt=regime_erlaubt,
                two_stage_config=two_stage_config,
            )
            regime_name = REGIME_NAMEN.get(regime, "?")

            # ATR-basiertes Stop-Loss berechnen (dynamisch!)
            if atr_sl_aktiv and atr_pct > 0:
                sl_aktuell = atr_pct * atr_sl_faktor  # z.B. 0.0021 × 1.5 = 0.0032
                tp_aktuell = sl_aktuell  # symmetrisches TP/SL (RRR 1:1)
                sl_info = f"ATR-SL={sl_aktuell:.2%} ({ATR_SL_FAKTOR}×ATR)"
            else:
                sl_aktuell = SL_PCT  # Fallback: festes SL
                tp_aktuell = TP_PCT
                sl_info = f"Fix-SL={sl_aktuell:.1%}"

            logger.info(
                f"[{symbol}] Signal={signal} | Prob={prob:.1%} | "
                f"Regime={regime} ({regime_name}) | {sl_info}"
            )

            # Signal/Heartbeat in CSV loggen
            # Aktuellen Close-Preis und SL/TP-Niveaus berechnen für Dashboard
            close_preis = (
                float(df_clean["close"].iloc[-1]) if len(df_clean) > 0 else 0.0
            )
            if signal != 0 and close_preis > 0:
                if signal == 2:  # Long
                    log_sl = round(close_preis * (1.0 - sl_aktuell), 5)
                    log_tp = round(close_preis * (1.0 + tp_aktuell), 5)
                else:  # Short
                    log_sl = round(close_preis * (1.0 + sl_aktuell), 5)
                    log_tp = round(close_preis * (1.0 - tp_aktuell), 5)
            else:
                log_sl = 0.0
                log_tp = 0.0

            if signal != 0:
                n_signale += 1
            if heartbeat_log or signal != 0:
                trade_loggen(
                    symbol,
                    signal,
                    prob,
                    regime,
                    paper_trading,
                    entry_price=close_preis,
                    sl_price=log_sl,
                    tp_price=log_tp,
                )

            # ---- Schritt 5: Trade ausführen ----
            if signal != 0:
                # Offene Position prüfen (nur 1 Trade gleichzeitig!)
                if mt5_offene_position(symbol):
                    logger.info(
                        f"[{symbol}] Bereits offene Position – kein neuer Trade"
                    )
                else:
                    erfolg = order_senden(
                        symbol, signal, lot, tp_aktuell, sl_aktuell, paper_trading
                    )
                    if erfolg:
                        n_trades += 1
                        richtung_str = "Long" if signal == 2 else "Short"
                        logger.info(
                            f"[{symbol}] Trade #{n_trades}: {richtung_str} | "
                            f"Prob={prob:.1%} | Regime={regime_name}"
                        )

                        # Paper-Modus: simulierten Kontostand aktualisieren
                        # Konservative Schätzung: 50% TP, 50% SL (basierend auf ~40% Win-Rate)
                        # Jeder Trade kostet im Schnitt: 0.5*TP - 0.5*SL = 0 bei 1:1 RRR
                        # Aber nach Kosten (Spread ~0.01%): leicht negativ
                        # → Für Kill-Switch: pauschale 0.1% Verlust pro Trade (konservativ)
                        if paper_trading:
                            paper_kapital -= (
                                paper_kapital * 0.001
                            )  # 0.1% konservative Schätzung
            else:
                # Keine irreführende "< Schwelle" Meldung – Grund steht bereits im Detail-Log
                logger.info(f"[{symbol}] Kein Trade-Signal (Details siehe oben)")

            # Zeitstempel der verarbeiteten Kerze speichern (effektiver Zeitrahmen)
            letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)

            # Statistik alle 100 Kerzen (bei M5 ≈ alle 8 Stunden, bei H1 ≈ alle 4 Tage)
            stat_intervall = 100 if ts_aktiv else 24
            if n_trades > 0 and n_trades % stat_intervall == 0:
                dd_aktuell = (
                    (start_equity - paper_kapital) / start_equity
                    if paper_trading
                    else 0.0
                )
                logger.info(
                    f"[{symbol}] Status: {n_trades} Trades | {n_signale} Signale | "
                    f"Modus: {'Paper' if paper_trading else 'LIVE'} | "
                    f"Sim-Kapital: {paper_kapital:,.2f} | DD: {dd_aktuell:.1%}"
                )

        except KeyboardInterrupt:
            logger.info(f"\n[{symbol}] Trader gestoppt (Ctrl+C)")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"[{symbol}] Fehler in Haupt-Schleife: {e}", exc_info=True)
            logger.info("Warte 60 Sekunden vor Neustart ...")
            time.sleep(60)  # Kurze Pause bei Fehlern, dann weiter


# ============================================================
# 9. Hauptprogramm
# ============================================================


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches
    """Startet den Live-Trader für ein Symbol."""

    parser = argparse.ArgumentParser(
        description=(
            "MT5 ML-Trading – Live-Trader (Phase 7)\n"
            "Läuft auf: Windows 11 Laptop mit MT5-Terminal\n\n"
            "Aktuelle Konfiguration (Option 1 Test-Phase):\n"
            "  USDCAD: --symbol USDCAD --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1\n"
            "  USDJPY: --symbol USDJPY --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="USDCAD",
        choices=SYMBOLE,
        help=(
            "Handelssymbol (Standard: USDCAD). "
            "Aktive Betriebs-Symbole laut Policy: USDCAD, USDJPY"
        ),
    )
    parser.add_argument(
        "--schwelle",
        type=float,
        default=0.55,
        help=(
            "Long-Schwelle für Trade-Ausführung (Standard: 0.55). "
            "Kongruenz-Filter (HTF+LTF müssen übereinstimmen) bietet zusätzliche Absicherung."
        ),
    )
    parser.add_argument(
        "--short_schwelle",
        type=float,
        default=-1.0,
        help=(
            "Optionale Short-Schwelle. "
            "Bei --decision_mapping class: proba_short >= short_schwelle. "
            "Bei --decision_mapping long_prob: proba_long <= short_schwelle. "
            "Standard: -1 (auto)."
        ),
    )
    parser.add_argument(
        "--decision_mapping",
        type=str,
        choices=["class", "long_prob"],
        default="class",
        help=(
            "Signal-Mapping: class (klassische Klassen-Proba) oder long_prob "
            "(Long bei >= long_schwelle, Short bei <= short_schwelle)."
        ),
    )
    parser.add_argument(
        "--regime_source",
        type=str,
        choices=["market_regime", "market_regime_hmm"],
        default="market_regime",
        help=(
            "Quelle für Regime-Filter. "
            "market_regime_hmm ist oft reaktiver (mehr Regime-Wechsel)."
        ),
    )
    parser.add_argument(
        "--two_stage_kongruenz",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "1 = HTF/LTF Kongruenzfilter aktiv (sicherer, weniger Trades), "
            "0 = Kongruenzfilter aus (aggressiver, mehr Trades)."
        ),
    )
    parser.add_argument(
        "--regime_filter",
        type=str,
        default="0,1,2",
        help=(
            "Komma-getrennte Regime-Nummern (Standard: '0,1,2'). "
            "0=Seitwärts, 1=Aufwärtstrend, 2=Abwärtstrend, 3=Hohe Vola. "
            "Option 1 (Test-Phase): Alle Regime erlaubt für mehr Feedback"
        ),
    )
    parser.add_argument(
        "--lot",
        type=float,
        default=LOT,
        help=f"Lot-Größe pro Trade (Standard: {LOT} = Micro-Lot). NICHT erhöhen ohne Erfahrung!",
    )
    parser.add_argument(
        "--paper_trading",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = Paper-Modus (Standard, empfohlen!), "
            "0 = Live-Modus (erst nach 2 Wochen Testlauf!)"
        ),
    )
    parser.add_argument(
        "--mt5_server",
        default="",
        help="MT5 Broker-Server (z.B. 'ICMarkets-Demo')",
    )
    parser.add_argument(
        "--mt5_login",
        type=int,
        default=0,
        help="MT5 Kontonummer",
    )
    parser.add_argument(
        "--mt5_password",
        default="",
        help="MT5 Passwort",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Modell-Versions-Suffix (Standard: v4). Muss mit train_model.py übereinstimmen.",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "M30", "M15"],
        help=(
            "Zeitrahmen für Datenabruf und Modell (Standard: H1). "
            "M30/M15 erfordern separat trainierte Modelle: "
            "lgbm_SYMBOL_TIMEFRAME_VERSION.pkl"
        ),
    )
    parser.add_argument(
        "--allow_research_symbol",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 = Policy erzwingen (nur USDCAD/USDJPY handelbar, Standard), "
            "1 = Forschungs-Override fuer andere Symbole (nur Paper-Modus empfohlen)."
        ),
    )
    parser.add_argument(
        "--kill_switch_dd",
        type=float,
        default=KILL_SWITCH_DD_DEFAULT,
        help=(
            f"Kill-Switch: Maximaler Drawdown in Dezimal (Standard: {KILL_SWITCH_DD_DEFAULT} = 15%%). "
            "Wenn Verlust diesen Wert überschreitet, stoppt der Trader automatisch. "
            "Empfehlung: 0.15 (15%%). Nie über 0.20 (20%%) setzen!"
        ),
    )
    parser.add_argument(
        "--kapital_start",
        type=float,
        default=10000.0,
        help=(
            "Startkapital für Kill-Switch-Berechnung im Paper-Modus (Standard: 10000.0 EUR). "
            "Im Live-Modus wird das echte MT5-Kontokapital automatisch ausgelesen."
        ),
    )
    parser.add_argument(
        "--heartbeat_log",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = pro neuer Kerze Heartbeat in CSV schreiben (Standard), "
            "0 = nur bei Signal/Trade schreiben."
        ),
    )
    parser.add_argument(
        "--atr_sl",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = ATR-basiertes Stop-Loss (dynamisch, Standard), "
            "0 = festes Stop-Loss (0.3%%). "
            "ATR-SL verbessert Sharpe um +1.5 bis +2.0 im Backtest."
        ),
    )
    parser.add_argument(
        "--atr_faktor",
        type=float,
        default=1.5,
        help=(
            "ATR-Multiplikator für SL-Berechnung (Standard: 1.5). "
            "SL = ATR_14 × Faktor. Empfehlung aus Backtest: 1.5"
        ),
    )
    parser.add_argument(
        "--two_stage_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 = Single-Stage (Standard), "
            "1 = Shadow-Mode für Two-Stage (USDCAD/USDJPY, v4-Modelle erforderlich). "
            "Shadow-Mode: beide Signale werden geloggt, Two-Stage wird verwendet."
        ),
    )
    parser.add_argument(
        "--two_stage_ltf_timeframe",
        default="M5",
        choices=["M5", "M15"],
        help=(
            "LTF-Zeitrahmen für Two-Stage (Standard: M5). "
            "HTF ist immer H1. Nur relevant wenn --two_stage_enable 1"
        ),
    )
    parser.add_argument(
        "--two_stage_version",
        default="v4",
        help=(
            "Modellversion für Two-Stage-Modelle (Standard: v4). "
            "Erwartet: lgbm_htf_bias_SYMBOL_H1_VERSION.pkl und "
            "lgbm_ltf_entry_SYMBOL_LTF-TF_VERSION.pkl"
        ),
    )
    args = parser.parse_args()

    # ---- ATR-SL Konfiguration aus CLI ----
    atr_sl_aktiv = bool(args.atr_sl)
    atr_sl_faktor = args.atr_faktor
    short_schwelle: Optional[float] = (
        None if float(args.short_schwelle) < 0.0 else float(args.short_schwelle)
    )
    two_stage_kongruenz = bool(args.two_stage_kongruenz)

    # ---- Regime-Filter parsen ----
    regime_erlaubt = None
    if args.regime_filter:
        try:
            regime_erlaubt = [int(r.strip()) for r in args.regime_filter.split(",")]
        except ValueError:
            print(
                f"Ungültiger --regime_filter: '{args.regime_filter}'. Erwartet: z.B. '1,2'"
            )
            return

    # ---- Paper-Trading sicher stellen ----
    paper_trading = bool(args.paper_trading)
    if not paper_trading:
        print("\n" + "⚠️ " * 20)
        print("WARNUNG: Live-Trading Modus aktiv!")
        print("  → Echte Orders mit echtem Geld!")
        print("  → Nur aktivieren nach 2 Wochen erfolgreichen Paper-Tradings!")
        print("  → Weiter? (ja eingeben zum Bestätigen)")
        bestaetigung = input("Bestätigung: ").strip().lower()
        if bestaetigung != "ja":
            print("Abgebrochen. Starte mit --paper_trading 1")
            return
        print("⚠️ " * 20 + "\n")

    # ---- Symbol-Policy prüfen (nur 2 aktive Paare) ----
    symbol = args.symbol.upper()
    if symbol not in AKTIVE_SYMBOLE and not bool(args.allow_research_symbol):
        print(
            "\nPolicy-Block: Dieses Symbol ist aktuell Research-only.\n"
            f"Aktive Betriebs-Symbole: {', '.join(AKTIVE_SYMBOLE)}\n"
            "Wenn du bewusst testen willst, setze --allow_research_symbol 1 "
            "(empfohlen nur mit --paper_trading 1)."
        )
        return

    # Forschungs-Override nur im Paper-Modus zulassen
    if (
        symbol not in AKTIVE_SYMBOLE
        and bool(args.allow_research_symbol)
        and not paper_trading
    ):
        print(
            "Research-Override darf nicht im Live-Modus laufen. "
            "Bitte --paper_trading 1 verwenden."
        )
        return

    # ---- Modell laden ----
    timeframe = args.timeframe.upper()
    if timeframe == "H1":
        modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{args.version}.pkl"
    else:
        modell_pfad = (
            MODEL_DIR / f"lgbm_{symbol.lower()}_{timeframe}_{args.version}.pkl"
        )
    if not modell_pfad.exists():
        modell_name_hinweis = (
            f"lgbm_{symbol.lower()}_{args.version}.pkl"
            if timeframe == "H1"
            else f"lgbm_{symbol.lower()}_{timeframe}_{args.version}.pkl"
        )
        print(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Bitte Modell vom Linux-Server übertragen:\n"
            f"  scp SERVER:/mnt/1T-Data/XGBoost-LightGBM/models/"
            f"{modell_name_hinweis} ./models/"
        )
        return

    logger.info(f"Lade Modell: {modell_pfad.name}")
    modell = joblib.load(modell_pfad)
    logger.info("Modell geladen ✓")

    # ---- MT5 verbinden (wenn Zugangsdaten vorhanden) ----
    if args.mt5_server and args.mt5_login and args.mt5_password:
        verbunden = mt5_verbinden(args.mt5_server, args.mt5_login, args.mt5_password)
        if not verbunden:
            logger.warning(
                "MT5-Verbindung fehlgeschlagen – starte im simulierten Modus.\n"
                "Stelle sicher dass MT5 Terminal geöffnet und eingeloggt ist!"
            )
            if not paper_trading:
                print("MT5-Verbindung für Live-Trading erforderlich! Abgebrochen.")
                return
    else:
        logger.info(
            "Keine MT5-Zugangsdaten angegeben – Paper-Trading ohne MT5-Verbindung.\n"
            "Für echte Verbindung: --mt5_server SERVER --mt5_login NUMMER --mt5_password PW"
        )

    # ---- MT5-Check für Live-Modus ----
    if not paper_trading and not MT5_VERFUEGBAR:
        print("MT5 nicht installiert! Live-Trading nicht möglich.")
        return

    # ---- Kill-Switch-Parameter validieren ----
    if args.kill_switch_dd > 0.20:
        logger.warning(
            f"⚠️  Kill-Switch-Limit {args.kill_switch_dd:.0%} ist sehr hoch! "
            "Empfehlung: max. 20% (0.20). Bitte überprüfen."
        )

    # ---- Two-Stage-Konfiguration vorbereiten (Shadow-Mode) ----
    two_stage_config = None
    TWO_STAGE_APPROVED = {"USDCAD", "USDJPY"}
    if bool(args.two_stage_enable):
        if symbol.upper() not in TWO_STAGE_APPROVED:
            logger.info(
                f"[{symbol}] Two-Stage-Shadow-Mode nur für {TWO_STAGE_APPROVED} freigeschaltet. "
                "Verwende Single-Stage."
            )
        else:
            logger.info(
                f"[{symbol}] Two-Stage-Shadow-Mode aktiv: HTF=H1, LTF={args.two_stage_ltf_timeframe}, "
                f"Version={args.two_stage_version}"
            )
            # Feature-Listen aus JSON-Metadatei laden (exakt wie beim Training)
            import json

            meta_pfad = (
                MODEL_DIR
                / f"two_stage_{symbol.lower()}_{args.two_stage_ltf_timeframe}_{args.two_stage_version}.json"
            )
            if meta_pfad.exists():
                with open(meta_pfad, "r") as f:
                    meta = json.load(f)
                htf_feats = meta["htf_features"]
                ltf_feats = meta["ltf_features"]
                logger.info(
                    f"[{symbol}] Metadaten geladen: HTF={len(htf_feats)} Features, "
                    f"LTF={len(ltf_feats)} Features"
                )
            else:
                logger.warning(
                    f"[{symbol}] Metadatei {meta_pfad.name} nicht gefunden – "
                    "verwende Standard-Feature-Liste"
                )
                htf_feats = FEATURE_SPALTEN
                ltf_feats = FEATURE_SPALTEN

            two_stage_config = {
                "enable": True,
                "ltf_timeframe": args.two_stage_ltf_timeframe,
                "version": args.two_stage_version,
                "htf_features": htf_feats,
                "ltf_features": ltf_feats,
                # HTF/LTF DataFrames werden in der trading_loop dynamisch geladen
                "htf_df": None,
                "ltf_df": None,
            }

    # ---- Hauptschleife starten ----
    trading_loop(
        symbol=symbol,
        schwelle=args.schwelle,
        short_schwelle=short_schwelle,
        decision_mapping=args.decision_mapping,
        regime_spalte=args.regime_source,
        two_stage_kongruenz=two_stage_kongruenz,
        regime_erlaubt=regime_erlaubt,
        paper_trading=paper_trading,
        lot=args.lot,
        modell=modell,
        kill_switch_dd=args.kill_switch_dd,
        kapital_start=args.kapital_start,
        heartbeat_log=bool(args.heartbeat_log),
        timeframe=timeframe,
        atr_sl_aktiv=atr_sl_aktiv,
        atr_sl_faktor=atr_sl_faktor,
        two_stage_config=two_stage_config,
    )

    # MT5 Verbindung beenden
    if MT5_VERFUEGBAR and mt5.terminal_info() is not None:
        mt5.shutdown()
        logger.info("MT5-Verbindung getrennt.")


if __name__ == "__main__":
    main()
