"""
indicators.py – Technische Indikatoren (identisch mit feature_engineering.py)

Die Implementierungen sind IDENTISCH mit dem Server-seitigen feature_engineering.py.
Das ist kritisch: Das Modell muss dieselben Feature-Werte sehen wie beim Training.

Läuft auf: Windows 11 Laptop
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from live import config  # Import als Paket (z.B. pytest, externe Aufrufe)
except ImportError:
    import config  # Import direkt aus live/-Verzeichnis

logger = logging.getLogger(__name__)


def _series_sign(values: pd.Series) -> pd.Series:
    """np.sign als echte Pandas-Serie zurückgeben, damit fillna/rolling typstabil bleiben."""
    return pd.Series(np.sign(values), index=values.index, dtype=float)


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
    macd_signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal_line,
        "macd_hist": macd_line - macd_signal_line,
    })


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
    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_k.rolling(window=d, min_periods=d).mean(),
    })


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
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def ind_bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: bb_upper, bb_mid, bb_lower."""
    bb_mid = series.rolling(window=length, min_periods=length).mean()
    bb_std = series.rolling(window=length, min_periods=length).std()
    return pd.DataFrame({
        "bb_upper": bb_mid + std * bb_std,
        "bb_mid": bb_mid,
        "bb_lower": bb_mid - std * bb_std,
    })


def ind_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    signale = _series_sign(close.diff())
    return (signale * volume).fillna(0).cumsum()


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
    if config.HAS_PANDAS_TA:
        adx_df = df.ta.adx(length=length)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
        return adx_df[adx_col]

    # Fallback: Manuelle Wilder-Implementierung
    alpha = 1.0 / length
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
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
    plus_di = (plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_s) * 100
    minus_di = (minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_s) * 100
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
