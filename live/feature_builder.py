"""
feature_builder.py – Feature-Berechnung (identisch mit der Trainings-Pipeline)

Berechnet alle 45+ Model-Features aus OHLCV-Rohdaten.
Die Feature-Namen und Formeln sind IDENTISCH mit feature_engineering.py auf dem Linux-Server.

Läuft auf: Windows 11 Laptop
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

try:
    from live import config  # Import als Paket (z.B. pytest, externe Aufrufe)
    from live.indicators import (
        _series_sign,
        ind_adx,
        ind_atr,
        ind_bbands,
        ind_ema,
        ind_macd,
        ind_obv,
        ind_roc,
        ind_rsi,
        ind_sma,
        ind_stoch,
        ind_williams_r,
    )
except ImportError:
    import config  # Import direkt aus live/-Verzeichnis
    from indicators import (
        _series_sign,
        ind_adx,
        ind_atr,
        ind_bbands,
        ind_ema,
        ind_macd,
        ind_obv,
        ind_roc,
        ind_rsi,
        ind_sma,
        ind_stoch,
        ind_williams_r,
    )

logger = logging.getLogger(__name__)


def features_berechnen(
    df: pd.DataFrame,
    timeframe: str = "H1",
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    Berechnet alle 45 Model-Features aus OHLCV-Rohdaten.

    Die Feature-Namen und Formeln sind IDENTISCH mit feature_engineering.py.
    Das ist kritisch: das Modell muss dieselben Werte wie beim Training sehen.

    Args:
        df: OHLCV DataFrame mit Spalten open, high, low, close, volume
            und DatetimeIndex (UTC)
        timeframe: Aktiver Zeitrahmen (H1, M30, M15, M5)

    Returns:
        DataFrame mit allen 45 Features (ohne NaN-Zeilen am Anfang)
    """
    result = df.copy()

    # Bars pro Stunde für zeitäquivalente Fenster bestimmen
    bars_per_hour = config.TIMEFRAME_CONFIG.get(
        timeframe, config.TIMEFRAME_CONFIG["H1"]
    )["bars_per_hour"]

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
    result["sma_20_50_cross"] = _series_sign(
        result["sma_20"] - result["sma_50"]
    ).fillna(0)
    result["sma_50_200_cross"] = _series_sign(
        result["sma_50"] - result["sma_200"]
    ).fillna(0)

    result["ema_12"] = ind_ema(result["close"], 12)
    result["ema_26"] = ind_ema(result["close"], 26)
    result["ema_cross"] = _series_sign(result["ema_12"] - result["ema_26"]).fillna(0)

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
    result["stoch_cross"] = _series_sign(result["stoch_k"] - result["stoch_d"]).fillna(
        0
    )

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

    log_ret = pd.Series(
        np.log(result["close"] / result["close"].shift(1)),
        index=result.index,
        dtype=float,
    )
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
    result["candle_dir"] = _series_sign(result["close"] - result["open"]).fillna(0)
    result["hl_range"] = (result["high"] - result["low"]) / result["close"]

    # --- Multi-Timeframe-Features (H4 + D1) ---
    # LOOK-AHEAD-BIAS: .shift(1) verhindert zukünftige Information
    close = result["close"]
    close_h4 = close.resample("4h").last().dropna()
    trend_h4 = _series_sign(ind_sma(close_h4, 20) - ind_sma(close_h4, 50)).fillna(0)
    result["trend_h4"] = trend_h4.shift(1).reindex(result.index, method="ffill")
    result["rsi_h4"] = (
        ind_rsi(close_h4, 14).shift(1).reindex(result.index, method="ffill")
    )

    close_d1 = close.resample("1D").last().dropna()
    trend_d1 = _series_sign(ind_sma(close_d1, 20) - ind_sma(close_d1, 50)).fillna(0)
    result["trend_d1"] = trend_d1.shift(1).reindex(result.index, method="ffill")

    # --- Zeitbasierte Features ---
    zeit_index = cast(pd.DatetimeIndex, result.index)
    result["hour"] = zeit_index.hour
    result["day_of_week"] = zeit_index.dayofweek
    h = result["hour"]
    result["session_london"] = ((h >= 8) & (h < 17)).astype(int)
    result["session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    result["session_asia"] = ((h >= 0) & (h < 9)).astype(int)
    result["session_overlap"] = ((h >= 13) & (h < 17)).astype(int)

    # --- Kill-Zone Features ---
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
        bull_fvg, bull_gap, np.where(bear_fvg, -bear_gap, 0.0)
    )

    # --- MSS/BOS (Marktstruktur) ---
    pivot_bars = 20
    prev_swing_high = result["high"].shift(1).rolling(pivot_bars).max()
    prev_swing_low = result["low"].shift(1).rolling(pivot_bars).min()
    result["bos_bull"] = (result["close"] > prev_swing_high).astype(int)
    result["bos_bear"] = (result["close"] < prev_swing_low).astype(int)

    structure_bias = np.where(
        result["bos_bull"] == 1, 1, np.where(result["bos_bear"] == 1, -1, 0)
    )
    result["structure_bias"] = (
        pd.Series(structure_bias, index=result.index).ffill().fillna(0)
    )
    prev_bias = result["structure_bias"].shift(1).fillna(0)
    result["mss_bull"] = ((result["bos_bull"] == 1) & (prev_bias < 0)).astype(int)
    result["mss_bear"] = ((result["bos_bear"] == 1) & (prev_bias > 0)).astype(int)

    # --- ADX + Regime-Detection ---
    result["adx_14"] = ind_adx(result)

    atr_pct = result["atr_pct"]
    median_atr = atr_pct.rolling(window=50, min_periods=50).median()
    adx = result["adx_14"]

    regime = pd.Series(0, index=result.index, dtype=int)
    hoch_vol = atr_pct > (config.REGIME_HIGH_VOL_FAKTOR * median_atr)
    aufwaerts = (
        (adx > config.REGIME_ADX_TREND_SCHWELLE)
        & (result["close"] > result["sma_50"])
        & ~hoch_vol
    )
    abwaerts = (
        (adx > config.REGIME_ADX_TREND_SCHWELLE)
        & (result["close"] < result["sma_50"])
        & ~hoch_vol
    )
    regime[aufwaerts] = 1
    regime[abwaerts] = 2
    regime[hoch_vol] = 3
    result["market_regime"] = regime

    # --- HMM-Regime (optional, mit robustem Fallback) ---
    if config.HAS_HMMLEARN and config.GaussianHMM is not None:
        try:
            hmm_input = np.column_stack(
                [
                    log_ret.fillna(0.0).to_numpy(dtype=float),
                    atr_pct.ffill().fillna(0.0).to_numpy(dtype=float),
                ]
            )

            hmm_regimes = np.full(len(result), np.nan)
            min_train_bars = min(400, max(120, len(result) // 3))
            refit_interval = 120
            hmm_model = None

            for i in range(min_train_bars, len(result)):
                if hmm_model is None or (i - min_train_bars) % refit_interval == 0:
                    hmm_model = config.GaussianHMM(
                        n_components=4,
                        covariance_type="diag",
                        n_iter=120,
                        random_state=42,
                    )
                    with config.mute_hmmlearn_convergence_logs():
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
