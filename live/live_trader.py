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
    6. Regime-Filter anwenden (z.B. nur im Aufwärtstrend handeln)
    7. Order senden (Paper: nur loggen / Live: echte MT5-Order)

Modell-Übertragung vom Linux-Server auf den Windows Laptop:
    Auf dem Linux-Server ausführen:
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_usdcad_v1.pkl USER@LAPTOP:./models/
        scp /mnt/1T-Data/XGBoost-LightGBM/models/lgbm_usdjpy_v1.pkl USER@LAPTOP:./models/

Verwendung (Windows, venv aktiviert):
    python live_trader.py --symbol USDCAD --schwelle 0.60 --regime_filter 1,2
    python live_trader.py --symbol USDJPY --schwelle 0.60 --regime_filter 1
    python live_trader.py --help

Voraussetzungen:
    pip install MetaTrader5 pandas numpy pandas_ta joblib requests python-dotenv
"""

# pylint: disable=too-many-lines,logging-fstring-interpolation

# Standard-Bibliotheken
import argparse
import logging
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
        logging.FileHandler(
            LOG_DIR / "live_trader.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Trading-Parameter (müssen mit backtest.py übereinstimmen!)
# ============================================================

# Risikomanagement
TP_PCT = 0.003        # Take-Profit: 0.3% (identisch mit Labeling)
SL_PCT = 0.003        # Stop-Loss:   0.3% (identisch mit Labeling)
LOT = 0.01            # Minimale Lot-Größe (0.01 = Micro-Lot, ~1€/Pip)
MAX_OFFENE_TRADES = 1  # Maximal 1 offene Position pro Symbol
MAGIC_NUMBER = 20260101  # Eindeutige Kennung für ML-Trades in MT5

# Kill-Switch – Harter Stopp bei zu hohem Drawdown (Review-Punkt 8)
KILL_SWITCH_DD_DEFAULT = 0.15  # Harter Stopp bei 15% Drawdown (Standard)

# Feature-Berechnung: Mindest-Barren für Warm-Up
N_BARREN = 500  # SMA200 braucht 200, MTF braucht mehr → 500 als Buffer

# Verfügbare Symbole
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

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
    "open", "high", "low", "close", "volume", "spread",
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "atr_14", "bb_upper", "bb_mid", "bb_lower", "obv",
    "label",
}

# Exakte Feature-Reihenfolge (identisch mit Trainings-CSV)
# Das Modell erwartet genau diese 45 Spalten
FEATURE_SPALTEN = [
    "price_sma20_ratio", "price_sma50_ratio", "price_sma200_ratio",
    "sma_20_50_cross", "sma_50_200_cross",
    "ema_cross",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14", "rsi_centered",
    "stoch_k", "stoch_d", "stoch_cross",
    "williams_r", "roc_10",
    "atr_pct",
    "bb_width", "bb_pct",
    "hist_vol_20",
    "obv_zscore", "volume_roc", "volume_ratio",
    "return_1h", "return_4h", "return_24h",
    "candle_body", "upper_wick", "lower_wick", "candle_dir", "hl_range",
    "trend_h4", "rsi_h4", "trend_d1",
    "hour", "day_of_week",
    "session_london", "session_ny", "session_asia", "session_overlap",
    "adx_14", "market_regime",
    "fear_greed_value", "fear_greed_class", "btc_funding_rate",
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
    high: pd.Series, low: pd.Series, close: pd.Series,
    k: int = 14, d: int = 3
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
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()


def ind_bbands(
    series: pd.Series, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
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
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

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
        plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        / atr_s
    ) * 100
    minus_di = (
        minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        / atr_s
    ) * 100
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    return dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


# ============================================================
# 2. Feature-Berechnung (identisch mit Trainings-Pipeline)
# ============================================================


def features_berechnen(df: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=too-many-locals
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

    # --- Trend-Features ---
    result["sma_20"] = ind_sma(result["close"], 20)
    result["sma_50"] = ind_sma(result["close"], 50)
    result["sma_200"] = ind_sma(result["close"], 200)

    result["price_sma20_ratio"] = (result["close"] - result["sma_20"]) / result["sma_20"]
    result["price_sma50_ratio"] = (result["close"] - result["sma_50"]) / result["sma_50"]
    result["price_sma200_ratio"] = (result["close"] - result["sma_200"]) / result["sma_200"]
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

    result["williams_r"] = ind_williams_r(result["high"], result["low"], result["close"])
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
    result["hist_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252 * 24)

    # --- Volumen-Features ---
    result["obv"] = ind_obv(result["close"], result["volume"])
    obv_mean = result["obv"].rolling(50).mean()
    obv_std = result["obv"].rolling(50).std().replace(0, np.nan)
    result["obv_zscore"] = (result["obv"] - obv_mean) / obv_std
    result["volume_roc"] = ind_roc(result["volume"], 14)
    vol_sma = result["volume"].rolling(20).mean().replace(0, np.nan)
    result["volume_ratio"] = result["volume"] / vol_sma

    # --- Kerzenmuster-Features ---
    result["return_1h"] = np.log(result["close"] / result["close"].shift(1))
    result["return_4h"] = np.log(result["close"] / result["close"].shift(4))
    result["return_24h"] = np.log(result["close"] / result["close"].shift(24))

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
    result["rsi_h4"] = ind_rsi(close_h4, 14).shift(1).reindex(
        result.index, method="ffill"
    )

    close_d1 = close.resample("1d").last().dropna()
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


def externe_features_einfuegen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt Fear & Greed und BTC Funding Rate als Features ein.

    Alle Zeilen erhalten denselben aktuellen Wert (für die letzte Kerze
    relevant – ältere Zeilen werden für die Vorhersage eh nicht verwendet).

    Args:
        df: Feature-DataFrame (bereits mit technischen Indikatoren)

    Returns:
        DataFrame mit 3 zusätzlichen Spalten.
    """
    fg = fear_greed_holen()
    btc_rate = btc_funding_holen()

    df["fear_greed_value"] = fg["fear_greed_value"]
    df["fear_greed_class"] = fg["fear_greed_class"]
    df["btc_funding_rate"] = btc_rate
    return df


# ============================================================
# 4. Signal-Generierung (Modell-Vorhersage)
# ============================================================


def signal_generieren(
    df: pd.DataFrame,
    modell: object,
    schwelle: float = 0.60,
    regime_erlaubt: Optional[list] = None,
) -> Tuple[int, float, int]:
    """
    Generiert ein Trade-Signal für die letzte Kerze (die gerade geschlossen hat).

    Args:
        df:              Feature-DataFrame (alle 45 Features vorhanden)
        modell:          Geladenes LightGBM-Modell
        schwelle:        Mindest-Wahrscheinlichkeit für Trade-Ausführung
        regime_erlaubt:  Erlaubte Regime-Nummern (None = alle)

    Returns:
        Tuple (signal, prob, regime):
            signal: 2=Long, -1=Short, 0=Kein Trade
            prob:   Wahrscheinlichkeit des Signals (0–1)
            regime: Aktuelles Markt-Regime (0–3)
    """
    # Letzte vollständige Kerze (Index -1 = aktuelle Kerze, -2 = letzte geschlossene)
    # Wir verwenden die letzte vollständige Kerze für das Signal
    letzte_kerze = df.iloc[[-2]]  # -2: letzte geschlossene Kerze (sicher!)

    # Aktuelles Regime
    aktuelles_regime = int(letzte_kerze["market_regime"].iloc[0])

    # Regime-Filter prüfen
    if regime_erlaubt is not None:
        if aktuelles_regime not in regime_erlaubt:
            regime_name = REGIME_NAMEN.get(aktuelles_regime, "?")
            logger.info(
                f"Signal übersprungen: Regime '{regime_name}' nicht in "
                f"{[REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]}"
            )
            return 0, 0.0, aktuelles_regime

    # Features für Modell vorbereiten (NaN-Werte mit Median auffüllen)
    verfuegbare = [f for f in FEATURE_SPALTEN if f in df.columns]
    fehlende = [f for f in FEATURE_SPALTEN if f not in df.columns]
    if fehlende:
        logger.warning(f"Fehlende Features: {fehlende} – werden mit 0 gefüllt")
        for feat in fehlende:
            letzte_kerze[feat] = 0.0

    x_features = letzte_kerze[verfuegbare].copy()

    # NaN auffüllen (Sicherheitsnetz)
    if x_features.isna().any().any():
        logger.warning("NaN-Werte in Features – werden mit Median der letzten 50 Kerzen gefüllt")
        nan_fill = df[verfuegbare].iloc[-50:].median()
        x_features = x_features.fillna(nan_fill)

    # Modell-Vorhersage: Wahrscheinlichkeiten für alle 3 Klassen
    # proba[:,0] = Short (0→-1), proba[:,1] = Neutral, proba[:,2] = Long
    proba = modell.predict_proba(x_features)[0]
    raw_pred = int(np.argmax(proba))

    # Signal mit Schwellenwert-Filter
    if raw_pred == 2 and proba[2] >= schwelle:
        return 2, float(proba[2]), aktuelles_regime   # Long-Signal
    if raw_pred == 0 and proba[0] >= schwelle:
        return -1, float(proba[0]), aktuelles_regime  # Short-Signal

    return 0, float(max(proba)), aktuelles_regime  # Kein Trade


# ============================================================
# 5. MetaTrader 5 Funktionen
# ============================================================


def mt5_verbinden(
    server: str, login: int, password: str, pfad: str = ""
) -> bool:
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


def mt5_daten_holen(symbol: str, n_barren: int = N_BARREN) -> Optional[pd.DataFrame]:
    """
    Holt die letzten H1-Barren von MT5.

    Args:
        symbol:   Handelssymbol (z.B. "USDCAD")
        n_barren: Anzahl der H1-Barren (Standard: 500)

    Returns:
        OHLCV DataFrame mit UTC-DatetimeIndex oder None bei Fehler.
    """
    if not MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – keine Live-Daten!")
        return None

    # Barren von Position 0 (aktuelle Kerze) bis n-1
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_barren)
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


def mt5_letzte_kerze_uhrzeit(symbol: str) -> Optional[datetime]:
    """
    Gibt die Öffnungszeit der letzten H1-Kerze zurück.

    Args:
        symbol: Handelssymbol

    Returns:
        datetime (UTC) der letzten Kerzen-Eröffnung oder None.
    """
    if not MT5_VERFUEGBAR:
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 2)
    if rates is None:
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
        "sl": sl_preis,          # Stop-Loss ist PFLICHT!
        "tp": tp_preis,
        "deviation": 20,         # Max. Slippage in Punkte
        "magic": MAGIC_NUMBER,   # Eindeutige ID für diesen Bot
        "comment": "ML-Phase6",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(
            f"Order fehlgeschlagen: Code={result.retcode} | {result.comment}"
        )
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
) -> None:
    """
    Schreibt jeden Trade-Versuch in eine CSV-Datei.

    Args:
        symbol:        Handelssymbol
        richtung:      2=Long, -1=Short, 0=Kein Trade
        prob:          Signal-Wahrscheinlichkeit
        regime:        Markt-Regime (0–3)
        paper_trading: True = Paper-Modus aktiv
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
    }

    df_log = pd.DataFrame([eintrag])
    # CSV anhängen (header=False wenn Datei bereits existiert)
    df_log.to_csv(
        log_pfad,
        mode="a",
        header=not log_pfad.exists(),
        index=False,
    )


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
            f"[{symbol}] Drawdown: {drawdown_pct:.1%} "
            f"(Limit: {max_dd_pct:.1%})"
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
        if pos.type == mt5.ORDER_TYPE_BUY:   # type: ignore[union-attr]
            close_type = mt5.ORDER_TYPE_SELL  # type: ignore[union-attr]
            preis = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY   # type: ignore[union-attr]
            preis = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,   # type: ignore[union-attr]
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": preis,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Kill-Switch",
            "type_time": mt5.ORDER_TIME_GTC,   # type: ignore[union-attr]
            "type_filling": mt5.ORDER_FILLING_IOC,  # type: ignore[union-attr]
        }
        result = mt5.order_send(request)  # type: ignore[union-attr]
        if result.retcode == mt5.TRADE_RETCODE_DONE:   # type: ignore[union-attr]
            logger.info(f"[{symbol}] Position {pos.ticket} geschlossen ✓")
        else:
            logger.error(
                f"[{symbol}] Schließen fehlgeschlagen: "
                f"Code={result.retcode} | {result.comment}"
            )


# ============================================================
# 8. Haupt-Trading-Schleife
# ============================================================


def neue_kerze_abwarten(symbol: str, letzte_kerzen_zeit: Optional[datetime]) -> bool:
    """
    Prüft ob eine neue H1-Kerze geöffnet wurde.

    Args:
        symbol:            Handelssymbol
        letzte_kerzen_zeit: Zeitstempel der letzten verarbeiteten Kerze

    Returns:
        True wenn neue Kerze verfügbar.
    """
    aktuelle_kerze = mt5_letzte_kerze_uhrzeit(symbol)
    if aktuelle_kerze is None:
        return False
    return letzte_kerzen_zeit is None or aktuelle_kerze != letzte_kerzen_zeit


def trading_loop(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    symbol: str,
    schwelle: float,
    regime_erlaubt: Optional[list],
    paper_trading: bool,
    lot: float,
    modell: object,
    kill_switch_dd: float = KILL_SWITCH_DD_DEFAULT,
    kapital_start: float = 10000.0,
) -> None:
    """
    Haupt-Schleife: Läuft dauerhaft und wartet auf neue H1-Kerzen.

    Bei jeder neuen Kerze:
    1. MT5-Daten holen
    2. Features berechnen
    3. Signal generieren
    4. Kill-Switch prüfen (Drawdown-Limit!)
    5. Trade ausführen (falls Signal stark genug)

    Args:
        symbol:          Handelssymbol
        schwelle:        Wahrscheinlichkeits-Schwelle (z.B. 0.60)
        regime_erlaubt:  Erlaubte Regime oder None für alle
        paper_trading:   True = Paper-Modus (kein echtes Geld!)
        lot:             Lot-Größe
        modell:          Geladenes LightGBM-Modell
        kill_switch_dd:  Max. Drawdown bis zum automatischen Stopp (Standard: 0.15 = 15%)
        kapital_start:   Startkapital für Paper-Tracking und Kill-Switch-Berechnung
    """
    modus_str = "PAPER-TRADING" if paper_trading else "⚠️  LIVE-TRADING MIT ECHTEM GELD!"
    regime_str = (
        [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        if regime_erlaubt else "alle"
    )

    logger.info("=" * 65)
    logger.info(f"LIVE-TRADER GESTARTET – {symbol}")
    logger.info(f"Modus:          {modus_str}")
    logger.info(f"Schwelle:       {schwelle:.0%}")
    logger.info(f"Regime-Filter:  {regime_str}")
    logger.info(f"TP/SL:          {TP_PCT:.1%} / {SL_PCT:.1%} (RRR={TP_PCT/SL_PCT:.1f}:1)")
    logger.info(f"Lot-Größe:      {lot}")
    logger.info(f"Kill-Switch:    Drawdown > {kill_switch_dd:.0%} → automatischer Stopp")
    logger.info(f"Startkapital:   {kapital_start:,.2f} (für Kill-Switch-Berechnung)")
    logger.info(f"Logs:           {LOG_DIR}")
    logger.info("Warte auf neue H1-Kerze ...")
    logger.info("=" * 65)

    letzte_kerzen_zeit: Optional[datetime] = None
    n_signale = 0   # Gesamt-Signale
    n_trades = 0    # Ausgeführte Trades

    # ---- Kill-Switch: Startkapital ermitteln ----
    # Im Live-Modus: echtes Kontostand von MT5 lesen
    # Im Paper-Modus: übergebenes Startkapital verwenden
    if not paper_trading and MT5_VERFUEGBAR:
        account = mt5.account_info()  # type: ignore[union-attr]
        if account:
            # Echtes Startkapital aus MT5-Konto
            start_equity = account.equity
            logger.info(f"[{symbol}] MT5-Startkapital: {start_equity:,.2f} {account.currency}")
        else:
            logger.warning(f"[{symbol}] MT5-Kontodaten nicht lesbar – Kill-Switch nutzt Startkapital {kapital_start}")
            start_equity = kapital_start
    else:
        # Paper-Modus: konfiguriertes Startkapital verwenden
        start_equity = kapital_start
        logger.info(f"[{symbol}] Paper-Startkapital: {start_equity:,.2f} (simuliert)")

    # Simuliertes Kapital für Paper-Modus (wird nach jedem Trade aktualisiert)
    paper_kapital = start_equity

    while True:
        try:
            # Neue Kerze abwarten (prüfe alle 15 Sekunden)
            if not neue_kerze_abwarten(symbol, letzte_kerzen_zeit):
                time.sleep(15)
                continue

            # Neue Kerze erkannt!
            jetzt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\n[{symbol}] Neue H1-Kerze | {jetzt} UTC")

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

            # ---- Schritt 1: Daten von MT5 holen ----
            df = mt5_daten_holen(symbol)
            if df is None or len(df) < 250:
                logger.warning(f"[{symbol}] Zu wenige Daten ({len(df) if df is not None else 0}) – übersprungen")
                letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol)
                time.sleep(30)
                continue

            # ---- Schritt 2: Features berechnen ----
            df = features_berechnen(df)

            # ---- Schritt 3: Externe Features (Fear & Greed, BTC) ----
            df = externe_features_einfuegen(df)

            # NaN-Zeilen am Anfang (Warm-Up) entfernen
            df_clean = df.dropna(subset=FEATURE_SPALTEN)
            if len(df_clean) < 10:
                logger.warning(f"[{symbol}] Nach NaN-Bereinigung zu wenige Zeilen – übersprungen")
                letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol)
                continue

            # ---- Schritt 4: Signal generieren ----
            signal, prob, regime = signal_generieren(
                df_clean, modell, schwelle, regime_erlaubt
            )
            regime_name = REGIME_NAMEN.get(regime, "?")
            logger.info(
                f"[{symbol}] Signal={signal} | Prob={prob:.1%} | "
                f"Regime={regime} ({regime_name})"
            )

            # Trade loggen (auch wenn kein Signal)
            if signal != 0:
                n_signale += 1
                trade_loggen(symbol, signal, prob, regime, paper_trading)

            # ---- Schritt 5: Trade ausführen ----
            if signal != 0:
                # Offene Position prüfen (nur 1 Trade gleichzeitig!)
                if mt5_offene_position(symbol):
                    logger.info(f"[{symbol}] Bereits offene Position – kein neuer Trade")
                else:
                    erfolg = order_senden(
                        symbol, signal, lot, TP_PCT, SL_PCT, paper_trading
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
                            paper_kapital -= paper_kapital * 0.001  # 0.1% konservative Schätzung
            else:
                logger.info(f"[{symbol}] Kein Trade-Signal (Prob={prob:.1%} < Schwelle={schwelle:.0%})")

            # Zeitstempel der verarbeiteten Kerze speichern
            letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol)

            # Statistik alle 24 Kerzen (ca. 1x täglich)
            if n_trades > 0 and n_trades % 24 == 0:
                dd_aktuell = (start_equity - paper_kapital) / start_equity if paper_trading else 0.0
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
            "MT5 ML-Trading – Live-Trader (Phase 6)\n"
            "Läuft auf: Windows 11 Laptop mit MT5-Terminal\n\n"
            "Empfohlene Konfiguration (aus Phase 5 Backtest):\n"
            "  USDCAD: --symbol USDCAD --schwelle 0.60 --regime_filter 1,2\n"
            "  USDJPY: --symbol USDJPY --schwelle 0.60 --regime_filter 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="USDCAD",
        choices=SYMBOLE,
        help="Handelssymbol (Standard: USDCAD – bestes Backtest-Ergebnis!)",
    )
    parser.add_argument(
        "--schwelle",
        type=float,
        default=0.60,
        help=(
            "Mindest-Wahrscheinlichkeit für Trade-Ausführung (Standard: 0.60). "
            "Empfehlung aus Phase 5: 0.60"
        ),
    )
    parser.add_argument(
        "--regime_filter",
        type=str,
        default="1,2",
        help=(
            "Komma-getrennte Regime-Nummern (Standard: '1,2'). "
            "0=Seitwärts, 1=Aufwärtstrend, 2=Abwärtstrend, 3=Hohe Vola. "
            "Für USDJPY: '1' (nur Aufwärtstrend)"
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
        default="v1",
        help="Modell-Versions-Suffix (Standard: v1). Muss mit train_model.py übereinstimmen.",
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
    args = parser.parse_args()

    # ---- Regime-Filter parsen ----
    regime_erlaubt = None
    if args.regime_filter:
        try:
            regime_erlaubt = [int(r.strip()) for r in args.regime_filter.split(",")]
        except ValueError:
            print(f"Ungültiger --regime_filter: '{args.regime_filter}'. Erwartet: z.B. '1,2'")
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

    # ---- Modell laden ----
    symbol = args.symbol.upper()
    modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{args.version}.pkl"
    if not modell_pfad.exists():
        print(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Bitte Modell vom Linux-Server übertragen:\n"
            f"  scp SERVER:/mnt/1T-Data/XGBoost-LightGBM/models/"
            f"lgbm_{symbol.lower()}_{args.version}.pkl ./models/"
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

    # ---- Hauptschleife starten ----
    trading_loop(
        symbol=symbol,
        schwelle=args.schwelle,
        regime_erlaubt=regime_erlaubt,
        paper_trading=paper_trading,
        lot=args.lot,
        modell=modell,
        kill_switch_dd=args.kill_switch_dd,
        kapital_start=args.kapital_start,
    )

    # MT5 Verbindung beenden
    if MT5_VERFUEGBAR and mt5.terminal_info() is not None:
        mt5.shutdown()
        logger.info("MT5-Verbindung getrennt.")


if __name__ == "__main__":
    main()
