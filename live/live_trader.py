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
        scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_usdcad_v1.pkl USER@LAPTOP:./models/
        scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_usdjpy_v1.pkl USER@LAPTOP:./models/

        # Two-Stage (H1 + M5, für USDJPY):
        scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_htf_bias_usdjpy_H1_v4.pkl USER@LAPTOP:./models/
        scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_ltf_entry_usdjpy_M5_v4.pkl USER@LAPTOP:./models/

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
from typing import Any, Optional, Protocol, Tuple, cast

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
    GaussianHMM = None  # type: ignore[assignment]

# MetaTrader5 – NUR auf Windows verfügbar!
try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_VERFUEGBAR = True
except ImportError:
    MT5_VERFUEGBAR = False
    mt5 = None  # type: ignore

# Pylance darf MT5 im Linux-Workspace nicht auf None festnageln.
mt5 = cast(Any, mt5)

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

    def two_stage_modelle_laden(*args: Any, **kwargs: Any) -> tuple[object, object]:
        """Fallback-Stummel wenn Two-Stage-Modul lokal nicht importierbar ist."""
        raise ImportError("two_stage_signal nicht verfügbar")

    def zwei_stufen_signal(*args: Any, **kwargs: Any) -> Any:
        """Fallback-Stummel wenn Two-Stage-Modul lokal nicht importierbar ist."""
        raise ImportError("two_stage_signal nicht verfügbar")

# ============================================================
# Konfiguration und Pfade
# ============================================================


def _resolve_log_dir(
    base_dir: Path,
    cli_log_dir: str = "",
    cli_log_subdir: str = "",
) -> Path:
    """
    Bestimmt den effektiven Log-Ordner optional über Umgebungsvariablen.

    Unterstützte Variablen:
        MT5_TRADING_LOG_DIR: Absoluter Log-Pfad (hat Priorität)
        MT5_TRADING_LOG_SUBDIR: Unterordner unter BASE_DIR/logs

    Args:
        base_dir: Projekt-Basisverzeichnis.
        cli_log_dir: Optionaler absoluter Log-Pfad aus der CLI.
        cli_log_subdir: Optionaler Unterordner unter ``base_dir / "logs"`` aus der CLI.

    Returns:
        Aufgelöster Log-Pfad als Path.
    """
    # CLI-Override hat höchste Priorität, weil er pro Prozess explizit gesetzt wird.
    cli_log_dir = cli_log_dir.strip()
    if cli_log_dir:
        return Path(cli_log_dir).expanduser()

    cli_log_subdir = cli_log_subdir.strip()
    if cli_log_subdir:
        normalized_cli_subdir = cli_log_subdir.replace("\\", "/").strip("/ ")
        if normalized_cli_subdir:
            return base_dir / "logs" / Path(normalized_cli_subdir)

    # Absoluter Override per Umgebung: ideal für dedizierte Testläufe wie Test 128.
    env_log_dir = os.environ.get("MT5_TRADING_LOG_DIR", "").strip()
    if env_log_dir:
        return Path(env_log_dir).expanduser()

    # Relativer Unterordner unterhalb des Standard-Logpfads.
    env_log_subdir = os.environ.get("MT5_TRADING_LOG_SUBDIR", "").strip()
    if env_log_subdir:
        # Backslashes vereinheitlichen und leere Segmente vermeiden.
        normalized_subdir = env_log_subdir.replace("\\", "/").strip("/ ")
        if normalized_subdir:
            return base_dir / "logs" / Path(normalized_subdir)

    # Fallback: bisheriger Standardordner.
    return base_dir / "logs"


# Pfade (relativ zum Skript-Verzeichnis)
BASE_DIR = Path(__file__).parent.parent  # Ordner mt5_ml_trading/
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = _resolve_log_dir(BASE_DIR)


def configure_logging(log_dir: Path) -> Path:
    """
    Konfiguriert Logging für Terminal + Datei neu.

    Args:
        log_dir: Zielordner für `live_trader.log` und CSV-Dateien.

    Returns:
        Finaler Log-Ordner.
    """
    resolved_log_dir = log_dir.expanduser()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    globals()["LOG_DIR"] = resolved_log_dir

    # force=True stellt sicher, dass ein CLI-Override alte Handler sauber ersetzt.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(resolved_log_dir / "live_trader.log", encoding="utf-8"),
        ],
        force=True,
    )
    return resolved_log_dir


# Logging initialisieren (Fallback: Umgebung oder Standardordner)
configure_logging(LOG_DIR)
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

# Lockerere Live-Defaults für Phase 7 Paper-Feedback:
# Ziel = mehr echte Trades sehen, ohne H1/M5-Logik komplett auszuschalten.
STANDARD_SCHWELLE = 0.45  # Vorher 0.55 → deutlich aggressiver für mehr Entries
STANDARD_TWO_STAGE_COOLDOWN_BARS = 3  # Vorher 12 → auf M5 nur noch 15 Min Pause
REGIME_ADX_TREND_SCHWELLE = 18.0  # Vorher implizit 25 → weniger "Seitwärts"-Klassifikation
REGIME_HIGH_VOL_FAKTOR = 1.8  # Vorher 1.5 → weniger Bars als reine Hochvola markieren
STANDARD_TWO_STAGE_ALLOW_NEUTRAL_HTF = True  # Neutraler H1-Bias darf starke M5-Entries zulassen
STANDARD_STARTUP_OBSERVATION_BARS = 5  # Erst Markt 5 Kerzen beobachten, dann Trades zulassen

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

# MT5 Auto-Reconnect: Laufzeit-Zustand für Reconnect und Fehlerzähler.
# Dict statt global-Neuzuweisung, damit keine global-Statements nötig sind.
_MT5_RUNTIME_STATE: dict[str, Any] = {
    "credentials": {},  # {"server": ..., "login": ..., "password": ..., "pfad": ...}
    "ipc_fail_count": 0,
    "kerzen_debug_last_ts": 0.0,
}
_MT5_RECONNECT_AFTER: int = 3  # Nach N aufeinanderfolgenden Fehlern → Reconnect

# Regime-Namen (für Logging)
REGIME_NAMEN = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

KLASSEN_NAMEN = {
    0: "Short",
    1: "Neutral",
    2: "Long",
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


class WahrscheinlichkeitsModell(Protocol):
    """Minimales Protokoll für die im Live-Trader genutzten Modellmethoden."""

    feature_name_: list[str]

    def predict_proba(self, x_features: pd.DataFrame) -> Any:
        """Gibt Klassenwahrscheinlichkeiten für die übergebenen Features zurück."""


def _series_sign(values: pd.Series) -> pd.Series:
    """np.sign als echte Pandas-Serie zurückgeben, damit fillna/rolling typstabil bleiben."""
    return pd.Series(np.sign(values), index=values.index, dtype=float)


def _modell_feature_namen(modell: object) -> list[str]:
    """Liest die Feature-Namen robust vom Modell oder fällt auf Standard-Features zurück."""
    modell_any = cast(Any, modell)
    namen = getattr(modell_any, "feature_name_", None)
    if namen:
        return list(namen)
    return FEATURE_SPALTEN


def _modell_predict_proba(modell: object, x_features: pd.DataFrame) -> np.ndarray:
    """Ruft predict_proba typrobust auf und liefert ein NumPy-Array zurück."""
    modell_any = cast(Any, modell)
    return np.asarray(modell_any.predict_proba(x_features), dtype=float)


def _mt5_api() -> Any:
    """Liefert das MT5-Modul typrobust oder wirft einen klaren Fehler."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 ist nicht verfügbar")
    return cast(Any, mt5)

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
    result["stoch_cross"] = _series_sign(
        result["stoch_k"] - result["stoch_d"]
    ).fillna(0)

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

    close_d1 = (
        close.resample("1D").last().dropna()
    )  # pandas 4.x erwartet Großbuchstabe "D"
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
    # Gelockerte Live-Regime-Regeln:
    # - Trend bereits ab ADX > 18 statt > 25
    # - Hochvola erst bei deutlicherem ATR-Ausreißer
    # Ergebnis: weniger Kerzen landen pauschal in "Seitwärts".
    hoch_vol = atr_pct > (REGIME_HIGH_VOL_FAKTOR * median_atr)
    aufwaerts = (
        (adx > REGIME_ADX_TREND_SCHWELLE)
        & (result["close"] > result["sma_50"])
        & ~hoch_vol
    )
    abwaerts = (
        (adx > REGIME_ADX_TREND_SCHWELLE)
        & (result["close"] < result["sma_50"])
        & ~hoch_vol
    )
    regime[aufwaerts] = 1
    regime[abwaerts] = 2
    regime[hoch_vol] = 3
    result["market_regime"] = regime

    # --- HMM-Regime (optional, mit robustem Fallback) ---
    if HAS_HMMLEARN and GaussianHMM is not None:
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
    regime_spalte_eff = (
        regime_spalte if regime_spalte in letzte_kerze.columns else "market_regime"
    )
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
    modell_features = _modell_feature_namen(modell)

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
    proba = _modell_predict_proba(modell, x_features)[0]
    raw_pred = int(np.argmax(proba))
    long_prob = float(proba[2])
    short_prob = float(proba[0])
    short_schwelle_eff = (
        float(short_schwelle)
        if short_schwelle is not None
        else float(1.0 - schwelle if decision_mapping == "long_prob" else schwelle)
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

        # Detaillierte HTF/LTF-Debug-Ausgabe:
        # Zeigt Rohklassen + Wahrscheinlichkeiten beider Stufen und macht sichtbar,
        # ob ein LTF-Entry am Schwellenfilter oder an der HTF-Logik scheitert.
        logger.info(
            f"[{symbol}] 🔍 TWO-STAGE DEBUG | "
            f"HTF={KLASSEN_NAMEN.get(ts_signal.htf_bias_klasse, str(ts_signal.htf_bias_klasse))} "
            f"(S={ts_signal.htf_bias_proba['short']:.1%}, "
            f"N={ts_signal.htf_bias_proba['neutral']:.1%}, "
            f"L={ts_signal.htf_bias_proba['long']:.1%}) | "
            f"LTF-Rohklasse={KLASSEN_NAMEN.get(ts_signal.ltf_klasse, str(ts_signal.ltf_klasse))} "
            f"/ Signal-vor-Filter={ts_signal.ltf_signal_vor_filter} "
            f"(S={ts_signal.ltf_proba['short']:.1%}, "
            f"N={ts_signal.ltf_proba['neutral']:.1%}, "
            f"L={ts_signal.ltf_proba['long']:.1%}) | "
            f"LTF-Schwelle={'OK' if ts_signal.ltf_entry_erlaubt else 'BLOCKIERT'}"
        )

        # ---- Kongruenz-Filter: HTF-Bias und LTF-Signal müssen übereinstimmen ----
        # HTF-Bias 0=Short, 1=Neutral, 2=Long | LTF-Signal -1=Short, 0=Neutral, 2=Long
        # Erlaubt: HTF-Short + LTF-Short, HTF-Long + LTF-Long
        # Blockiert: HTF-Bias widerspricht LTF-Signal, oder HTF=Neutral
        htf_bias = ts_signal.htf_bias_klasse
        ltf_signal = ts_signal.signal

        # HTF-Bias und LTF-Signal im Config speichern (für Trade-Logging)
        two_stage_config["last_htf_bias"] = htf_bias
        two_stage_config["last_ltf_signal"] = ltf_signal

        allow_neutral_htf_entries = bool(
            two_stage_config.get(
                "allow_neutral_htf_entries", STANDARD_TWO_STAGE_ALLOW_NEUTRAL_HTF
            )
        )
        kongruent = True  # Standardannahme: neutral-Signale sind immer OK

        if ltf_signal != 0:  # Nur aktive Trades prüfen (Short/Long)
            if htf_bias == 1:
                # Gelockerte Logik: neutraler H1-Bias blockiert nicht mehr hart.
                # So dürfen starke M5-Entries trotzdem als Test-Trade durchgehen.
                if allow_neutral_htf_entries:
                    logger.info(
                        f"[{symbol}] ⚠️ HTF-BIAS NEUTRAL, aber M5-Entry erlaubt | "
                        f"LTF-Signal={ltf_signal} | LTF-Prob={ts_signal.prob:.1%}"
                    )
                else:
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

        # Zusatzhinweis wenn Signal=0 nicht durch HTF-Filter, sondern bereits im LTF-Modell entsteht.
        if ts_signal.signal == 0:
            if ts_signal.ltf_signal_vor_filter == 0:
                logger.info(
                    f"[{symbol}] ℹ️ LTF bleibt neutral – kein Entry vor HTF-Gates. "
                    f"Rohklasse={KLASSEN_NAMEN.get(ts_signal.ltf_klasse, str(ts_signal.ltf_klasse))}"
                )
            elif not ts_signal.ltf_entry_erlaubt:
                logger.info(
                    f"[{symbol}] ℹ️ LTF-Entry an Schwelle gescheitert | "
                    f"Rohsignal={ts_signal.ltf_signal_vor_filter} | Prob={ts_signal.prob:.1%} | "
                    f"Schwelle={schwelle:.1%}"
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

    mt5_api = _mt5_api()

    # MT5 initialisieren und verbinden
    if pfad:
        ok = mt5_api.initialize(
            path=pfad, server=server, login=login, password=password
        )
    else:
        ok = mt5_api.initialize(server=server, login=login, password=password)

    if not ok:
        logger.error(f"MT5 Verbindung fehlgeschlagen: {mt5_api.last_error()}")
        return False

    # Zugangsdaten für Auto-Reconnect speichern
    _MT5_RUNTIME_STATE["credentials"] = {
        "server": server,
        "login": login,
        "password": password,
        "pfad": pfad,
    }
    _MT5_RUNTIME_STATE["ipc_fail_count"] = 0

    # Konto-Info ausgeben
    konto = mt5_api.account_info()
    logger.info(
        f"MT5 verbunden | Server: {server} | "
        f"Konto: {konto.login} | Saldo: {konto.balance:.2f} {konto.currency}"
    )
    return True


def mt5_reconnect() -> bool:
    """
    Versucht die MT5-IPC-Verbindung wiederherzustellen.

    Wird automatisch aufgerufen nach mehreren aufeinanderfolgenden IPC-Fehlern.
    Nutzt die bei mt5_verbinden() gespeicherten Zugangsdaten.

    Returns:
        True wenn Reconnect erfolgreich, False sonst.
    """
    if not MT5_VERFUEGBAR or not _MT5_RUNTIME_STATE["credentials"]:
        logger.error("MT5-Reconnect nicht möglich – keine Zugangsdaten gespeichert")
        return False

    mt5_api = _mt5_api()

    logger.warning("🔄 MT5 Auto-Reconnect: Verbindung wird wiederhergestellt ...")

    # Alte Verbindung sauber beenden
    try:
        mt5_api.shutdown()
    except (RuntimeError, OSError):
        pass  # Shutdown kann fehlschlagen wenn IPC bereits tot ist

    time.sleep(2)  # Kurze Pause damit MT5 Terminal sich stabilisieren kann

    # Neu verbinden mit gespeicherten Zugangsdaten
    creds = _MT5_RUNTIME_STATE["credentials"]
    if creds["pfad"]:
        ok = mt5_api.initialize(
            path=creds["pfad"],
            server=creds["server"],
            login=creds["login"],
            password=creds["password"],
        )
    else:
        ok = mt5_api.initialize(
            server=creds["server"],
            login=creds["login"],
            password=creds["password"],
        )

    if ok:
        _MT5_RUNTIME_STATE["ipc_fail_count"] = 0
        konto = mt5_api.account_info()
        if konto:
            logger.info(
                f"✅ MT5 Reconnect erfolgreich | Server: {creds['server']} | "
                f"Konto: {konto.login} | Saldo: {konto.balance:.2f} {konto.currency}"
            )
        else:
            logger.info("✅ MT5 Reconnect erfolgreich (Kontodaten nicht lesbar)")
        return True
    else:
        logger.error(f"❌ MT5 Reconnect fehlgeschlagen: {mt5_api.last_error()}")
        return False


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

    mt5_api = _mt5_api()

    cfg = TIMEFRAME_CONFIG.get(timeframe)
    if cfg is None:
        return None

    konst_name = cfg["mt5_name"]
    return getattr(mt5_api, konst_name, None)


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

    mt5_api = _mt5_api()

    # Barren von Position 0 (aktuelle Kerze) bis n-1
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        logger.error(f"Unbekannter oder nicht verfügbarer Zeitrahmen: {timeframe}")
        return None

    n_bars_effektiv = (
        n_barren if n_barren is not None else n_barren_fuer_timeframe(timeframe)
    )

    rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, n_bars_effektiv)
    if rates is None:
        fehler = mt5_api.last_error()
        _MT5_RUNTIME_STATE["ipc_fail_count"] += 1
        logger.warning(
            f"[{symbol}] Keine Daten von MT5: {fehler} "
            f"(Fehler {_MT5_RUNTIME_STATE['ipc_fail_count']}/{_MT5_RECONNECT_AFTER})"
        )
        # Nach N aufeinanderfolgenden Fehlern: Auto-Reconnect versuchen
        if _MT5_RUNTIME_STATE["ipc_fail_count"] >= _MT5_RECONNECT_AFTER:
            if mt5_reconnect():
                # Retry nach erfolgreichem Reconnect
                rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, n_bars_effektiv)
                if rates is not None:
                    logger.info(
                        f"[{symbol}] Daten nach Reconnect erfolgreich geladen ✓"
                    )
                else:
                    logger.error(
                        f"[{symbol}] Auch nach Reconnect keine Daten: {mt5_api.last_error()}"
                    )
                    return None
            else:
                return None
        else:
            return None

    # Erfolg → Fehlerzähler zurücksetzen
    _MT5_RUNTIME_STATE["ipc_fail_count"] = 0

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
    mt5_api = _mt5_api()
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        return None
    rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, 2)
    if rates is None or len(rates) < 2:
        fehler = mt5_api.last_error()
        _MT5_RUNTIME_STATE["ipc_fail_count"] += 1
        logger.warning(
            f"[{symbol}] MT5 liefert keine Kerzen-Daten: {fehler} "
            f"(Fehler {_MT5_RUNTIME_STATE['ipc_fail_count']}/{_MT5_RECONNECT_AFTER})"
        )
        # Nach N aufeinanderfolgenden Fehlern: Auto-Reconnect versuchen
        if _MT5_RUNTIME_STATE["ipc_fail_count"] >= _MT5_RECONNECT_AFTER:
            if mt5_reconnect():
                # Retry nach erfolgreichem Reconnect
                rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, 2)
                if rates is not None and len(rates) >= 2:
                    _MT5_RUNTIME_STATE["ipc_fail_count"] = 0
                    logger.info(f"[{symbol}] Kerzen-Daten nach Reconnect geladen ✓")
                    return datetime.fromtimestamp(
                        int(rates[1]["time"]), tz=timezone.utc
                    )
        return None
    # Erfolg → Fehlerzähler zurücksetzen
    _MT5_RUNTIME_STATE["ipc_fail_count"] = 0
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
    mt5_api = _mt5_api()
    positionen = mt5_api.positions_get(symbol=symbol)
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
) -> Optional[dict[str, Any]]:
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
        Dict mit Order-Metadaten bei Erfolg, sonst None.
        Keys: success, deal_ticket, position_ticket, entry_price, sl_price, tp_price
    """
    richtung_str = "LONG (Kaufen)" if richtung == 2 else "SHORT (Verkaufen)"

    if paper_trading:
        logger.info(
            f"[PAPER] {symbol} {richtung_str} | "
            f"Lot={lot} | TP={tp_pct:.1%} | SL={sl_pct:.1%}"
        )
        return {
            "success": True,
            "deal_ticket": None,
            "position_ticket": None,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
        }

    # ====== ECHTE ORDER ======
    if not MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – Order nicht gesendet!")
        return None

    mt5_api = _mt5_api()

    # Symbol aktivieren (falls nicht im Market Watch)
    if not mt5_api.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} nicht verfügbar!")
        return None

    symbol_info = mt5_api.symbol_info(symbol)
    tick = mt5_api.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        logger.error(f"Symbol-Info für {symbol} nicht abrufbar!")
        return None

    # Preis und TP/SL berechnen
    if richtung == 2:  # Long: Buy
        order_type = mt5_api.ORDER_TYPE_BUY
        preis = tick.ask
        sl_preis = round(preis * (1.0 - sl_pct), symbol_info.digits)
        tp_preis = round(preis * (1.0 + tp_pct), symbol_info.digits)
    else:  # Short: Sell
        order_type = mt5_api.ORDER_TYPE_SELL
        preis = tick.bid
        sl_preis = round(preis * (1.0 + sl_pct), symbol_info.digits)
        tp_preis = round(preis * (1.0 - tp_pct), symbol_info.digits)

    # Order-Request (PFLICHT: Stop-Loss!)
    request = {
        "action": mt5_api.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": preis,
        "sl": sl_preis,  # Stop-Loss ist PFLICHT!
        "tp": tp_preis,
        "deviation": 20,  # Max. Slippage in Punkte
        "magic": MAGIC_NUMBER,  # Eindeutige ID für diesen Bot
        "comment": "ML-Phase6",
        "type_time": mt5_api.ORDER_TIME_GTC,
        "type_filling": mt5_api.ORDER_FILLING_IOC,
    }

    result = mt5_api.order_send(request)
    if result.retcode != mt5_api.TRADE_RETCODE_DONE:
        logger.error(f"Order fehlgeschlagen: Code={result.retcode} | {result.comment}")
        return None

    logger.info(
        f"Order ausgeführt: {symbol} {richtung_str} | "
        f"{lot} Lot @ {preis:.5f} | SL={sl_preis:.5f} | TP={tp_preis:.5f}"
    )
    deal_ticket = int(getattr(result, "deal", 0) or 0)
    position_ticket = int(getattr(result, "order", 0) or 0)
    return {
        "success": True,
        "deal_ticket": deal_ticket,
        "position_ticket": position_ticket,
        "entry_price": preis,
        "sl_price": sl_preis,
        "tp_price": tp_preis,
    }


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
    htf_bias: Optional[int] = None,
    ltf_signal: Optional[int] = None,
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
        htf_bias:      HTF-Bias aus Two-Stage (0=Short, 1=Neutral, 2=Long, None=kein Two-Stage)
        ltf_signal:    LTF-Signal aus Two-Stage (-1=Short, 0=Neutral, 2=Long, None=kein Two-Stage)
    """
    log_pfad = LOG_DIR / f"{symbol}_signals.csv"

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
        "htf_bias": htf_bias if htf_bias is not None else "",
        "ltf_signal": ltf_signal if ltf_signal is not None else "",
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
            mt5_csv = mt5_common / f"{symbol}_signals.csv"
            df_log.to_csv(
                mt5_csv,
                mode="a",
                header=not mt5_csv.exists(),
                index=False,
            )
    except OSError:
        pass  # Dashboard-Sync ist nice-to-have, kein Fehler


def trade_close_loggen(
    symbol: str,
    ticket: int,
    richtung: int,
    entry_price: float,
    exit_price: float,
    pnl_pips: float,
    pnl_money: float,
    close_grund: str,
    dauer_minuten: int,
    htf_bias: Optional[int] = None,
    ltf_signal: Optional[int] = None,
) -> None:
    """
    Schreibt ein CLOSE-Event in die Trade-Log-CSV (gleiche Datei wie Signale).

    Wird aufgerufen wenn ein zuvor geöffneter Trade geschlossen wurde (SL/TP/manuell).

    Args:
        symbol:         Handelssymbol
        ticket:         MT5-Position-Ticket
        richtung:       2=Long, -1=Short
        entry_price:    Einstiegspreis
        exit_price:     Ausstiegspreis
        pnl_pips:       Gewinn/Verlust in Pips
        pnl_money:      Gewinn/Verlust in Kontowährung (USD)
        close_grund:    Grund der Schließung (TP/SL/manuell/Kill-Switch)
        dauer_minuten:  Dauer des Trades in Minuten
        htf_bias:       HTF-Bias bei Eröffnung (0/1/2 oder None)
        ltf_signal:     LTF-Signal bei Eröffnung (-1/0/2 oder None)
    """
    log_pfad = LOG_DIR / f"{symbol}_closes.csv"

    eintrag = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "richtung": "CLOSE-Long" if richtung == 2 else "CLOSE-Short",
        "signal": 0,  # Kein neues Signal – Schließung
        "prob": 0.0,
        "regime": -1,  # Nicht relevant bei Schließung
        "regime_name": "CLOSE",
        "paper_trading": False,  # Wurde tatsächlich ausgeführt
        "modus": "CLOSE",
        "entry_price": round(entry_price, 5),
        "sl_price": 0.0,  # Nicht relevant bei Schließung
        "tp_price": 0.0,
        "htf_bias": htf_bias if htf_bias is not None else "",
        "ltf_signal": ltf_signal if ltf_signal is not None else "",
        "exit_price": round(exit_price, 5),
        "pnl_pips": round(pnl_pips, 1),
        "pnl_money": round(pnl_money, 2),
        "close_grund": close_grund,
        "dauer_min": dauer_minuten,
        "ticket": ticket,
    }

    df_log = pd.DataFrame([eintrag])
    df_log.to_csv(
        log_pfad,
        mode="a",
        header=not log_pfad.exists(),
        index=False,
    )

    # Auch in MT5 Common/Files kopieren
    try:
        mt5_common = (
            Path(os.environ.get("APPDATA", ""))
            / "MetaQuotes"
            / "Terminal"
            / "Common"
            / "Files"
        )
        if mt5_common.exists():
            mt5_csv = mt5_common / f"{symbol}_closes.csv"
            df_log.to_csv(
                mt5_csv,
                mode="a",
                header=not mt5_csv.exists(),
                index=False,
            )
    except OSError:
        pass

    logger.info(
        f"[{symbol}] 📝 CLOSE-Event geloggt: Ticket={ticket} | "
        f"PnL={pnl_money:+.2f} USD ({pnl_pips:+.1f} Pips) | "
        f"Grund={close_grund} | Dauer={dauer_minuten} Min"
    )


def offenen_trade_pruefen(
    symbol: str,
    letzter_trade: Optional[dict],
    paper_trading: bool,
) -> Optional[dict]:
    """
    Prüft ob ein zuvor geöffneter Trade noch offen ist.

    Wenn der Trade geschlossen wurde (SL/TP/manuell), wird ein CLOSE-Event
    geloggt und None zurückgegeben. Wenn noch offen → letzter_trade unverändert.

    Args:
        symbol:         Handelssymbol
        letzter_trade:  Dict mit Trade-Info oder None wenn kein offener Trade
        paper_trading:  True = Paper-Modus (kein MT5-Abfrage möglich)

    Returns:
        letzter_trade unverändert wenn Trade noch offen, None wenn geschlossen/geloggt
    """
    if letzter_trade is None:
        return None

    # Paper-Modus: Keine echte Position → wir können nicht prüfen
    # Paper-Trades werden sofort als "geschlossen" betrachtet (kein Tracking möglich)
    if paper_trading:
        return None

    # Live-Modus: MT5 fragen ob die Position noch existiert
    if not MT5_VERFUEGBAR:
        return letzter_trade  # Kann nicht prüfen → behalten

    positionen = mt5.positions_get(symbol=symbol)  # type: ignore[union-attr]
    position_ticket = int(letzter_trade.get("position_ticket", 0) or 0)
    deal_ticket = int(letzter_trade.get("deal_ticket", 0) or 0)

    # Position noch offen?
    if positionen and position_ticket > 0:
        for pos in positionen:
            if int(pos.ticket) == position_ticket:
                return letzter_trade  # Noch offen

    # Falls kein position_ticket bekannt ist, aber noch ML-Position offen ist,
    # Trade-Tracking beibehalten statt fälschlich als geschlossen zu markieren.
    if positionen and position_ticket <= 0:
        if any(int(getattr(p, "magic", 0)) == MAGIC_NUMBER for p in positionen):
            return letzter_trade

    # Position ist geschlossen! → History abfragen für PnL
    try:
        from datetime import timedelta

        # Trade-History der letzten 7 Tage abfragen (ausreichend für offene Trades)
        jetzt = datetime.now(timezone.utc)
        von = jetzt - timedelta(days=7)
        deals = mt5.history_deals_get(von, jetzt, group=symbol)  # type: ignore[union-attr]

        if deals:
            # Deal robust finden:
            # 1) Primär über position_ticket + DEAL_ENTRY_OUT
            # 2) Fallback über deal_ticket (Open-Deal) -> gleiche Position
            deal_entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)

            target_position_id = position_ticket
            if target_position_id <= 0 and deal_ticket > 0:
                open_deals = [
                    d for d in deals if int(getattr(d, "ticket", 0)) == deal_ticket
                ]
                if open_deals:
                    target_position_id = int(
                        getattr(open_deals[-1], "position_id", 0) or 0
                    )

            close_deals = []
            if target_position_id > 0:
                close_deals = [
                    d
                    for d in deals
                    if int(getattr(d, "position_id", 0) or 0) == target_position_id
                    and int(getattr(d, "entry", -1)) == int(deal_entry_out)
                ]

            # Fallback: falls MT5 entry-Konstanten anders gemappt sind
            if not close_deals and target_position_id > 0:
                close_deals = [
                    d
                    for d in deals
                    if int(getattr(d, "position_id", 0) or 0) == target_position_id
                    and int(getattr(d, "entry", -1)) in (1, 3)
                ]

            if close_deals:
                deal = close_deals[-1]  # Letzter Close-Deal
                exit_price = deal.price
                pnl_money = deal.profit + deal.commission + deal.swap
                # PnL in Pips berechnen
                entry_price = letzter_trade.get("entry_price", 0.0)
                richtung = letzter_trade.get("richtung", 0)
                if richtung == 2:  # Long
                    pnl_pips = (exit_price - entry_price) * _pip_faktor(symbol)
                else:  # Short
                    pnl_pips = (entry_price - exit_price) * _pip_faktor(symbol)

                # Dauer berechnen
                open_zeit = letzter_trade.get("open_zeit", jetzt)
                dauer_min = int((jetzt - open_zeit).total_seconds() / 60)

                # Close-Grund aus MT5 Deal-Feldern ableiten (kein Raten via PnL)
                close_grund = _close_grund_aus_deal(deal)

                trade_close_loggen(
                    symbol=symbol,
                    ticket=target_position_id,
                    richtung=richtung,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pips=pnl_pips,
                    pnl_money=pnl_money,
                    close_grund=close_grund,
                    dauer_minuten=dauer_min,
                    htf_bias=letzter_trade.get("htf_bias"),
                    ltf_signal=letzter_trade.get("ltf_signal"),
                )
                return None  # Trade abgeschlossen und geloggt

        # Kein Deal gefunden → vorsichtshalber loggen
        logger.warning(
            f"[{symbol}] Trade Position={position_ticket} (Deal={deal_ticket}) geschlossen, "
            "aber kein Deal in History gefunden"
        )
    except (AttributeError, TypeError, ValueError, OSError) as e:
        logger.error(f"[{symbol}] Fehler beim PnL-Abruf: {e}", exc_info=True)

    return None  # Trade-Tracking zurücksetzen


def _pip_faktor(symbol: str) -> float:
    """
    Gibt den Pip-Multiplikator für ein Symbol zurück.

    JPY-Paare haben 2 Dezimalstellen (1 Pip = 0.01),
    alle anderen 4 Dezimalstellen (1 Pip = 0.0001).

    Args:
        symbol: Handelssymbol (z.B. USDJPY, USDCAD)

    Returns:
        Multiplikator um Preisdifferenz in Pips umzurechnen
    """
    if "JPY" in symbol.upper():
        return 100.0  # 1 Pip = 0.01 bei JPY-Paaren
    return 10000.0  # 1 Pip = 0.0001 bei Standard-Paaren


def _close_grund_aus_deal(deal: Any) -> str:
    """
    Leitet den Schließungsgrund aus MT5-Deal-Feldern ab.

    Nutzt primär Deal-Reason, sekundär Entry-Typ. Kein PnL-basiertes Raten.

    Args:
        deal: MT5 Deal-Objekt aus history_deals_get

    Returns:
        TP, SL, SO, MANUAL, SYSTEM, OUT_BY, UNKNOWN
    """
    reason = int(getattr(deal, "reason", -1))
    entry = int(getattr(deal, "entry", -1))

    # Reason-Mapping (robust über getattr, falls Konstanten variieren)
    if reason == int(getattr(mt5, "DEAL_REASON_TP", -999)):
        return "TP"
    if reason == int(getattr(mt5, "DEAL_REASON_SL", -999)):
        return "SL"
    if reason == int(getattr(mt5, "DEAL_REASON_SO", -999)):
        return "SO"  # Stop-Out
    if reason in {
        int(getattr(mt5, "DEAL_REASON_CLIENT", -999)),
        int(getattr(mt5, "DEAL_REASON_MOBILE", -999)),
        int(getattr(mt5, "DEAL_REASON_WEB", -999)),
    }:
        return "MANUAL"
    if reason == int(getattr(mt5, "DEAL_REASON_EXPERT", -999)):
        return "SYSTEM"

    # Fallback über Entry-Typ
    if entry == int(getattr(mt5, "DEAL_ENTRY_OUT_BY", -999)):
        return "OUT_BY"
    if entry == int(getattr(mt5, "DEAL_ENTRY_OUT", -999)):
        return "UNKNOWN"

    return "UNKNOWN"


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

    mt5_api = _mt5_api()

    positionen = mt5_api.positions_get(symbol=symbol)
    if not positionen:
        return

    logger.info(f"[{symbol}] Schließe {len(positionen)} offene Position(en) ...")
    for pos in positionen:
        # Gegenläufige Order zum Schließen der Position
        tick = mt5_api.symbol_info_tick(symbol)
        if tick is None:
            continue

        # Long-Position → mit Sell schließen; Short-Position → mit Buy schließen
        if pos.type == mt5_api.ORDER_TYPE_BUY:
            close_type = mt5_api.ORDER_TYPE_SELL
            preis = tick.bid
        else:
            close_type = mt5_api.ORDER_TYPE_BUY
            preis = tick.ask

        request = {
            "action": mt5_api.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": preis,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Kill-Switch",
            "type_time": mt5_api.ORDER_TIME_GTC,
            "type_filling": mt5_api.ORDER_FILLING_IOC,
        }
        result = mt5_api.order_send(request)
        if result.retcode == mt5_api.TRADE_RETCODE_DONE:
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
    jetzt = time.time()
    if jetzt - float(_MT5_RUNTIME_STATE.get("kerzen_debug_last_ts", 0.0)) > 60:
        logger.debug(
            f"[{symbol}] Kerzen-Check: Letzte={letzte_kerzen_zeit} | "
            f"Aktuell={aktuelle_kerze}"
        )
        _MT5_RUNTIME_STATE["kerzen_debug_last_ts"] = jetzt

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
    tp_pct: float = 0.006,
    sl_pct: float = 0.003,
    cooldown_bars: int = 12,
    startup_observation_bars: int = STANDARD_STARTUP_OBSERVATION_BARS,
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
        tp_pct:          Take-Profit in Dezimal (Standard: 0.006 = 0.6%)
        sl_pct:          Stop-Loss Fallback in Dezimal (Standard: 0.003 = 0.3%)
        cooldown_bars:   Mindest-Bars zwischen Trades (Standard: 12, bei M5 = 1h)
        startup_observation_bars: Anzahl neuer Kerzen, die erst nur beobachtet werden.
    """
    modus_str = "PAPER-TRADING" if paper_trading else "⚠️  LIVE-TRADING MIT ECHTEM GELD!"
    regime_str = (
        [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        if regime_erlaubt
        else "alle"
    )

    # Two-Stage M5-Takt: Flag vorab setzen (wird für Logging + Loop gebraucht)
    ts_aktiv = two_stage_config is not None and two_stage_config.get("enable", False)
    ts_cfg: dict[str, Any] = cast(dict[str, Any], two_stage_config) if ts_aktiv else {}

    # RRR (Risk-Reward-Ratio) aus tp_pct/sl_pct berechnen
    rrr = tp_pct / sl_pct if sl_pct > 0 else 1.0

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
        logger.info(f"TP/SL-RRR:      {rrr:.1f}:1 (TP = SL × {rrr:.1f})")
    else:
        logger.info(f"TP/SL:          {tp_pct:.1%} / {sl_pct:.1%} (RRR={rrr:.1f}:1)")
    logger.info(f"Cooldown:       {cooldown_bars} Bars zwischen Trades")
    logger.info(f"Beobachtung:    {startup_observation_bars} neue Kerzen nur beobachten")
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
            f"M5-Takt:        AKTIV → Loop auf {ts_cfg.get('ltf_timeframe', 'M5')}, "
            f"HTF-Bias auf H1 (gecached)"
        )
        logger.info(
            f"Warte auf neue {ts_cfg.get('ltf_timeframe', 'M5')}-Kerze ..."
        )
    else:
        logger.info(f"Warte auf neue {timeframe}-Kerze ...")
    logger.info("=" * 65)

    letzte_kerzen_zeit: Optional[datetime] = None
    n_signale = 0  # Gesamt-Signale
    n_trades = 0  # Ausgeführte Trades
    bars_seit_letztem_trade = cooldown_bars  # Cooldown-Zähler (startet "bereit")

    # Letzter eröffneter Trade (für Schließungs-Erkennung und PnL-Logging)
    # Dict-Keys: ticket, richtung, entry_price, open_zeit, htf_bias, ltf_signal
    letzter_trade: Optional[dict] = None
    verarbeitete_kerzen = 0  # Anzahl seit Start verarbeiteter neuer Kerzen

    # ---- Kill-Switch: Startkapital ermitteln ----
    # Im Live-Modus: echtes Kontostand von MT5 lesen
    # Im Paper-Modus: übergebenes Startkapital verwenden
    if not paper_trading and MT5_VERFUEGBAR:
        account = _mt5_api().account_info()
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
        effektiver_tf = ts_cfg.get("ltf_timeframe", "M5")
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
            # ---- Trade-Schließung prüfen (PnL-Logging) ----
            # Wenn ein Trade offen war, prüfen ob er inzwischen geschlossen wurde
            if letzter_trade is not None:
                letzter_trade = offenen_trade_pruefen(
                    symbol, letzter_trade, paper_trading
                )

            # Neue Kerze abwarten (M5 bei Two-Stage, sonst H1)
            if not neue_kerze_abwarten(symbol, letzte_kerzen_zeit, effektiver_tf):
                time.sleep(15)
                continue

            # Neue Kerze erkannt!
            aktuelle_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
            if aktuelle_kerzen_zeit is not None:
                kerzen_ts = aktuelle_kerzen_zeit.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Fallback nur fürs Logging, falls MT5-Zeitpunkt kurzzeitig nicht lesbar ist.
                kerzen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\n[{symbol}] Neue {effektiver_tf}-Kerze | {kerzen_ts} UTC")
            verarbeitete_kerzen += 1

            # Cooldown-Zähler bei jeder neuen Kerze erhöhen
            bars_seit_letztem_trade += 1

            externe_features = externe_features_holen(cache=externe_feature_cache)

            # ---- Kill-Switch prüfen ----
            # Im Live-Modus: aktuellen MT5-Kontostand lesen
            # Im Paper-Modus: simuliertes Kapital prüfen
            if not paper_trading and MT5_VERFUEGBAR:
                account = _mt5_api().account_info()
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
                        ts_cfg["htf_df"] = cached_h1_df_clean
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
                ltf_tf = ts_cfg["ltf_timeframe"]
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
                ts_cfg["ltf_df"] = ltf_df_clean

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
                modell_feature_namen = _modell_feature_namen(modell)
                dropna_features = [f for f in modell_feature_namen if f in df.columns]
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
                tp_aktuell = sl_aktuell * rrr  # asymmetrisches TP/SL (z.B. RRR 2:1)
                sl_info = (
                    f"ATR-SL={sl_aktuell:.2%} | TP={tp_aktuell:.2%} (RRR {rrr:.1f}:1)"
                )
            else:
                sl_aktuell = sl_pct  # Fallback: festes SL aus CLI
                tp_aktuell = tp_pct  # Festes TP aus CLI
                sl_info = f"Fix-SL={sl_aktuell:.1%} | Fix-TP={tp_aktuell:.1%}"

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
                # HTF-Bias und LTF-Signal aus Two-Stage-Config extrahieren (falls aktiv)
                log_htf_bias = (
                    two_stage_config.get("last_htf_bias") if two_stage_config else None
                )
                log_ltf_signal = (
                    two_stage_config.get("last_ltf_signal")
                    if two_stage_config
                    else None
                )
                trade_loggen(
                    symbol,
                    signal,
                    prob,
                    regime,
                    paper_trading,
                    entry_price=close_preis,
                    sl_price=log_sl,
                    tp_price=log_tp,
                    htf_bias=log_htf_bias,
                    ltf_signal=log_ltf_signal,
                )

            # ---- Schritt 5: Trade ausführen ----
            if signal != 0:
                if verarbeitete_kerzen <= startup_observation_bars:
                    logger.info(
                        f"[{symbol}] 👀 Beobachtungsphase aktiv – Kerze {verarbeitete_kerzen}/{startup_observation_bars}. "
                        f"Signal wird nur beobachtet, noch kein Trade."
                    )
                # Cooldown prüfen (verhindert Overtrading bei dauerhaftem Signal)
                elif bars_seit_letztem_trade < cooldown_bars:
                    logger.info(
                        f"[{symbol}] Cooldown aktiv – noch {cooldown_bars - bars_seit_letztem_trade} "
                        f"Bars warten (von {cooldown_bars})"
                    )
                # Offene Position prüfen (nur 1 Trade gleichzeitig!)
                elif mt5_offene_position(symbol):
                    logger.info(
                        f"[{symbol}] Bereits offene Position – kein neuer Trade"
                    )
                else:
                    order_info = order_senden(
                        symbol, signal, lot, tp_aktuell, sl_aktuell, paper_trading
                    )
                    if order_info is not None:
                        n_trades += 1
                        bars_seit_letztem_trade = 0  # Cooldown-Zähler zurücksetzen
                        richtung_str = "Long" if signal == 2 else "Short"
                        logger.info(
                            f"[{symbol}] Trade #{n_trades}: {richtung_str} | "
                            f"Prob={prob:.1%} | Regime={regime_name}"
                        )

                        # Trade-Info speichern für Schließungs-Erkennung
                        trade_htf = (
                            two_stage_config.get("last_htf_bias")
                            if two_stage_config
                            else None
                        )
                        trade_ltf = (
                            two_stage_config.get("last_ltf_signal")
                            if two_stage_config
                            else None
                        )
                        if not paper_trading and MT5_VERFUEGBAR:
                            # Ticket/Deal direkt aus order_send-Resultat verwenden
                            letzter_trade = {
                                "position_ticket": int(
                                    order_info.get("position_ticket", 0) or 0
                                ),
                                "deal_ticket": int(
                                    order_info.get("deal_ticket", 0) or 0
                                ),
                                "richtung": signal,
                                "entry_price": float(
                                    order_info.get("entry_price", close_preis)
                                ),
                                "open_zeit": datetime.now(timezone.utc),
                                "htf_bias": trade_htf,
                                "ltf_signal": trade_ltf,
                            }
                        else:
                            # Paper-Modus: kein echtes Ticket, PnL nicht trackbar
                            letzter_trade = None

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
            letzte_kerzen_zeit = aktuelle_kerzen_zeit

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
            "  USDCAD: --symbol USDCAD --schwelle 0.48 --regime_filter 0,1,2,3 --atr_sl 1\n"
            "  USDJPY: --symbol USDJPY --schwelle 0.45 --regime_filter 0,1,2,3 --atr_sl 1"
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
        default=STANDARD_SCHWELLE,
        help=(
            "Long-Schwelle für Trade-Ausführung (Standard: 0.45). "
            "Bewusst etwas lockerer, damit im Paper-Betrieb mehr M5-Entries sichtbar werden."
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
        default=0,
        help=(
            "1 = HTF/LTF Kongruenzfilter aktiv (sicherer, weniger Trades), "
            "0 = Kongruenzfilter aus (aggressiver, mehr Trades, Standard)."
        ),
    )
    parser.add_argument(
        "--two_stage_allow_neutral_htf",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "1 = neutraler H1-Bias darf starke M5-Entries trotzdem zulassen (Standard), "
            "0 = neutraler H1-Bias blockiert aktive M5-Signale."
        ),
    )
    parser.add_argument(
        "--regime_filter",
        type=str,
        default="0,1,2,3",
        help=(
            "Komma-getrennte Regime-Nummern (Standard: '0,1,2,3'). "
            "0=Seitwärts, 1=Aufwärtstrend, 2=Abwärtstrend, 3=Hohe Vola. "
            "Paper-Test-Phase: Standardmäßig alle Regime erlaubt für mehr Feedback."
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
        "--log_dir",
        default="",
        help=(
            "Optionaler absoluter Log-Ordner. Überschreibt MT5_TRADING_LOG_DIR und "
            "den Standardpfad. Ideal für dedizierte Testläufe wie Test 128."
        ),
    )
    parser.add_argument(
        "--log_subdir",
        default="",
        help=(
            "Optionaler Unterordner unter BASE_DIR/logs. Beispiel: paper_test128. "
            "Wird nur genutzt wenn --log_dir leer ist."
        ),
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
    parser.add_argument(
        "--tp_pct",
        type=float,
        default=0.006,
        help=(
            "Take-Profit in Dezimal (Standard: 0.006 = 0.6%%). "
            "Wird als RRR-Faktor relativ zum SL verwendet wenn ATR-SL aktiv. "
            "Backtest-Optimum: 0.006 (RRR 2:1 mit SL=0.3%%)."
        ),
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=0.003,
        help=(
            "Stop-Loss in Dezimal (Standard: 0.003 = 0.3%%). "
            "Fallback wenn ATR-SL deaktiviert ist. "
            "Bei ATR-SL: RRR wird aus tp_pct/sl_pct berechnet (z.B. 0.006/0.003 = 2:1)."
        ),
    )
    parser.add_argument(
        "--two_stage_cooldown_bars",
        type=int,
        default=STANDARD_TWO_STAGE_COOLDOWN_BARS,
        help=(
            "Cooldown in Bars nach einem Trade (Standard: 3). "
            "Verhindert Overtrading bei dauerhaftem Signal. "
            "Bei M5: 3 Bars = 15 Minuten Pause zwischen Trades."
        ),
    )
    parser.add_argument(
        "--startup_observation_bars",
        type=int,
        default=STANDARD_STARTUP_OBSERVATION_BARS,
        help=(
            "Anzahl neuer Kerzen direkt nach Start, die erst nur beobachtet werden (Standard: 5). "
            "Signale werden geloggt, aber noch nicht gehandelt."
        ),
    )
    args = parser.parse_args()

    # ---- Logging-Ziel möglichst früh fixieren ----
    configure_logging(
        _resolve_log_dir(
            BASE_DIR,
            cli_log_dir=args.log_dir,
            cli_log_subdir=args.log_subdir,
        )
    )

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
            f"  scp SERVER:/mnt/1Tb-Data/XGBoost-LightGBM/models/"
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
                with open(meta_pfad, "r", encoding="utf-8") as f:
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
                "allow_neutral_htf_entries": bool(args.two_stage_allow_neutral_htf),
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
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        cooldown_bars=args.two_stage_cooldown_bars,
        startup_observation_bars=args.startup_observation_bars,
    )

    # MT5 Verbindung beenden
    if MT5_VERFUEGBAR and _mt5_api().terminal_info() is not None:
        _mt5_api().shutdown()
        logger.info("MT5-Verbindung getrennt.")


if __name__ == "__main__":
    main()
