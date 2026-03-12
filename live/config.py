"""
config.py – Gemeinsame Konfiguration, Konstanten und optionale Imports

Alle anderen Module (indicators, feature_builder, etc.) importieren aus dieser Datei.
Läuft auf: Windows 11 Laptop
"""

# Standard-Bibliotheken
import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

# Logging
logger = logging.getLogger(__name__)

# ============================================================
# Pfade (relativ zum Skript-Verzeichnis von live_trader.py)
# ============================================================

BASE_DIR = Path(__file__).parent.parent  # live/ → Projekt-Root
MODEL_DIR = BASE_DIR / "models"
# LOG_DIR wird beim Start über configure_logging() gesetzt (mutable!)
LOG_DIR: Path = BASE_DIR / "logs"

# ============================================================
# Optionale Bibliotheken (plattformabhängig)
# ============================================================

# HMM-Regime-Detection (optional)
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

# Pylance darf mt5 nicht auf None festnageln
mt5 = cast(Any, mt5)

# pandas_ta – für ADX-Berechnung
try:
    import pandas_ta  # noqa: F401  # pylint: disable=unused-import
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# Two-Stage Signal (Shadow-Mode)
try:
    from two_stage_signal import (
        modelle_laden as two_stage_modelle_laden,
        zwei_stufen_signal,
    )
    TWO_STAGE_VERFUEGBAR = True
except ImportError:
    TWO_STAGE_VERFUEGBAR = False

    def two_stage_modelle_laden(*args: Any, **kwargs: Any) -> tuple[object, object]:
        """Fallback wenn Two-Stage-Modul nicht verfügbar."""
        raise ImportError("two_stage_signal nicht verfügbar")

    def zwei_stufen_signal(*args: Any, **kwargs: Any) -> Any:
        """Fallback wenn Two-Stage-Modul nicht verfügbar."""
        raise ImportError("two_stage_signal nicht verfügbar")

# ============================================================
# Trading-Parameter
# ============================================================

TP_PCT = 0.003           # Take-Profit: 0.3%
SL_PCT = 0.003           # Stop-Loss: 0.3% (Fallback ohne ATR-SL)
LOT = 0.01               # Minimale Lot-Größe
MAX_OFFENE_TRADES = 1    # Max. 1 offene Position pro Symbol
MAGIC_NUMBER = 20260101  # Eindeutige Kennung für ML-Trades in MT5

# ATR-basiertes Stop-Loss
ATR_SL_ENABLED = True
ATR_SL_FAKTOR = 1.5

# Phase 7 Paper-Defaults
STANDARD_SCHWELLE = 0.45
STANDARD_TWO_STAGE_COOLDOWN_BARS = 3
REGIME_ADX_TREND_SCHWELLE = 18.0
REGIME_HIGH_VOL_FAKTOR = 1.8
STANDARD_TWO_STAGE_ALLOW_NEUTRAL_HTF = True
STANDARD_STARTUP_OBSERVATION_BARS = 5

# Kill-Switch
KILL_SWITCH_DD_DEFAULT = 0.15

# Heartbeat-Logging
HEARTBEAT_LOG_DEFAULT = True

# MT5-Zeitkorrektur
MT5_FUTURE_TIMESTAMP_GRACE_MINUTES = 2
MT5_MAX_AUTO_UTC_SHIFT_HOURS = 14

# Feature-Berechnung
N_BARREN = 500  # SMA200 + MTF-Buffer

# ============================================================
# Zeitrahmen-Konfiguration
# ============================================================

TIMEFRAME_CONFIG = {
    "H1": {"mt5_name": "TIMEFRAME_H1", "bars_per_hour": 1, "minutes_per_bar": 60},
    "M30": {"mt5_name": "TIMEFRAME_M30", "bars_per_hour": 2, "minutes_per_bar": 30},
    "M15": {"mt5_name": "TIMEFRAME_M15", "bars_per_hour": 4, "minutes_per_bar": 15},
    "M5": {"mt5_name": "TIMEFRAME_M5", "bars_per_hour": 12, "minutes_per_bar": 5},
}

# ============================================================
# Symbole & Regime
# ============================================================

SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
AKTIVE_SYMBOLE = ["USDCAD", "USDJPY"]  # Operative Policy

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

# ============================================================
# MT5-Laufzeit-Status (mutable Dict – kein global-Statement nötig)
# ============================================================

_MT5_RUNTIME_STATE: dict[str, Any] = {
    "credentials": {},
    "ipc_fail_count": 0,
    "kerzen_debug_last_ts": 0.0,
}
_MT5_RECONNECT_AFTER: int = 3

# MT5 Common-Files Sync
MT5_COMMON_FILES_ENV = "MT5_COMMON_FILES_DIR"
MT5_COMMON_SYNC_RETRIES = 3
MT5_COMMON_SYNC_RETRY_DELAY_SEC = 0.2

# ============================================================
# Feature-Spalten (exakte Trainings-Reihenfolge, identisch mit train_model.py)
# ============================================================

AUSSCHLUSS_SPALTEN = {
    "open", "high", "low", "close", "volume", "spread",
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "atr_14",
    "bb_upper", "bb_mid", "bb_lower", "obv", "pdh", "pdl", "pwh", "pwl", "label",
}

FEATURE_SPALTEN = [
    "price_sma20_ratio", "price_sma50_ratio", "price_sma200_ratio",
    "sma_20_50_cross", "sma_50_200_cross", "ema_cross",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14", "rsi_centered",
    "stoch_k", "stoch_d", "stoch_cross",
    "williams_r", "roc_10",
    "atr_pct", "bb_width", "bb_pct", "hist_vol_20",
    "obv_zscore", "volume_roc", "volume_ratio",
    "return_1h", "return_4h", "return_24h",
    "candle_body", "upper_wick", "lower_wick", "candle_dir", "hl_range",
    "trend_h4", "rsi_h4", "trend_d1",
    "hour", "day_of_week",
    "session_london", "session_ny", "session_asia", "session_overlap",
    "killzone_london_open", "killzone_ny_open", "killzone_asia_open",
    "dist_pdh_pct", "dist_pdl_pct", "dist_pwh_pct", "dist_pwl_pct", "near_key_level",
    "fvg_bullish", "fvg_bearish", "fvg_gap_pct",
    "bos_bull", "bos_bear", "mss_bull", "mss_bear", "structure_bias",
    "adx_14", "market_regime",
    "market_regime_hmm",
    "fear_greed_value", "fear_greed_class", "btc_funding_rate",
]

# ============================================================
# Hilfsfunktion: HMM-Logging unterdrücken
# ============================================================


@contextmanager
def mute_hmmlearn_convergence_logs() -> Any:
    """Unterdrückt hmmlearn-Konvergenzmeldungen nur während des Fits."""
    hmm_logger = logging.getLogger("hmmlearn.base")
    previous_level = hmm_logger.level
    try:
        hmm_logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Model is not converging.*")
            yield
    finally:
        hmm_logger.setLevel(previous_level)


# ============================================================
# MT5 API Zugriff (typsicher)
# ============================================================


def mt5_api() -> Any:
    """Liefert das MT5-Modul typrobust oder wirft einen klaren Fehler."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 ist nicht verfügbar")
    return cast(Any, mt5)
