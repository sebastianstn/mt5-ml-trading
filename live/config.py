"""
config.py – Gemeinsame Konfiguration, Konstanten und optionale Imports

Alle anderen Module (indicators, feature_builder, etc.) importieren aus dieser Datei.
Läuft auf: Windows 11 Laptop ODER Linux Mint Laptop (MT5 via Wine + mt5linux)
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

# Explizite Referenz, damit statische Analyse den optionalen Export als genutzt erkennt
_GAUSSIAN_HMM_EXPORT = GaussianHMM

# MetaTrader5 – Windows nativ ODER Linux via mt5linux (RPyC-Bridge über Wine)
_MT5_IS_LINUX_BRIDGE = False
_MT5LinuxClass: Any = None

try:
    import MetaTrader5 as mt5  # type: ignore

    MT5_VERFUEGBAR = True
except ImportError:
    # Linux: mt5linux als Bridge zu Wine-MT5 über RPyC
    try:
        from mt5linux import MetaTrader5 as _MT5LinuxClass  # type: ignore

        mt5 = None  # type: ignore  # Lazy Init in mt5_api()
        MT5_VERFUEGBAR = True
        _MT5_IS_LINUX_BRIDGE = True
        logger.info("mt5linux erkannt – RPyC-Bridge-Modus (Wine)")
    except ImportError:
        MT5_VERFUEGBAR = False
        mt5 = None  # type: ignore

# Pylance darf mt5 nicht auf None festnageln
if not _MT5_IS_LINUX_BRIDGE:
    mt5 = cast(Any, mt5)

# pandas_ta – für ADX-Berechnung
try:
    import pandas_ta  # noqa: F401  # pylint: disable=unused-import

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

# Two-Stage Signal (Shadow-Mode)
try:
    from live.two_stage_signal import (
        modelle_laden as two_stage_modelle_laden,
        zwei_stufen_signal,
    )

    TWO_STAGE_VERFUEGBAR = True
except ImportError:
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


# Explizite Referenz, damit statische Analyse den optionalen Export als genutzt erkennt
_TWO_STAGE_EXPORTS = (two_stage_modelle_laden, zwei_stufen_signal)


# ============================================================
# Trading-Parameter
# ============================================================

TP_PCT = 0.003  # Take-Profit: 0.3%
SL_PCT = 0.003  # Stop-Loss: 0.3% (Fallback ohne ATR-SL)
LOT = 0.01  # Minimale Lot-Größe
MAX_OFFENE_TRADES = 1  # Max. 1 offene Position pro Symbol
MAGIC_NUMBER = 20260101  # Eindeutige Kennung für ML-Trades in MT5

# ATR-basiertes Stop-Loss
ATR_SL_ENABLED = True
ATR_SL_FAKTOR = 2.0  # Erhöht von 1.5 auf 2.0 → SL nicht enger als 2×ATR (vermeidet Rauschen-Stopouts)

# Spread-Filter (NEU – verhindert Trades bei zu hohem Spread)
MAX_SPREAD_PIPS = 2.0  # Maximaler Spread in Pips – darüber wird kein Trade eröffnet

# Phase 7 Paper-Defaults
STANDARD_SCHWELLE = 0.45
STANDARD_TWO_STAGE_COOLDOWN_BARS = 3
REGIME_ADX_TREND_SCHWELLE = 18.0
REGIME_HIGH_VOL_FAKTOR = 1.8
STANDARD_TWO_STAGE_ALLOW_NEUTRAL_HTF = (
    True  # K1-Testlauf: Neutral-HTF darf LTF-Entries passieren (mehr Aktivität)
)
STANDARD_TWO_STAGE_MODE = "primary"  # M1: Two-Stage als Hauptsignal statt Shadow
STANDARD_TWO_STAGE_HTF_SCHWELLE = 0.35  # M3: HTF separat kalibrieren
STANDARD_TWO_STAGE_LTF_SCHWELLE = 0.50  # M3: LTF separat kalibrieren
STANDARD_STARTUP_OBSERVATION_BARS = 5

# Regime-abhängige Schwellenerhöhung (Seitwärts-Märkte brauchen stärkere Signale)
REGIME_SEITWAERTS_SCHWELLE_AUFSCHLAG = (
    0.08  # Bei Regime 0: Schwelle + 0.08 (z.B. 0.45 → 0.53)
)

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


def mt5_runtime_state_get(key: str, default: Any = None) -> Any:
    """Liest einen Wert aus dem internen MT5-Runtime-State typrobust aus."""
    return _MT5_RUNTIME_STATE.get(key, default)


def mt5_runtime_state_set(key: str, value: Any) -> None:
    """Schreibt einen Wert in den internen MT5-Runtime-State."""
    _MT5_RUNTIME_STATE[key] = value


# MT5 Common-Files Sync
MT5_COMMON_FILES_ENV = "MT5_COMMON_FILES_DIR"
MT5_COMMON_SYNC_RETRIES = 3
MT5_COMMON_SYNC_RETRY_DELAY_SEC = 0.2

# ============================================================
# Feature-Spalten (exakte Trainings-Reihenfolge, identisch mit train_model.py)
# ============================================================

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

# mt5linux RPyC-Bridge Instanz (Lazy Init – Verbindung erst bei erster Nutzung)
_mt5_linux_instance: Any = None
_MT5_RPYC_MAX_RETRIES = 3  # Max. Reconnect-Versuche bevor Exception


def _mt5_linux_connect() -> Any:
    """Erstellt eine neue mt5linux RPyC-Verbindung zu Wine-Python."""
    host = os.environ.get("MT5_RPYC_HOST", "localhost")
    port = int(os.environ.get("MT5_RPYC_PORT", "18812"))
    try:
        instance = _MT5LinuxClass(host=host, port=port)
        logger.info(f"mt5linux RPyC-Bridge verbunden: {host}:{port}")
        return instance
    except Exception as exc:
        raise RuntimeError(
            f"mt5linux RPyC-Bridge nicht erreichbar ({host}:{port}): {exc}\n"
            "Ist der RPyC-Server gestartet? → bash scripts/start_mt5_rpyc_server.sh"
        ) from exc


def _mt5_linux_is_alive() -> bool:
    """Prüft ob die RPyC-Verbindung noch lebt (Ping)."""
    if _mt5_linux_instance is None:
        return False
    try:
        conn = _mt5_linux_instance._MetaTrader5__conn  # type: ignore[attr-defined]
        # Schneller Check: closed-Flag prüfen
        if conn.closed:
            return False
        # Aktiver Ping – wirft EOFError wenn Verbindung tot
        conn.ping()
        return True
    except Exception:
        return False


def mt5_rpyc_reconnect() -> None:
    """Erzwingt eine neue RPyC-Verbindung (z.B. nach 'connection closed by peer')."""
    global _mt5_linux_instance  # noqa: PLW0603
    if _mt5_linux_instance is not None:
        try:
            _mt5_linux_instance._MetaTrader5__conn.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    _mt5_linux_instance = None
    logger.info("RPyC-Verbindung zurückgesetzt – nächster API-Aufruf verbindet neu")


def mt5_api() -> Any:
    """Liefert das MT5-Modul (Windows) oder mt5linux-Instanz (Linux) typrobust.

    Auf Linux wird die RPyC-Verbindung beim ersten Aufruf aufgebaut (Lazy Init).
    Bei Verbindungsabbruch ('connection closed by peer') wird automatisch reconnected.
    Konfiguration über Umgebungsvariablen: MT5_RPYC_HOST, MT5_RPYC_PORT.
    """
    global _mt5_linux_instance  # noqa: PLW0603 – Lazy Init braucht globalen Zustand

    if _MT5_IS_LINUX_BRIDGE:
        # Prüfe ob existing Verbindung noch lebt
        if _mt5_linux_instance is not None and not _mt5_linux_is_alive():
            logger.warning("RPyC-Verbindung verloren – versuche Reconnect...")
            _mt5_linux_instance = None

        if _mt5_linux_instance is None:
            import time as _time
            for attempt in range(1, _MT5_RPYC_MAX_RETRIES + 1):
                try:
                    _mt5_linux_instance = _mt5_linux_connect()
                    break
                except RuntimeError:
                    if attempt == _MT5_RPYC_MAX_RETRIES:
                        raise
                    logger.warning(f"RPyC-Reconnect fehlgeschlagen (Versuch {attempt}/{_MT5_RPYC_MAX_RETRIES}), warte 5s...")
                    _time.sleep(5)
        return cast(Any, _mt5_linux_instance)

    if mt5 is None:
        raise RuntimeError("MetaTrader5 ist nicht verfügbar")
    return cast(Any, mt5)
