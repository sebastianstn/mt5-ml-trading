"""
mt5_connector.py – MetaTrader 5 Verbindung, Datenabruf und Order-Ausführung

Kapselt alle MT5-Interaktionen: Verbindung, Auto-Reconnect, Datenabruf, Orders.
Läuft auf: Windows 11 Laptop (MetaTrader5-Bibliothek NUR auf Windows!)
"""

# pylint: disable=logging-fstring-interpolation,protected-access

import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

try:
    from live import config  # Import als Paket (z.B. pytest, externe Aufrufe)
except ImportError:
    import config  # Import direkt aus live/-Verzeichnis

logger = logging.getLogger(__name__)


# ============================================================
# MT5 Common-Files Sync (für MT5 Dashboard)
# ============================================================


def _resolve_mt5_common_files_dir() -> Optional[Path]:
    """Ermittelt den MT5-Common-Files-Ordner über Override, APPDATA (Windows) oder Wine (Linux)."""
    # Manueller Override hat höchste Priorität (beide Plattformen)
    override_dir = os.environ.get(config.MT5_COMMON_FILES_ENV, "").strip()
    if override_dir:
        return Path(override_dir).expanduser()

    # Ein explizites APPDATA soll auch unter Linux-Tests/Hybrid-Setups Vorrang haben.
    appdata_dir = os.environ.get("APPDATA", "").strip()
    if appdata_dir:
        return Path(appdata_dir) / "MetaQuotes" / "Terminal" / "Common" / "Files"

    # Linux: MT5 läuft via Wine → Pfad im Wine-Prefix
    if sys.platform.startswith("linux"):
        wine_user = os.environ.get("USER", "")
        wine_prefix = Path(os.environ.get("WINEPREFIX", str(Path.home() / ".wine")))
        wine_appdata = wine_prefix / "drive_c" / "users" / wine_user / "AppData" / "Roaming"
        return wine_appdata / "MetaQuotes" / "Terminal" / "Common" / "Files"

    return None


def mirror_csv_to_mt5_common(local_csv_path: Path, target_file_name: str) -> bool:
    """
    Spiegelt eine lokale CSV robust nach MT5 Common/Files.

    Args:
        local_csv_path: Lokale Quell-CSV im aktiven Log-Verzeichnis.
        target_file_name: Dateiname im MT5 Common/Files-Verzeichnis.

    Returns:
        True wenn die Spiegelung erfolgreich war.
    """
    if not local_csv_path.exists():
        logger.warning(
            "MT5-Common-Sync übersprungen: lokale CSV fehlt (%s)", local_csv_path
        )
        return False

    mt5_common_dir = _resolve_mt5_common_files_dir()
    if mt5_common_dir is None:
        logger.warning(
            "MT5-Common-Sync übersprungen: weder %s noch APPDATA verfügbar",
            config.MT5_COMMON_FILES_ENV,
        )
        return False

    try:
        mt5_common_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "MT5-Common-Ordner konnte nicht erstellt werden (%s): %s",
            mt5_common_dir,
            exc,
        )
        return False

    target_csv_path = mt5_common_dir / target_file_name
    last_error: Optional[OSError] = None
    for attempt in range(1, config.MT5_COMMON_SYNC_RETRIES + 1):
        try:
            shutil.copy2(local_csv_path, target_csv_path)
            return True
        except OSError as exc:
            last_error = exc
            if attempt < config.MT5_COMMON_SYNC_RETRIES:
                time.sleep(config.MT5_COMMON_SYNC_RETRY_DELAY_SEC)

    logger.warning(
        "MT5-Common-Sync fehlgeschlagen nach %d Versuchen: %s -> %s | Fehler: %s",
        config.MT5_COMMON_SYNC_RETRIES,
        local_csv_path,
        target_csv_path,
        last_error,
    )
    return False


# ============================================================
# MT5 Verbindung & Reconnect
# ============================================================


def mt5_verbinden(server: str, login: int, password: str, pfad: str = "") -> bool:
    """
    Verbindet mit dem MT5-Terminal.

    Args:
        server:   Broker-Server (z.B. "ICMarkets-Demo")
        login:    Kontonummer
        password: Passwort
        pfad:     Optionaler Pfad zur terminal64.exe

    Returns:
        True bei Erfolg, False bei Fehler.
    """
    if not config.MT5_VERFUEGBAR:
        logger.warning("MetaTrader5 nicht installiert – nur Paper-Trading möglich!")
        return False

    mt5_api = config.mt5_api()

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
    config._MT5_RUNTIME_STATE["credentials"] = {
        "server": server,
        "login": login,
        "password": password,
        "pfad": pfad,
    }
    config._MT5_RUNTIME_STATE["ipc_fail_count"] = 0

    konto = mt5_api.account_info()
    logger.info(
        f"MT5 verbunden | Server: {server} | Konto: {konto.login} | Saldo: {konto.balance:.2f} {konto.currency}"
    )
    return True


def mt5_reconnect() -> bool:
    """
    Versucht die MT5-IPC-Verbindung wiederherzustellen.

    Returns:
        True wenn Reconnect erfolgreich.
    """
    if not config.MT5_VERFUEGBAR or not config._MT5_RUNTIME_STATE["credentials"]:
        logger.error("MT5-Reconnect nicht möglich – keine Zugangsdaten gespeichert")
        return False

    mt5_api = config.mt5_api()
    logger.warning("MT5 Auto-Reconnect: Verbindung wird wiederhergestellt ...")

    try:
        mt5_api.shutdown()
    except (RuntimeError, OSError):
        pass

    time.sleep(2)

    creds = config._MT5_RUNTIME_STATE["credentials"]
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
        config._MT5_RUNTIME_STATE["ipc_fail_count"] = 0
        konto = mt5_api.account_info()
        if konto:
            logger.info(
                f"MT5 Reconnect erfolgreich | Konto: {konto.login} | Saldo: {konto.balance:.2f}"
            )
        else:
            logger.info("MT5 Reconnect erfolgreich (Kontodaten nicht lesbar)")
        return True

    logger.error(f"MT5 Reconnect fehlgeschlagen: {mt5_api.last_error()}")
    return False


# ============================================================
# MT5 Hilfsfunktionen
# ============================================================


def mt5_timeframe_konstante(timeframe: str) -> Optional[int]:
    """Übersetzt den String-Zeitrahmen in die passende MT5-Konstante."""
    if not config.MT5_VERFUEGBAR:
        return None
    mt5_api = config.mt5_api()
    cfg = config.TIMEFRAME_CONFIG.get(timeframe)
    if cfg is None:
        return None
    return getattr(mt5_api, cfg["mt5_name"], None)


def n_barren_fuer_timeframe(timeframe: str) -> int:
    """Berechnet die benötigte Barrenanzahl für denselben Zeit-Buffer je Zeitrahmen."""
    bars_per_hour = config.TIMEFRAME_CONFIG.get(
        timeframe, config.TIMEFRAME_CONFIG["H1"]
    )["bars_per_hour"]
    return config.N_BARREN * bars_per_hour


def _infer_mt5_utc_shift_hours(raw_timestamp_utc: datetime) -> int:
    """Leitet einen Broker→UTC-Stundenshift ab, falls MT5-Zeiten in der Zukunft liegen."""
    normalized = raw_timestamp_utc
    now_utc = datetime.now(timezone.utc)
    future_limit = now_utc + timedelta(
        minutes=config.MT5_FUTURE_TIMESTAMP_GRACE_MINUTES
    )
    shift_hours = 0

    while (
        normalized > future_limit and shift_hours < config.MT5_MAX_AUTO_UTC_SHIFT_HOURS
    ):
        normalized -= timedelta(hours=1)
        shift_hours += 1

    return shift_hours


def _log_mt5_utc_shift(symbol: str, timeframe: str, shift_hours: int) -> None:
    """Loggt erkannte MT5-Zeitkorrekturen nur bei Änderungen."""
    shift_cache = cast(
        dict[str, int],
        config._MT5_RUNTIME_STATE.setdefault("utc_shift_by_stream", {}),
    )
    cache_key = f"{symbol}:{timeframe}"
    if shift_cache.get(cache_key) == shift_hours:
        return
    shift_cache[cache_key] = shift_hours
    if shift_hours > 0:
        logger.warning(
            f"[{symbol}] MT5-Zeitkorrektur aktiv: -{shift_hours}h auf {timeframe}"
        )


# ============================================================
# MT5 Datenabruf
# ============================================================


def mt5_daten_holen(
    symbol: str,
    timeframe: str = "H1",
    n_barren: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Holt die letzten Barren im gewählten Zeitrahmen von MT5.

    Args:
        symbol:   Handelssymbol
        timeframe: Zeitrahmen ("H1", "M30", "M15", "M5")
        n_barren: Anzahl der Barren (None = automatisch je Zeitrahmen)

    Returns:
        OHLCV DataFrame mit UTC-DatetimeIndex oder None bei Fehler.
    """
    if not config.MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – keine Live-Daten!")
        return None

    mt5_api = config.mt5_api()
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        logger.error(f"Unbekannter Zeitrahmen: {timeframe}")
        return None

    n_bars_effektiv = (
        n_barren if n_barren is not None else n_barren_fuer_timeframe(timeframe)
    )
    rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, n_bars_effektiv)

    if rates is None:
        fehler = mt5_api.last_error()
        config._MT5_RUNTIME_STATE["ipc_fail_count"] += 1
        logger.warning(
            f"[{symbol}] Keine Daten von MT5: {fehler} "
            f"(Fehler {config._MT5_RUNTIME_STATE['ipc_fail_count']}/{config._MT5_RECONNECT_AFTER})"
        )
        if config._MT5_RUNTIME_STATE["ipc_fail_count"] >= config._MT5_RECONNECT_AFTER:
            if mt5_reconnect():
                rates = mt5_api.copy_rates_from_pos(
                    symbol, tf_const, 0, n_bars_effektiv
                )
                if rates is not None:
                    logger.info(f"[{symbol}] Daten nach Reconnect erfolgreich geladen")
                else:
                    logger.error(f"[{symbol}] Auch nach Reconnect keine Daten")
                    return None
            else:
                return None
        else:
            return None

    config._MT5_RUNTIME_STATE["ipc_fail_count"] = 0

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    last_raw_ts = df["time"].dropna().max()
    if pd.notna(last_raw_ts):
        shift_hours = _infer_mt5_utc_shift_hours(last_raw_ts.to_pydatetime())
        if shift_hours > 0:
            df["time"] = df["time"] - pd.Timedelta(hours=shift_hours)
        _log_mt5_utc_shift(symbol, timeframe, shift_hours)

    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def mt5_letzte_kerze_uhrzeit(symbol: str, timeframe: str = "H1") -> Optional[datetime]:
    """
    Gibt die Öffnungszeit der letzten geschlossenen Kerze zurück.

    Returns:
        datetime (UTC) der letzten Kerzen-Eröffnung oder None.
    """
    if not config.MT5_VERFUEGBAR:
        return None

    mt5_api = config.mt5_api()
    tf_const = mt5_timeframe_konstante(timeframe)
    if tf_const is None:
        return None

    rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, 2)
    if rates is None or len(rates) < 2:
        fehler = mt5_api.last_error()
        config._MT5_RUNTIME_STATE["ipc_fail_count"] += 1
        logger.warning(
            f"[{symbol}] MT5 liefert keine Kerzen-Daten: {fehler} "
            f"(Fehler {config._MT5_RUNTIME_STATE['ipc_fail_count']}/{config._MT5_RECONNECT_AFTER})"
        )
        if config._MT5_RUNTIME_STATE["ipc_fail_count"] >= config._MT5_RECONNECT_AFTER:
            if mt5_reconnect():
                rates = mt5_api.copy_rates_from_pos(symbol, tf_const, 0, 2)
                if rates is not None and len(rates) >= 2:
                    config._MT5_RUNTIME_STATE["ipc_fail_count"] = 0
                    raw_timestamp = datetime.fromtimestamp(
                        int(rates[1]["time"]), tz=timezone.utc
                    )
                    shift_hours = _infer_mt5_utc_shift_hours(raw_timestamp)
                    _log_mt5_utc_shift(symbol, timeframe, shift_hours)
                    return raw_timestamp - timedelta(hours=shift_hours)
        return None

    config._MT5_RUNTIME_STATE["ipc_fail_count"] = 0
    raw_timestamp = datetime.fromtimestamp(int(rates[1]["time"]), tz=timezone.utc)
    shift_hours = _infer_mt5_utc_shift_hours(raw_timestamp)
    _log_mt5_utc_shift(symbol, timeframe, shift_hours)
    return raw_timestamp - timedelta(hours=shift_hours)


def mt5_offene_position(symbol: str) -> bool:
    """
    Prüft ob bereits eine offene ML-Position für das Symbol existiert.

    Returns:
        True wenn eine offene Position mit MAGIC_NUMBER existiert.
    """
    if not config.MT5_VERFUEGBAR:
        return False
    mt5_api = config.mt5_api()
    positionen = mt5_api.positions_get(symbol=symbol)
    if positionen is None:
        return False
    return any(p.magic == config.MAGIC_NUMBER for p in positionen)


def order_senden(
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
        lot:           Lot-Größe
        tp_pct:        Take-Profit in Dezimal
        sl_pct:        Stop-Loss in Dezimal
        paper_trading: True = nur loggen (kein echtes Geld!)

    Returns:
        Dict mit Order-Metadaten bei Erfolg, sonst None.
    """
    richtung_str = "LONG (Kaufen)" if richtung == 2 else "SHORT (Verkaufen)"

    if paper_trading:
        logger.info(
            f"[PAPER] {symbol} {richtung_str} | Lot={lot} | TP={tp_pct:.1%} | SL={sl_pct:.1%}"
        )
        return {
            "success": True,
            "deal_ticket": None,
            "position_ticket": None,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
        }

    if not config.MT5_VERFUEGBAR:
        logger.error("MT5 nicht verfügbar – Order nicht gesendet!")
        return None

    mt5_api = config.mt5_api()

    if not mt5_api.symbol_select(symbol, True):
        logger.error(f"Symbol {symbol} nicht verfügbar!")
        return None

    symbol_info = mt5_api.symbol_info(symbol)
    tick = mt5_api.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        logger.error(f"Symbol-Info für {symbol} nicht abrufbar!")
        return None

    if richtung == 2:  # Long: Buy
        order_type = mt5_api.ORDER_TYPE_BUY
        preis = tick.ask
        sl_price = round(preis * (1.0 - sl_pct), symbol_info.digits)
        tp_price = round(preis * (1.0 + tp_pct), symbol_info.digits)
    else:  # Short: Sell
        order_type = mt5_api.ORDER_TYPE_SELL
        preis = tick.bid
        sl_price = round(preis * (1.0 + sl_pct), symbol_info.digits)
        tp_price = round(preis * (1.0 - tp_pct), symbol_info.digits)

    request = {
        "action": mt5_api.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": preis,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": config.MAGIC_NUMBER,
        "comment": f"ML-Signal-{richtung_str[:4]}",
        "type_time": mt5_api.ORDER_TIME_GTC,
        "type_filling": mt5_api.ORDER_FILLING_IOC,
    }

    result = mt5_api.order_send(request)
    if result is None or result.retcode != mt5_api.TRADE_RETCODE_DONE:
        fehler_code = result.retcode if result else "None"
        fehler_msg = result.comment if result else "Keine Antwort"
        logger.error(
            f"[{symbol}] Order FEHLGESCHLAGEN: Code={fehler_code} | {fehler_msg}"
        )
        return None

    logger.info(
        f"[{symbol}] {richtung_str} ausgeführt | Lot={lot} | "
        f"Preis={result.price:.5f} | SL={sl_price:.5f} | TP={tp_price:.5f} | "
        f"Deal={result.deal} | Position={result.order}"
    )

    return {
        "success": True,
        "deal_ticket": int(result.deal),
        "position_ticket": int(result.order),
        "entry_price": float(result.price),
        "sl_price": sl_price,
        "tp_price": tp_price,
    }


def alle_positionen_schliessen(symbol: str) -> None:
    """
    Schließt alle offenen MT5-Positionen für dieses Symbol (nach Kill-Switch).

    Args:
        symbol: Handelssymbol
    """
    if not config.MT5_VERFUEGBAR:
        return

    mt5_api = config.mt5_api()
    positionen = mt5_api.positions_get(symbol=symbol)
    if not positionen:
        return

    logger.info(f"[{symbol}] Schließe {len(positionen)} offene Position(en) ...")
    for pos in positionen:
        tick = mt5_api.symbol_info_tick(symbol)
        if tick is None:
            continue

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
            "magic": config.MAGIC_NUMBER,
            "comment": "Kill-Switch",
            "type_time": mt5_api.ORDER_TIME_GTC,
            "type_filling": mt5_api.ORDER_FILLING_IOC,
        }
        result = mt5_api.order_send(request)
        if result.retcode == mt5_api.TRADE_RETCODE_DONE:
            logger.info(f"[{symbol}] Position {pos.ticket} geschlossen")
        else:
            logger.error(
                f"[{symbol}] Schließen fehlgeschlagen: Code={result.retcode} | {result.comment}"
            )
