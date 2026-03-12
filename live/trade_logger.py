"""
trade_logger.py – Trade- und Signal-Logging in CSV + SQLite + MT5 Common/Files

Schreibt alle Signal- und Close-Events parallel in:
  1. CSV-Dateien (für MT5-Dashboard-Kompatibilität)
  2. SQLite-Datenbank (für robuste Abfragen und KPI-Berichte)

Läuft auf: Windows 11 Laptop
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config
import db_manager
from mt5_connector import mirror_csv_to_mt5_common

logger = logging.getLogger(__name__)

# Schema beim ersten Import sicherstellen (idempotent, kein Overhead)
try:
    db_manager.schema_erstellen()
except Exception as _e:  # pylint: disable=broad-exception-caught
    logger.warning("SQLite-Schema konnte nicht erstellt werden: %s", _e)


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
    Schreibt Signal-/Heartbeat-Events in CSV und SQLite.

    Schreibt zusätzlich eine Kopie in den MT5 Common/Files-Ordner,
    damit das LiveSignalDashboard.mq5 die Daten lesen kann.

    Args:
        symbol:        Handelssymbol
        richtung:      2=Long, -1=Short, 0=Kein Trade (Heartbeat)
        prob:          Signal-Wahrscheinlichkeit
        regime:        Markt-Regime (0–3)
        paper_trading: True = Paper-Modus aktiv
        entry_price:   Einstiegspreis (0 bei Heartbeat)
        sl_price:      Stop-Loss-Preis (0 bei Heartbeat)
        tp_price:      Take-Profit-Preis (0 bei Heartbeat)
        htf_bias:      HTF-Bias aus Two-Stage (None = kein Two-Stage)
        ltf_signal:    LTF-Signal aus Two-Stage (None = kein Two-Stage)
    """
    # LOG_DIR wird zur Laufzeit aus config gelesen (nach configure_logging() gesetzt)
    log_pfad = config.LOG_DIR / f"{symbol}_signals.csv"

    eintrag = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "richtung": "Long" if richtung == 2 else "Short" if richtung == -1 else "Kein",
        "signal": richtung,
        "prob": round(prob, 4),
        "regime": regime,
        "regime_name": config.REGIME_NAMEN.get(regime, "?"),
        "paper_trading": paper_trading,
        "modus": "PAPER" if paper_trading else "LIVE",
        "entry_price": round(entry_price, 5),
        "sl_price": round(sl_price, 5),
        "tp_price": round(tp_price, 5),
        "htf_bias": htf_bias if htf_bias is not None else "",
        "ltf_signal": ltf_signal if ltf_signal is not None else "",
    }

    # 1) CSV schreiben (MT5-Dashboard-Kompatibilität)
    df_log = pd.DataFrame([eintrag])
    df_log.to_csv(log_pfad, mode="a", header=not log_pfad.exists(), index=False)

    # Komplette aktive Signal-CSV nach MT5 Common/Files spiegeln
    mirror_csv_to_mt5_common(log_pfad, f"{symbol}_signals.csv")

    # 2) SQLite schreiben (parallel, Fehler werden nur geloggt)
    db_manager.signal_einfuegen(
        symbol=symbol,
        signal=richtung,
        prob=prob,
        regime=regime,
        paper_trading=paper_trading,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        htf_bias=htf_bias,
        ltf_signal=ltf_signal,
    )


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
    paper_trading: bool = False,
    sl_price: float = 0.0,
    tp_price: float = 0.0,
) -> None:
    """
    Schreibt ein CLOSE-Event in CSV und SQLite.

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
        htf_bias:       HTF-Bias bei Eröffnung (None = kein Two-Stage)
        ltf_signal:     LTF-Signal bei Eröffnung (None = kein Two-Stage)
        paper_trading:  True wenn simulierter Paper-Close
        sl_price:       Ursprünglicher Stop-Loss-Preis
        tp_price:       Ursprünglicher Take-Profit-Preis
    """
    log_pfad = config.LOG_DIR / f"{symbol}_closes.csv"

    eintrag = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "richtung": "CLOSE-Long" if richtung == 2 else "CLOSE-Short",
        "signal": 0,
        "prob": 0.0,
        "regime": -1,
        "regime_name": "CLOSE",
        "paper_trading": paper_trading,
        "modus": "CLOSE_PAPER" if paper_trading else "CLOSE_LIVE",
        "entry_price": round(entry_price, 5),
        "sl_price": round(sl_price, 5),
        "tp_price": round(tp_price, 5),
        "htf_bias": htf_bias if htf_bias is not None else "",
        "ltf_signal": ltf_signal if ltf_signal is not None else "",
        "exit_price": round(exit_price, 5),
        "pnl_pips": round(pnl_pips, 1),
        "pnl_money": round(pnl_money, 2),
        "close_grund": close_grund,
        "dauer_min": dauer_minuten,
        "ticket": ticket,
    }

    # 1) CSV schreiben
    df_log = pd.DataFrame([eintrag])
    df_log.to_csv(log_pfad, mode="a", header=not log_pfad.exists(), index=False)

    mirror_csv_to_mt5_common(log_pfad, f"{symbol}_closes.csv")

    # 2) SQLite schreiben
    db_manager.trade_close_einfuegen(
        symbol=symbol,
        ticket=ticket,
        richtung=richtung,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pips=pnl_pips,
        pnl_money=pnl_money,
        close_grund=close_grund,
        dauer_minuten=dauer_minuten,
        htf_bias=htf_bias,
        ltf_signal=ltf_signal,
    )

    logger.info(
        f"[{symbol}] CLOSE-Event geloggt: Ticket={ticket} | "
        f"PnL={pnl_money:+.2f} USD ({pnl_pips:+.1f} Pips) | "
        f"Grund={close_grund} | Dauer={dauer_minuten} Min | "
        f"Modus={'PAPER' if paper_trading else 'LIVE'}"
    )
