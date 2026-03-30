"""
paper_trading.py – Paper-Trading PnL-Berechnung und Trade-Tracking

Verfolgt simulierte Trades gegen abgeschlossene OHLC-Kerzen.
Konservative Logik: Bei gleichzeitigem TP/SL-Touch in einer Kerze → SL angenommen.

Läuft auf: Windows 11 Laptop
"""

# pylint: disable=logging-fstring-interpolation

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, cast

import pandas as pd

try:
    from live import config  # Import als Paket (z.B. pytest, externe Aufrufe)
    from live.trade_logger import trade_close_loggen
except ImportError:
    import config  # Import direkt aus live/-Verzeichnis
    from trade_logger import trade_close_loggen

logger = logging.getLogger(__name__)


def _letzte_geschlossene_kerze(df: pd.DataFrame) -> pd.Series:
    """
    Liefert konsistent die letzte vollständig geschlossene Kerze.

    Args:
        df: Markt- oder Feature-DataFrame mit DatetimeIndex.

    Returns:
        Series der letzten geschlossenen Kerze.

    Raises:
        ValueError: Wenn weniger als zwei Zeilen vorhanden sind.
    """
    if len(df) < 2:
        raise ValueError(
            "Für die letzte geschlossene Kerze werden mindestens 2 Zeilen benötigt"
        )
    return df.iloc[-2]


def _paper_pnl_money_berechnen(
    entry_price: float,
    exit_price: float,
    richtung: int,
    lot: float,
) -> float:
    """
    Berechnet eine einfache Paper-PnL in Quote-Currency.

    Args:
        entry_price: Einstiegspreis.
        exit_price: Ausstiegspreis.
        richtung: 2=Long, -1=Short.
        lot: Lot-Größe des Trades.

    Returns:
        PnL als Geldwert in Quote-Currency.
    """
    contract_size = 100000.0
    price_diff = exit_price - entry_price if richtung == 2 else entry_price - exit_price
    return price_diff * contract_size * lot


def _pip_faktor(symbol: str) -> float:
    """
    Gibt den Pip-Multiplikator für ein Symbol zurück.

    JPY-Paare: 1 Pip = 0.01, alle anderen: 1 Pip = 0.0001.
    """
    if "JPY" in symbol.upper():
        return 100.0
    return 10000.0


def paper_trade_pruefen_und_loggen(
    symbol: str,
    letzter_trade: Optional[dict[str, Any]],
    markt_df: pd.DataFrame,
    paper_kapital: float,
    timeframe: str,
) -> tuple[Optional[dict[str, Any]], float]:
    """
    Prüft einen offenen Paper-Trade gegen die seit Entry abgeschlossenen Kerzen.

    Konservative Logik: TP und SL in derselben Kerze → SL angenommen.

    Args:
        symbol: Handelssymbol.
        letzter_trade: Aktuell verfolgter Paper-Trade oder None.
        markt_df: Rohes OHLCV-DataFrame des aktiven Zeitrahmens.
        paper_kapital: Aktueller simulierter Kontostand.
        timeframe: Aktiver Zeitrahmen (z.B. M5 oder H1).

    Returns:
        Tuple aus aktualisiertem Trade-Zustand und aktualisiertem Paper-Kapital.
    """
    if letzter_trade is None or markt_df.empty or len(markt_df) < 2:
        return letzter_trade, paper_kapital

    entry_bar_time = letzter_trade.get("entry_bar_time")
    if not isinstance(entry_bar_time, datetime):
        return letzter_trade, paper_kapital

    letzte_geschlossene_zeit = cast(pd.Timestamp, markt_df.index[-2]).to_pydatetime()
    if letzte_geschlossene_zeit <= entry_bar_time:
        return letzter_trade, paper_kapital

    bars_to_check = markt_df.loc[
        (markt_df.index > pd.Timestamp(entry_bar_time))
        & (markt_df.index <= pd.Timestamp(letzte_geschlossene_zeit))
    ]
    if bars_to_check.empty:
        return letzter_trade, paper_kapital

    richtung = int(letzter_trade.get("richtung", 0))
    entry_price = float(letzter_trade.get("entry_price", 0.0))
    sl_price = float(letzter_trade.get("sl_price", 0.0))
    tp_price = float(letzter_trade.get("tp_price", 0.0))
    lot = float(letzter_trade.get("lot", config.LOT))
    open_zeit = cast(
        datetime, letzter_trade.get("open_zeit", datetime.now(timezone.utc))
    )
    bar_minutes = config.TIMEFRAME_CONFIG.get(timeframe, config.TIMEFRAME_CONFIG["H1"])[
        "minutes_per_bar"
    ]

    for bar_zeit, bar in bars_to_check.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if richtung == 2:
            sl_hit = bar_low <= sl_price
            tp_hit = bar_high >= tp_price
        else:
            sl_hit = bar_high >= sl_price
            tp_hit = bar_low <= tp_price

        if not sl_hit and not tp_hit:
            continue

        if sl_hit and tp_hit:
            close_grund = "SL"
            logger.warning(
                f"[{symbol}] [PAPER] TP und SL in derselben Kerze – konservativ als SL gewertet"
            )
        elif tp_hit:
            close_grund = "TP"
        else:
            close_grund = "SL"

        exit_price = tp_price if close_grund == "TP" else sl_price
        pnl_pips = (
            (exit_price - entry_price) * _pip_faktor(symbol)
            if richtung == 2
            else (entry_price - exit_price) * _pip_faktor(symbol)
        )
        pnl_money = _paper_pnl_money_berechnen(
            entry_price=entry_price,
            exit_price=exit_price,
            richtung=richtung,
            lot=lot,
        )

        bar_close_time = cast(pd.Timestamp, bar_zeit).to_pydatetime() + timedelta(
            minutes=bar_minutes
        )
        dauer_min = max(int((bar_close_time - open_zeit).total_seconds() / 60), 0)

        trade_close_loggen(
            symbol=symbol,
            ticket=int(letzter_trade.get("position_ticket", 0) or 0),
            richtung=richtung,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_money=pnl_money,
            close_grund=close_grund,
            dauer_minuten=dauer_min,
            htf_bias=cast(Optional[int], letzter_trade.get("htf_bias")),
            ltf_signal=cast(Optional[int], letzter_trade.get("ltf_signal")),
            paper_trading=True,
            sl_price=sl_price,
            tp_price=tp_price,
        )
        logger.info(
            f"[{symbol}] [PAPER] Trade geschlossen | Grund={close_grund} | "
            f"Exit={exit_price:.5f} | PnL={pnl_money:+.2f}"
        )
        return None, paper_kapital + pnl_money

    return letzter_trade, paper_kapital
