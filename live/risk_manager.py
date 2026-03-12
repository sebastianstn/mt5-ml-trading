"""
risk_manager.py – Kill-Switch, Drawdown-Überwachung und Trade-Schließungs-Erkennung

Schützt das Kapital durch automatische Stopp-Logik und überwacht offene Positionen.
Läuft auf: Windows 11 Laptop
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config
from mt5_connector import alle_positionen_schliessen
from trade_logger import trade_close_loggen

logger = logging.getLogger(__name__)


def _pip_faktor(symbol: str) -> float:
    """
    Gibt den Pip-Multiplikator für ein Symbol zurück.

    JPY-Paare: 1 Pip = 0.01, alle anderen: 1 Pip = 0.0001.
    """
    if "JPY" in symbol.upper():
        return 100.0
    return 10000.0


def _close_grund_aus_deal(deal: Any) -> str:
    """
    Leitet den Schließungsgrund aus MT5-Deal-Feldern ab.

    Nutzt primär Deal-Reason, sekundär Entry-Typ. Kein PnL-basiertes Raten.

    Returns:
        TP, SL, SO, MANUAL, SYSTEM, OUT_BY oder UNKNOWN
    """
    mt5_api = config.mt5_api()
    reason = int(getattr(deal, "reason", -1))
    entry = int(getattr(deal, "entry", -1))

    if reason == int(getattr(mt5_api, "DEAL_REASON_TP", -999)):
        return "TP"
    if reason == int(getattr(mt5_api, "DEAL_REASON_SL", -999)):
        return "SL"
    if reason == int(getattr(mt5_api, "DEAL_REASON_SO", -999)):
        return "SO"
    if reason in {
        int(getattr(mt5_api, "DEAL_REASON_CLIENT", -999)),
        int(getattr(mt5_api, "DEAL_REASON_MOBILE", -999)),
        int(getattr(mt5_api, "DEAL_REASON_WEB", -999)),
    }:
        return "MANUAL"
    if reason == int(getattr(mt5_api, "DEAL_REASON_EXPERT", -999)):
        return "SYSTEM"
    if entry == int(getattr(mt5_api, "DEAL_ENTRY_OUT_BY", -999)):
        return "OUT_BY"
    if entry == int(getattr(mt5_api, "DEAL_ENTRY_OUT", -999)):
        return "UNKNOWN"
    return "UNKNOWN"


def kill_switch_pruefen(
    symbol: str,
    start_equity: float,
    aktuell_equity: float,
    max_dd_pct: float,
    paper_trading: bool,
) -> bool:
    """
    Prüft ob der Kill-Switch ausgelöst werden soll.

    Args:
        symbol:         Handelssymbol (für Logging)
        start_equity:   Startkapital der Session
        aktuell_equity: Aktueller Kontostand oder simuliertes Kapital
        max_dd_pct:     Maximaler Drawdown in Dezimal (z.B. 0.15 = 15%)
        paper_trading:  True = Paper-Modus

    Returns:
        True wenn Kill-Switch ausgelöst wird (Trader soll stoppen).
    """
    verlust = start_equity - aktuell_equity
    drawdown_pct = verlust / start_equity if start_equity > 0 else 0.0

    if drawdown_pct >= max_dd_pct:
        modus = "PAPER" if paper_trading else "LIVE"
        logger.critical("=" * 65)
        logger.critical(f"[{symbol}] KILL-SWITCH AUSGELÖST! [{modus}]")
        logger.critical(f"[{symbol}] Drawdown: {drawdown_pct:.1%} (Limit: {max_dd_pct:.1%})")
        logger.critical(
            f"[{symbol}] Startkapital: {start_equity:.2f} | "
            f"Aktuell: {aktuell_equity:.2f} | Verlust: {verlust:.2f}"
        )
        logger.critical(f"[{symbol}] Trader wird automatisch gestoppt!")
        logger.critical("=" * 65)
        return True

    if drawdown_pct >= max_dd_pct * 0.75:
        logger.warning(
            f"[{symbol}] Drawdown-WARNUNG: {drawdown_pct:.1%} (Kill-Switch-Limit: {max_dd_pct:.1%})"
        )
    elif drawdown_pct >= max_dd_pct * 0.50:
        logger.warning(f"[{symbol}] Drawdown {drawdown_pct:.1%} – Kill-Switch bei {max_dd_pct:.1%}")

    return False


def offenen_trade_pruefen(
    symbol: str,
    letzter_trade: Optional[dict],
    paper_trading: bool,
) -> Optional[dict]:
    """
    Prüft ob ein zuvor geöffneter Trade noch offen ist.

    Im Paper-Modus: Keine MT5-Abfrage, Prüfung erfolgt über OHLC-Kerzen.
    Im Live-Modus: MT5 wird nach Position-Status befragt.

    Args:
        symbol:         Handelssymbol
        letzter_trade:  Dict mit Trade-Info oder None
        paper_trading:  True = Paper-Modus

    Returns:
        letzter_trade wenn noch offen, None wenn geschlossen.
    """
    if letzter_trade is None:
        return None

    if paper_trading:
        return letzter_trade  # Paper: Prüfung via OHLC-Kerzen in trading_loop

    if not config.MT5_VERFUEGBAR:
        return letzter_trade

    mt5_api = config.mt5_api()
    positionen = mt5_api.positions_get(symbol=symbol)
    position_ticket = int(letzter_trade.get("position_ticket", 0) or 0)
    deal_ticket = int(letzter_trade.get("deal_ticket", 0) or 0)

    # Position noch offen?
    if positionen and position_ticket > 0:
        for pos in positionen:
            if int(pos.ticket) == position_ticket:
                return letzter_trade

    if positionen and position_ticket <= 0:
        if any(int(getattr(p, "magic", 0)) == config.MAGIC_NUMBER for p in positionen):
            return letzter_trade

    # Position ist geschlossen – History abfragen für PnL
    try:
        jetzt = datetime.now(timezone.utc)
        von = jetzt - timedelta(days=7)
        deals = mt5_api.history_deals_get(von, jetzt, group=symbol)

        if deals:
            deal_entry_out = getattr(mt5_api, "DEAL_ENTRY_OUT", 1)
            target_position_id = position_ticket

            if target_position_id <= 0 and deal_ticket > 0:
                open_deals = [d for d in deals if int(getattr(d, "ticket", 0)) == deal_ticket]
                if open_deals:
                    target_position_id = int(getattr(open_deals[-1], "position_id", 0) or 0)

            close_deals = []
            if target_position_id > 0:
                close_deals = [
                    d for d in deals
                    if int(getattr(d, "position_id", 0) or 0) == target_position_id
                    and int(getattr(d, "entry", -1)) == int(deal_entry_out)
                ]

            if not close_deals and target_position_id > 0:
                close_deals = [
                    d for d in deals
                    if int(getattr(d, "position_id", 0) or 0) == target_position_id
                    and int(getattr(d, "entry", -1)) in (1, 3)
                ]

            if close_deals:
                deal = close_deals[-1]
                exit_price = deal.price
                pnl_money = deal.profit + deal.commission + deal.swap
                entry_price = letzter_trade.get("entry_price", 0.0)
                richtung = letzter_trade.get("richtung", 0)

                if richtung == 2:
                    pnl_pips = (exit_price - entry_price) * _pip_faktor(symbol)
                else:
                    pnl_pips = (entry_price - exit_price) * _pip_faktor(symbol)

                open_zeit = letzter_trade.get("open_zeit", jetzt)
                dauer_min = int((jetzt - open_zeit).total_seconds() / 60)
                close_grund = _close_grund_aus_deal(deal)

                trade_close_loggen(
                    symbol=symbol, ticket=target_position_id, richtung=richtung,
                    entry_price=entry_price, exit_price=exit_price,
                    pnl_pips=pnl_pips, pnl_money=pnl_money, close_grund=close_grund,
                    dauer_minuten=dauer_min,
                    htf_bias=letzter_trade.get("htf_bias"),
                    ltf_signal=letzter_trade.get("ltf_signal"),
                )
                return None

        logger.warning(
            f"[{symbol}] Trade Position={position_ticket} (Deal={deal_ticket}) "
            "geschlossen, aber kein Deal in History gefunden"
        )
    except (AttributeError, TypeError, ValueError, OSError) as e:
        logger.error(f"[{symbol}] Fehler beim PnL-Abruf: {e}", exc_info=True)

    return None
