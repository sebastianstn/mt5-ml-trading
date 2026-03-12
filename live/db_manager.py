"""
db_manager.py – SQLite-Datenbankmanager für Trade-Logs

Ergänzt das CSV-basierte Logging um eine SQLite-Datenbank für robuste,
abfragebare Trade-Historie. CSV bleibt für MT5-Dashboard-Kompatibilität erhalten.

Läuft auf: Windows 11 Laptop
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Datenbankdatei im Log-Verzeichnis
DB_DATEINAME = "trades.db"


def db_pfad() -> Path:
    """Gibt den aktuellen Pfad zur SQLite-Datenbankdatei zurück."""
    return config.LOG_DIR / DB_DATEINAME


def db_verbinden() -> sqlite3.Connection:
    """
    Öffnet (oder erstellt) die SQLite-Datenbank und gibt eine Verbindung zurück.

    Aktiviert WAL-Modus für bessere Concurrent-Read-Performance (Dashboard-Lesezugriffe
    während der Trader schreibt).

    Returns:
        Offene sqlite3.Connection mit Row-Factory für dict-ähnlichen Zugriff.
    """
    pfad = db_pfad()
    pfad.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(pfad), timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL-Modus: Leser blockieren Schreiber nicht (wichtig für Dashboard)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def schema_erstellen() -> None:
    """
    Erstellt die Tabellen falls sie noch nicht existieren (idempotent).

    Tabellen:
        signals – Jede Kerzen-Auswertung (Heartbeat + Signale + Trades)
        trades  – Abgeschlossene Trades mit PnL
    """
    with db_verbinden() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,          -- UTC-Zeitstempel ISO-8601
                symbol          TEXT    NOT NULL,
                signal          INTEGER NOT NULL,          -- 2=Long, -1=Short, 0=Kein
                prob            REAL    NOT NULL,
                regime          INTEGER NOT NULL,
                paper_trading   INTEGER NOT NULL,          -- 1=Paper, 0=Live
                entry_price     REAL,
                sl_price        REAL,
                tp_price        REAL,
                htf_bias        INTEGER,                   -- Two-Stage HTF-Bias (optional)
                ltf_signal      INTEGER                    -- Two-Stage LTF-Signal (optional)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_close        TEXT    NOT NULL,          -- UTC-Zeitstempel ISO-8601
                symbol          TEXT    NOT NULL,
                ticket          INTEGER,
                richtung        INTEGER NOT NULL,          -- 2=Long, -1=Short
                entry_price     REAL    NOT NULL,
                exit_price      REAL    NOT NULL,
                pnl_pips        REAL    NOT NULL,
                pnl_money       REAL    NOT NULL,
                close_grund     TEXT    NOT NULL,          -- TP, SL, SO, MANUAL, SYSTEM, UNKNOWN
                dauer_minuten   INTEGER,
                htf_bias        INTEGER,
                ltf_signal      INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
                ON signals (symbol, ts);

            CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts_close
                ON trades (symbol, ts_close);
        """)
    logger.debug("SQLite-Schema sichergestellt: %s", db_pfad())


def signal_einfuegen(
    symbol: str,
    signal: int,
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
    Schreibt einen Signal-Eintrag in die signals-Tabelle.

    Args:
        symbol:       Handelssymbol
        signal:       2=Long, -1=Short, 0=Kein Signal
        prob:         Modell-Wahrscheinlichkeit
        regime:       Markt-Regime (0–3)
        paper_trading: True = Paper-Modus
        entry_price:  Aktueller Close-Preis
        sl_price:     Berechnetes Stop-Loss-Niveau
        tp_price:     Berechnetes Take-Profit-Niveau
        htf_bias:     Two-Stage HTF-Bias (optional)
        ltf_signal:   Two-Stage LTF-Signal (optional)
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with db_verbinden() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (ts, symbol, signal, prob, regime, paper_trading,
                     entry_price, sl_price, tp_price, htf_bias, ltf_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, symbol, signal, round(prob, 6), regime,
                    int(paper_trading), entry_price, sl_price, tp_price,
                    htf_bias, ltf_signal,
                ),
            )
    except sqlite3.Error as e:
        logger.error("SQLite signal_einfuegen Fehler: %s", e)


def trade_close_einfuegen(
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
    Schreibt einen abgeschlossenen Trade in die trades-Tabelle.

    Args:
        symbol:        Handelssymbol
        ticket:        MT5-Position-Ticket (0 im Paper-Modus)
        richtung:      2=Long, -1=Short
        entry_price:   Einstiegskurs
        exit_price:    Ausstiegskurs
        pnl_pips:      PnL in Pips
        pnl_money:     PnL in Kontowährung
        close_grund:   Schließungsgrund (TP, SL, MANUAL, etc.)
        dauer_minuten: Trade-Dauer in Minuten
        htf_bias:      Two-Stage HTF-Bias (optional)
        ltf_signal:    Two-Stage LTF-Signal (optional)
    """
    ts_close = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with db_verbinden() as conn:
            conn.execute(
                """
                INSERT INTO trades
                    (ts_close, symbol, ticket, richtung, entry_price, exit_price,
                     pnl_pips, pnl_money, close_grund, dauer_minuten,
                     htf_bias, ltf_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_close, symbol, ticket, richtung,
                    round(entry_price, 6), round(exit_price, 6),
                    round(pnl_pips, 2), round(pnl_money, 4),
                    close_grund, dauer_minuten,
                    htf_bias, ltf_signal,
                ),
            )
    except sqlite3.Error as e:
        logger.error("SQLite trade_close_einfuegen Fehler: %s", e)


def trades_lesen(
    symbol: Optional[str] = None,
    limit: int = 1000,
) -> list[dict]:
    """
    Liest abgeschlossene Trades aus der Datenbank.

    Args:
        symbol: Optional filtern nach Symbol (None = alle)
        limit:  Maximale Anzahl zurückgegebener Zeilen (neueste zuerst)

    Returns:
        Liste von Dicts mit allen Spalten der trades-Tabelle.
    """
    try:
        with db_verbinden() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY ts_close DESC LIMIT ?",
                    (symbol.upper(), limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM trades ORDER BY ts_close DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("SQLite trades_lesen Fehler: %s", e)
        return []


def kpi_aus_db(symbol: str) -> dict:
    """
    Berechnet KPIs direkt aus der SQLite-Datenbank (für weekly_kpi_report.py).

    Args:
        symbol: Handelssymbol (z.B. "USDCAD")

    Returns:
        Dict mit: n_trades, n_wins, win_rate, avg_pnl_pips, avg_pnl_money,
                  total_pnl_money, n_tp, n_sl, n_manual
    """
    try:
        with db_verbinden() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                      AS n_trades,
                    SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) AS n_wins,
                    AVG(pnl_pips)                                 AS avg_pnl_pips,
                    AVG(pnl_money)                                AS avg_pnl_money,
                    SUM(pnl_money)                                AS total_pnl_money,
                    SUM(CASE WHEN close_grund = 'TP' THEN 1 ELSE 0 END)     AS n_tp,
                    SUM(CASE WHEN close_grund = 'SL' THEN 1 ELSE 0 END)     AS n_sl,
                    SUM(CASE WHEN close_grund = 'MANUAL' THEN 1 ELSE 0 END) AS n_manual
                FROM trades
                WHERE symbol = ?
                """,
                (symbol.upper(),),
            ).fetchone()

            if row is None or row["n_trades"] == 0:
                return {"n_trades": 0}

            n_trades = int(row["n_trades"])
            n_wins = int(row["n_wins"] or 0)
            return {
                "n_trades": n_trades,
                "n_wins": n_wins,
                "win_rate": n_wins / n_trades if n_trades > 0 else 0.0,
                "avg_pnl_pips": float(row["avg_pnl_pips"] or 0.0),
                "avg_pnl_money": float(row["avg_pnl_money"] or 0.0),
                "total_pnl_money": float(row["total_pnl_money"] or 0.0),
                "n_tp": int(row["n_tp"] or 0),
                "n_sl": int(row["n_sl"] or 0),
                "n_manual": int(row["n_manual"] or 0),
            }
    except sqlite3.Error as e:
        logger.error("SQLite kpi_aus_db Fehler: %s", e)
        return {"n_trades": 0}
