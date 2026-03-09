"""
evaluate_mt5_testphase.py – Bewertung der MT5-Paper-Testphase (USDCAD/USDJPY).

Ziel:
    Liest die aktuellen Live-Logs aus `logs/*_signals.csv` und optional
    `logs/*_closes.csv`, berechnet Kern-KPIs und schreibt einen
    reproduzierbaren Testphasen-Report als CSV.

Laufort:
    Linux-Server
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EvalConfig:
    """Konfiguration für die Testphasen-Auswertung."""

    log_dir: Path
    output_dir: Path
    symbols: tuple[str, ...]
    hours: int
    timeframe: str
    stale_factor: float
    start_equity: float
    min_signals: int
    min_closes: int
    min_pf: float
    min_win_rate_pct: float
    max_dd_pct_limit: float


def parse_args() -> argparse.Namespace:
    """Liest CLI-Parameter ein."""
    parser = argparse.ArgumentParser(
        description=(
            "Bewertet MT5-Testphase aus *_signals.csv und *_closes.csv "
            "für USDCAD/USDJPY."
        )
    )
    parser.add_argument("--log_dir", default="logs", help="Pfad zu den Live-Logs")
    parser.add_argument(
        "--output_dir",
        default="reports/testphase",
        help="Zielordner für Report-CSV",
    )
    parser.add_argument(
        "--symbols",
        default="USDCAD,USDJPY",
        help="Komma-getrennte Symbol-Liste (Standard: USDCAD,USDJPY)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Rückblickfenster in Stunden (Standard: 48)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["M5", "M15", "M30", "H1"],
        default="M5",
        help="Timeframe zur Freshness-Bewertung (Standard: M5)",
    )
    parser.add_argument(
        "--stale_factor",
        type=float,
        default=2.0,
        help="Freshness-Grenze = timeframe_minuten * stale_factor (Standard: 2.0)",
    )
    parser.add_argument(
        "--start_equity",
        type=float,
        default=10000.0,
        help="Referenzkapital für Drawdown-Prozent (Standard: 10000)",
    )
    parser.add_argument(
        "--min_signals",
        type=int,
        default=20,
        help="Mindestanzahl aktiver Signale im Fenster (Standard: 20)",
    )
    parser.add_argument(
        "--min_closes",
        type=int,
        default=5,
        help="Mindestanzahl Closings für belastbare PnL-Bewertung (Standard: 5)",
    )
    parser.add_argument(
        "--min_pf",
        type=float,
        default=1.3,
        help="Mindest-Gewinnfaktor (Standard: 1.3)",
    )
    parser.add_argument(
        "--min_win_rate_pct",
        type=float,
        default=45.0,
        help="Mindest-Win-Rate in Prozent (Standard: 45)",
    )
    parser.add_argument(
        "--max_dd_pct_limit",
        type=float,
        default=-10.0,
        help="Maximal erlaubter Drawdown in Prozent, negativ (Standard: -10)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Baut typsichere Konfiguration aus CLI-Parametern."""
    symbols = tuple(
        sorted(
            {
                token.strip().upper()
                for token in str(args.symbols).split(",")
                if token.strip()
            }
        )
    )
    if not symbols:
        raise ValueError("Keine gültigen Symbole angegeben.")
    return EvalConfig(
        log_dir=Path(args.log_dir),
        output_dir=Path(args.output_dir),
        symbols=symbols,
        hours=int(args.hours),
        timeframe=str(args.timeframe),
        stale_factor=float(args.stale_factor),
        start_equity=float(args.start_equity),
        min_signals=int(args.min_signals),
        min_closes=int(args.min_closes),
        min_pf=float(args.min_pf),
        min_win_rate_pct=float(args.min_win_rate_pct),
        max_dd_pct_limit=float(args.max_dd_pct_limit),
    )


def timeframe_minutes(timeframe: str) -> int:
    """Mappt Timeframe auf Minuten pro Kerze."""
    return {"M5": 5, "M15": 15, "M30": 30, "H1": 60}[timeframe]


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Lädt CSV robust oder gibt None zurück, wenn Datei fehlt/leer ist."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def prepare_time_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Filtert ein DataFrame auf das gewünschte UTC-Zeitfenster."""
    result = df.copy()
    if "time" not in result.columns:
        return result.iloc[0:0].copy()
    result["time"] = pd.to_datetime(result["time"], errors="coerce", utc=True)
    result = result.dropna(subset=["time"]).copy()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return result[result["time"] >= cutoff].copy()


def profit_factor_from_series(pnl_series: pd.Series) -> Optional[float]:
    """Berechnet Gewinnfaktor aus PnL-Serie."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty:
        return None
    wins = float(clean[clean > 0].sum())
    losses_abs = abs(float(clean[clean < 0].sum()))
    if losses_abs <= 0.0:
        return float("inf") if wins > 0 else None
    return wins / losses_abs


def max_drawdown_pct_from_money(
    pnl_series: pd.Series, start_equity: float
) -> Optional[float]:
    """Berechnet max. Drawdown in Prozent aus kumulativer PnL-Kurve."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty or start_equity <= 0.0:
        return None
    equity = start_equity + clean.cumsum()
    drawdown_money = equity - equity.cummax()
    dd_pct = (drawdown_money / start_equity) * 100.0
    return float(dd_pct.min())


def evaluate_symbol(symbol: str, config: EvalConfig) -> dict[str, object]:
    """Bewertet ein einzelnes Symbol auf Basis Signal- und Close-Logs."""
    signal_path = config.log_dir / f"{symbol}_signals.csv"
    close_path = config.log_dir / f"{symbol}_closes.csv"

    signals_raw = load_csv_if_exists(signal_path)
    closes_raw = load_csv_if_exists(close_path)

    signals = (
        prepare_time_window(signals_raw, config.hours)
        if signals_raw is not None
        else pd.DataFrame()
    )
    closes = (
        prepare_time_window(closes_raw, config.hours)
        if closes_raw is not None
        else pd.DataFrame()
    )

    # Aktive Signale: nur echte Trade-Signale (Long/Short), Heartbeats ausgeschlossen.
    active_signals = pd.DataFrame()
    if not signals.empty and "signal" in signals.columns:
        signals["signal"] = pd.to_numeric(signals["signal"], errors="coerce")
        active_signals = signals[signals["signal"].isin([-1, 2])].copy()

    now_utc = datetime.now(timezone.utc)
    last_signal_utc = None
    minutes_since_last_signal: Optional[float] = None
    if not signals.empty:
        last_ts = pd.to_datetime(signals["time"], errors="coerce", utc=True).max()
        if pd.notna(last_ts):
            last_signal_utc = last_ts.strftime("%Y-%m-%d %H:%M:%S")
            minutes_since_last_signal = (
                now_utc - last_ts.to_pydatetime()
            ).total_seconds() / 60.0

    stale_limit_minutes = timeframe_minutes(config.timeframe) * config.stale_factor
    fresh = (
        minutes_since_last_signal is not None
        and minutes_since_last_signal <= stale_limit_minutes
    )

    closes_count = int(len(closes))
    pnl_money = pd.Series(dtype=float)
    if not closes.empty and "pnl_money" in closes.columns:
        pnl_money = pd.to_numeric(closes["pnl_money"], errors="coerce").dropna()

    pf = profit_factor_from_series(pnl_money)
    win_rate_pct: Optional[float] = None
    net_pnl_money: Optional[float] = None
    avg_pnl_money: Optional[float] = None
    max_dd_pct: Optional[float] = None

    if not pnl_money.empty:
        win_rate_pct = float((pnl_money > 0).mean() * 100.0)
        net_pnl_money = float(pnl_money.sum())
        avg_pnl_money = float(pnl_money.mean())
        max_dd_pct = max_drawdown_pct_from_money(pnl_money, config.start_equity)

    # Gates / Bewertung
    gate_fresh = bool(fresh)
    gate_signal_activity = int(len(active_signals)) >= config.min_signals
    gate_has_enough_closes = closes_count >= config.min_closes
    gate_pf = (pf is not None) and (pf >= config.min_pf)
    gate_win = (win_rate_pct is not None) and (win_rate_pct >= config.min_win_rate_pct)
    gate_dd = (max_dd_pct is not None) and (max_dd_pct >= config.max_dd_pct_limit)

    if not gate_fresh:
        status = "NO_GO"
        reason = "Stale oder fehlende Live-Signale"
    elif not gate_signal_activity:
        status = "NO_GO"
        reason = "Zu wenig aktive Signale"
    elif not gate_has_enough_closes:
        status = "WATCH"
        reason = "Noch zu wenige Close-Events für belastbare PnL-Bewertung"
    elif gate_pf and gate_win and gate_dd:
        status = "GO"
        reason = "Alle KPI-Gates bestanden"
    else:
        status = "NO_GO"
        reason = "Mindestens ein KPI-Gate verfehlt"

    return {
        "symbol": symbol,
        "window_hours": config.hours,
        "signals_rows": int(len(signals)),
        "active_signals": int(len(active_signals)),
        "closes_rows": closes_count,
        "last_signal_utc": last_signal_utc,
        "minutes_since_last_signal": (
            None
            if minutes_since_last_signal is None
            else round(float(minutes_since_last_signal), 2)
        ),
        "stale_limit_minutes": float(stale_limit_minutes),
        "fresh": gate_fresh,
        "profit_factor": None if pf is None else round(float(pf), 4),
        "win_rate_pct": None if win_rate_pct is None else round(float(win_rate_pct), 2),
        "net_pnl_money": (
            None if net_pnl_money is None else round(float(net_pnl_money), 2)
        ),
        "avg_pnl_money": (
            None if avg_pnl_money is None else round(float(avg_pnl_money), 2)
        ),
        "max_dd_pct": None if max_dd_pct is None else round(float(max_dd_pct), 2),
        "gate_fresh": gate_fresh,
        "gate_signal_activity": gate_signal_activity,
        "gate_has_enough_closes": gate_has_enough_closes,
        "gate_pf": gate_pf,
        "gate_win_rate": gate_win,
        "gate_dd": gate_dd,
        "status": status,
        "reason": reason,
    }


def main() -> None:
    """CLI-Einstiegspunkt für Testphasen-Bewertung."""
    args = parse_args()
    config = build_config(args)

    rows: list[dict[str, object]] = []
    for symbol in config.symbols:
        rows.append(evaluate_symbol(symbol, config))

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["status", "symbol"], ascending=[True, True]
    ).reset_index(drop=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = config.output_dir / f"mt5_testphase_eval_{stamp}.csv"
    latest_path = config.output_dir / "mt5_testphase_eval_latest.csv"

    result.to_csv(out_path, index=False)
    result.to_csv(latest_path, index=False)

    print("=== MT5 TESTPHASE EVAL ===")
    print(f"Output: {out_path}")
    print(f"Latest: {latest_path}")
    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
