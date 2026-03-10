"""daily_phase7_dashboard.py – kompaktes Tagesdashboard für Phase 7.

Zweck:
    Fasst die operative Gesundheit des laufenden Paper-/Live-Betriebs für
    `USDCAD` und `USDJPY` in einer täglichen Ampel zusammen.

Ampellogik:
    - OK:       frische Daten, operative Aktivität sichtbar, keine kritische DD
    - WATCH:    frische Daten, aber zu wenig Aktivität oder schwache Close-KPIs
    - INCIDENT: stale Logs oder kritischer Drawdown / operative Störung

Läuft auf:
    Linux-Server
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
STANDARD_SYMBOLE = ("USDCAD", "USDJPY")
SIGNAL_SUFFIX = "_signals.csv"
CLOSE_SUFFIX = "_closes.csv"
STALE_FACTOR = 1.5
START_EQUITY = 10000.0
CRITICAL_DD_PCT = -10.0
WATCH_SIGNAL_MIN = 2
WATCH_CLOSE_MIN = 2
RUNTIME_FUTURE_GRACE_MINUTES = 2
MAX_AUTO_UTC_SHIFT_HOURS = 14
RUNTIME_HEARTBEAT_RE = re.compile(
    r"\[(?P<symbol>[A-Z]{6})\]\s+Neue\s+[A-Z0-9_]+-Kerze\s+\|\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+UTC"
)
WATCHDOG_JSON_NAME = "live_log_watchdog_latest.json"


@dataclass(frozen=True)
class WatchdogSnapshot:
    """Synchronisierte Watchdog-Daten vom Windows-Host."""

    overall_status: str
    generated_at_utc: str
    stale_limit_minutes: Optional[float]
    lag_limit_minutes: Optional[float]
    symbols: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class DailySymbolStatus:
    """Operativer Tagesstatus eines Symbols."""

    symbol: str
    ampel: str
    begruendung: str
    live_fresh: bool
    last_event_utc: str
    minutes_since_last: Optional[float]
    runtime_fresh: bool
    last_runtime_utc: str
    minutes_since_runtime: Optional[float]
    csv_runtime_lag_min: Optional[float]
    watchdog_status: str
    watchdog_reason: str
    events: int
    signale: int
    closes: int
    long_signale: int
    short_signale: int
    tp_closes: int
    sl_closes: int
    net_pnl: Optional[float]
    profit_factor: Optional[float]
    win_rate_pct: Optional[float]
    max_drawdown_pct: Optional[float]
    avg_prob_pct: float
    avg_dauer_min: Optional[float]


def parse_args() -> argparse.Namespace:
    """Parst CLI-Argumente für das Tagesdashboard."""
    parser = argparse.ArgumentParser(
        description="Erstellt ein kompaktes Tagesdashboard für Phase 7 (OK/WATCH/INCIDENT)."
    )
    parser.add_argument(
        "--log_dir",
        default="auto",
        help=(
            "Ordner mit *_signals.csv und *_closes.csv "
            "(Standard: auto = frischester aktiver Ordner unter logs/)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="reports",
        help="Ausgabeordner für CSV/Markdown (Standard: reports)",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(STANDARD_SYMBOLE),
        help="Komma-getrennte Symbol-Liste (Standard: USDCAD,USDJPY)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Rückblickfenster in Stunden (Standard: 24)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["H1", "M30", "M15", "M5_TWO_STAGE"],
        default="M5_TWO_STAGE",
        help="Timeframe für Freshness-Grenze (Standard: M5_TWO_STAGE)",
    )
    return parser.parse_args()


def timeframe_minutes(timeframe: str) -> int:
    """Mappt Timeframe-String auf Minuten pro Kerze."""
    mapping = {"H1": 60, "M30": 30, "M15": 15, "M5_TWO_STAGE": 5}
    return mapping.get(timeframe, 60)


def _candidate_log_dirs(base_log_dir: Path) -> list[Path]:
    """Liefert den Basis-Logordner plus direkte Unterordner als Kandidaten."""
    if not base_log_dir.exists() or not base_log_dir.is_dir():
        return [base_log_dir]

    candidates = [base_log_dir]
    for child in sorted(base_log_dir.iterdir()):
        if child.is_dir():
            candidates.append(child)
    return candidates


def _latest_signal_mtime(log_dir: Path, symbols: tuple[str, ...]) -> Optional[float]:
    """Gibt die jüngste mtime passender Signaldateien eines Ordners zurück."""
    mtimes: list[float] = []
    for symbol in symbols:
        signal_path = log_dir / f"{symbol}{SIGNAL_SUFFIX}"
        if (
            signal_path.exists()
            and signal_path.is_file()
            and signal_path.stat().st_size > 0
        ):
            mtimes.append(signal_path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def resolve_log_dir(log_dir_arg: str, symbols: tuple[str, ...]) -> Path:
    """Löst den Logordner auf oder erkennt automatisch den frischesten aktiven Ordner."""
    requested = Path(log_dir_arg)
    if log_dir_arg.lower() != "auto":
        return requested if requested.is_absolute() else BASE_DIR / requested

    base_log_dir = LOG_DIR
    best_dir = base_log_dir
    best_mtime: Optional[float] = None

    for candidate in _candidate_log_dirs(base_log_dir):
        candidate_mtime = _latest_signal_mtime(candidate, symbols)
        if candidate_mtime is None:
            continue
        if best_mtime is None or candidate_mtime > best_mtime:
            best_dir = candidate
            best_mtime = candidate_mtime

    return best_dir


def _load_window_csv(path: Path, hours: int) -> pd.DataFrame:
    """Lädt eine CSV und filtert sie auf das gewünschte Zeitfenster."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    return df[df["time"] >= cutoff].copy()


def _normalize_runtime_timestamp_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Korrigiert Heartbeats, die durch Broker-Serverzeit fälschlich in der Zukunft liegen."""
    normalized = timestamp
    future_limit = pd.Timestamp.now(tz="UTC") + pd.Timedelta(
        minutes=RUNTIME_FUTURE_GRACE_MINUTES
    )
    shift_hours = 0

    while normalized > future_limit and shift_hours < MAX_AUTO_UTC_SHIFT_HOURS:
        normalized -= pd.Timedelta(hours=1)
        shift_hours += 1

    return normalized


def _letzter_runtime_heartbeat(log_dir: Path, symbol: str) -> Optional[pd.Timestamp]:
    """Liest den letzten Kerzen-Heartbeat eines Symbols aus live_trader.log."""
    log_path = log_dir / "live_trader.log"
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None

    last_ts: Optional[pd.Timestamp] = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = RUNTIME_HEARTBEAT_RE.search(line)
        if match is None or match.group("symbol") != symbol:
            continue
        parsed = pd.to_datetime(match.group("ts"), errors="coerce", utc=True)
        if pd.isna(parsed):
            continue
        parsed = _normalize_runtime_timestamp_utc(parsed)
        if last_ts is None or parsed > last_ts:
            last_ts = parsed
    return last_ts


def load_watchdog_snapshot(log_dir: Path) -> WatchdogSnapshot:
    """Lädt optional die synchronisierte Windows-Watchdog-Sicht pro Symbol."""
    path = log_dir / WATCHDOG_JSON_NAME
    if not path.exists() or path.stat().st_size == 0:
        return WatchdogSnapshot(
            overall_status="",
            generated_at_utc="",
            stale_limit_minutes=None,
            lag_limit_minutes=None,
            symbols={},
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Watchdog-JSON konnte nicht gelesen werden: %s", path)
        return WatchdogSnapshot(
            overall_status="",
            generated_at_utc="",
            stale_limit_minutes=None,
            lag_limit_minutes=None,
            symbols={},
        )

    symbol_entries = payload.get("symbols", [])
    if not isinstance(symbol_entries, list):
        symbol_entries = []

    result: dict[str, dict[str, Any]] = {}
    for entry in symbol_entries:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        result[symbol] = entry
    return WatchdogSnapshot(
        overall_status=str(payload.get("overall_status", "")).upper().strip(),
        generated_at_utc=str(payload.get("generated_at_utc", "")).strip(),
        stale_limit_minutes=_as_float(payload.get("stale_limit_minutes")),
        lag_limit_minutes=_as_float(payload.get("lag_limit_minutes")),
        symbols=result,
    )


def _profit_factor_from_money(pnl_series: pd.Series) -> Optional[float]:
    """Berechnet Gewinnfaktor aus Geld-PnL."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty:
        return None
    wins = float(clean[clean > 0].sum())
    losses_abs = abs(float(clean[clean < 0].sum()))
    if losses_abs <= 0.0:
        return float("inf") if wins > 0 else None
    return wins / losses_abs


def _max_drawdown_pct_from_money(pnl_series: pd.Series) -> Optional[float]:
    """Berechnet maximalen Drawdown in Prozent aus kumulierter Geld-PnL."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty:
        return None
    equity = START_EQUITY + clean.cumsum()
    drawdown_money = equity - equity.cummax()
    dd_pct = (drawdown_money / START_EQUITY) * 100.0
    return float(dd_pct.min())


def _tp_sl_counts(closes: pd.DataFrame) -> tuple[int, int]:
    """Zählt TP- und SL-Close-Gründe robust."""
    if closes.empty or "close_grund" not in closes.columns:
        return 0, 0
    close_grund = closes["close_grund"].fillna("").astype(str)
    return int((close_grund == "TP").sum()), int((close_grund == "SL").sum())


def _as_float(value: Any) -> Optional[float]:
    """Konvertiert Einzelwerte robust nach float oder gibt None zurück."""
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int:
    """Konvertiert Einzelwerte robust nach int oder gibt 0 zurück."""
    if value is None:
        return 0
    return int(value)


def status_bewerten(metrics: dict[str, object]) -> tuple[str, str]:
    """Leitet aus den Tagesmetriken OK/WATCH/INCIDENT ab."""
    watchdog_status = str(metrics.get("watchdog_status", "")).upper().strip()
    watchdog_reason = str(metrics.get("watchdog_reason", "")).strip()

    if watchdog_status == "INCIDENT":
        return "INCIDENT", watchdog_reason or "Windows-Watchdog meldet INCIDENT"
    if watchdog_status == "WATCH":
        return "WATCH", watchdog_reason or "Windows-Watchdog meldet WATCH"

    stale_limit_minutes = _as_float(metrics.get("stale_limit_minutes")) or 0.0
    runtime_fresh = bool(metrics.get("runtime_fresh", False))
    signal_fresh = bool(metrics.get("live_fresh", False))
    csv_runtime_lag_min = _as_float(metrics.get("csv_runtime_lag_min"))

    if runtime_fresh and not signal_fresh:
        if csv_runtime_lag_min is not None:
            return (
                "INCIDENT",
                f"Trader-Heartbeat frisch, aber Signal-CSV hinkt {csv_runtime_lag_min:.1f} Min hinterher",
            )
        return "INCIDENT", "Trader-Heartbeat frisch, aber Signal-CSV stale"

    if (
        csv_runtime_lag_min is not None
        and stale_limit_minutes > 0.0
        and csv_runtime_lag_min > stale_limit_minutes
    ):
        return (
            "INCIDENT",
            f"Signal-CSV hinkt {csv_runtime_lag_min:.1f} Min hinter Runtime-Heartbeat",
        )

    if not bool(metrics.get("live_fresh", False)):
        return "INCIDENT", "Keine frischen Signal-Updates"

    max_dd = _as_float(metrics.get("max_drawdown_pct"))
    if max_dd is not None and float(max_dd) <= CRITICAL_DD_PCT:
        return "INCIDENT", "Kritischer Drawdown in Close-Daten"

    signale = _as_int(metrics.get("signale"))
    closes = _as_int(metrics.get("closes"))
    net_pnl = _as_float(metrics.get("net_pnl"))
    profit_factor = _as_float(metrics.get("profit_factor"))

    if signale < WATCH_SIGNAL_MIN:
        return "WATCH", "Frische Daten, aber sehr geringe Signalaktivität"

    if closes == 0:
        return "OK", "Frische Daten und Signale vorhanden, aber noch keine Closes"

    if closes < WATCH_CLOSE_MIN:
        return "WATCH", "Erste Closes vorhanden, aber noch geringe Stichprobe"

    if net_pnl is not None and float(net_pnl) < 0:
        return "WATCH", "Negative Tages-PnL in Close-Daten"

    if profit_factor is not None and float(profit_factor) < 1.0:
        return "WATCH", "Profit Factor unter 1.0"

    return "OK", "Frische Daten und unkritische Tages-KPIs"


def symbol_status_berechnen(
    symbol: str,
    log_dir: Path,
    hours: int,
    timeframe: str,
    watchdog_snapshot: Optional[WatchdogSnapshot] = None,
) -> DailySymbolStatus:
    """Berechnet den Tagesstatus eines Symbols."""
    signals = _load_window_csv(log_dir / f"{symbol}{SIGNAL_SUFFIX}", hours)
    closes = _load_window_csv(log_dir / f"{symbol}{CLOSE_SUFFIX}", hours)

    events = int(len(signals))
    signale = 0
    long_signale = 0
    short_signale = 0
    avg_prob_pct = 0.0
    last_event_utc = ""
    minutes_since_last: Optional[float] = None
    live_fresh = False
    last_signal_ts: Optional[pd.Timestamp] = None

    if not signals.empty:
        signals["signal"] = pd.to_numeric(signals["signal"], errors="coerce")
        if "prob" in signals.columns:
            signals["prob"] = pd.to_numeric(signals["prob"], errors="coerce")
        else:
            signals["prob"] = pd.Series(dtype=float)
        signale = int(signals["signal"].isin([-1, 2]).sum())
        long_signale = int((signals["signal"] == 2).sum())
        short_signale = int((signals["signal"] == -1).sum())
        avg_prob_pct = (
            float(signals["prob"].dropna().mean() * 100.0)
            if not signals["prob"].dropna().empty
            else 0.0
        )

        last_ts = pd.to_datetime(signals["time"], errors="coerce", utc=True).max()
        if pd.notna(last_ts):
            last_signal_ts = last_ts
            now_ts = (
                pd.Timestamp.now(tz=last_ts.tz)
                if getattr(last_ts, "tz", None)
                else pd.Timestamp.now()
            )
            minutes_since_last = float((now_ts - last_ts).total_seconds() / 60.0)
            live_fresh = minutes_since_last <= (
                timeframe_minutes(timeframe) * STALE_FACTOR
            )
            last_event_utc = str(last_ts)

    runtime_ts = _letzter_runtime_heartbeat(log_dir, symbol)
    last_runtime_utc = ""
    minutes_since_runtime: Optional[float] = None
    runtime_fresh = False
    stale_limit_minutes = float(timeframe_minutes(timeframe) * STALE_FACTOR)
    if runtime_ts is not None:
        now_runtime = (
            pd.Timestamp.now(tz=runtime_ts.tz)
            if getattr(runtime_ts, "tz", None)
            else pd.Timestamp.now()
        )
        minutes_since_runtime = float((now_runtime - runtime_ts).total_seconds() / 60.0)
        runtime_fresh = minutes_since_runtime <= stale_limit_minutes
        last_runtime_utc = str(runtime_ts)

    csv_runtime_lag_min: Optional[float] = None
    if (
        last_signal_ts is not None
        and runtime_ts is not None
        and runtime_ts > last_signal_ts
    ):
        csv_runtime_lag_min = float(
            (runtime_ts - last_signal_ts).total_seconds() / 60.0
        )

    closes_count = int(len(closes))
    tp_closes, sl_closes = _tp_sl_counts(closes)
    pnl_money = pd.to_numeric(
        closes.get("pnl_money", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    dauer = pd.to_numeric(
        closes.get("dauer_min", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    net_pnl = float(pnl_money.sum()) if not pnl_money.empty else None
    profit_factor = _profit_factor_from_money(pnl_money)
    win_rate_pct = (
        float((pnl_money > 0).mean() * 100.0) if not pnl_money.empty else None
    )
    max_dd_pct = (
        _max_drawdown_pct_from_money(pnl_money) if not pnl_money.empty else None
    )
    avg_dauer_min = float(dauer.mean()) if not dauer.empty else None
    watchdog_entry = (watchdog_snapshot.symbols if watchdog_snapshot else {}).get(
        symbol.upper(), {}
    )
    watchdog_status = str(watchdog_entry.get("status", "")).upper().strip()
    watchdog_reason = str(watchdog_entry.get("reason", "")).strip()

    ampel, begruendung = status_bewerten(
        {
            "live_fresh": live_fresh,
            "signale": signale,
            "closes": closes_count,
            "net_pnl": net_pnl,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd_pct,
            "runtime_fresh": runtime_fresh,
            "csv_runtime_lag_min": csv_runtime_lag_min,
            "stale_limit_minutes": stale_limit_minutes,
            "watchdog_status": watchdog_status,
            "watchdog_reason": watchdog_reason,
        }
    )

    return DailySymbolStatus(
        symbol=symbol,
        ampel=ampel,
        begruendung=begruendung,
        live_fresh=live_fresh,
        last_event_utc=last_event_utc,
        minutes_since_last=minutes_since_last,
        runtime_fresh=runtime_fresh,
        last_runtime_utc=last_runtime_utc,
        minutes_since_runtime=minutes_since_runtime,
        csv_runtime_lag_min=csv_runtime_lag_min,
        watchdog_status=watchdog_status,
        watchdog_reason=watchdog_reason,
        events=events,
        signale=signale,
        closes=closes_count,
        long_signale=long_signale,
        short_signale=short_signale,
        tp_closes=tp_closes,
        sl_closes=sl_closes,
        net_pnl=net_pnl,
        profit_factor=profit_factor,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_dd_pct,
        avg_prob_pct=avg_prob_pct,
        avg_dauer_min=avg_dauer_min,
    )


def statuses_to_dataframe(statuses: list[DailySymbolStatus]) -> pd.DataFrame:
    """Wandelt Statusobjekte in ein exportierbares DataFrame um."""
    rows: list[dict[str, object]] = []
    for status in statuses:
        rows.append(
            {
                "symbol": status.symbol,
                "ampel": status.ampel,
                "begruendung": status.begruendung,
                "live_fresh": status.live_fresh,
                "last_event_utc": status.last_event_utc,
                "minutes_since_last": (
                    round(status.minutes_since_last, 1)
                    if status.minutes_since_last is not None
                    else None
                ),
                "runtime_fresh": status.runtime_fresh,
                "last_runtime_utc": status.last_runtime_utc,
                "minutes_since_runtime": (
                    round(status.minutes_since_runtime, 1)
                    if status.minutes_since_runtime is not None
                    else None
                ),
                "csv_runtime_lag_min": (
                    round(status.csv_runtime_lag_min, 1)
                    if status.csv_runtime_lag_min is not None
                    else None
                ),
                "watchdog_status": status.watchdog_status,
                "watchdog_reason": status.watchdog_reason,
                "events": status.events,
                "signale": status.signale,
                "long_signale": status.long_signale,
                "short_signale": status.short_signale,
                "closes": status.closes,
                "tp_closes": status.tp_closes,
                "sl_closes": status.sl_closes,
                "avg_prob_pct": round(status.avg_prob_pct, 2),
                "net_pnl": None if status.net_pnl is None else round(status.net_pnl, 2),
                "profit_factor": (
                    None
                    if status.profit_factor is None
                    else round(status.profit_factor, 3)
                ),
                "win_rate_pct": (
                    None
                    if status.win_rate_pct is None
                    else round(status.win_rate_pct, 2)
                ),
                "max_drawdown_pct": (
                    None
                    if status.max_drawdown_pct is None
                    else round(status.max_drawdown_pct, 2)
                ),
                "avg_dauer_min": (
                    None
                    if status.avg_dauer_min is None
                    else round(status.avg_dauer_min, 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_dashboard_schreiben(
    statuses: list[DailySymbolStatus],
    output_dir: Path,
    hours: int,
    timeframe: str,
    log_dir: Path,
    watchdog_snapshot: WatchdogSnapshot,
) -> Path:
    """Schreibt das Tagesdashboard als Markdown-Datei."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if all(status.ampel == "OK" for status in statuses):
        gesamt = "OK"
    elif any(status.ampel == "INCIDENT" for status in statuses):
        gesamt = "INCIDENT"
    else:
        gesamt = "WATCH"

    erstellt = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = output_dir / "phase7_daily_dashboard_latest.md"
    watchdog_datei = log_dir / WATCHDOG_JSON_NAME
    watchdog_verfuegbar = bool(watchdog_snapshot.symbols)
    stale_limit_text = (
        str(watchdog_snapshot.stale_limit_minutes)
        if watchdog_snapshot.stale_limit_minutes is not None
        else "-"
    )
    lag_limit_text = (
        str(watchdog_snapshot.lag_limit_minutes)
        if watchdog_snapshot.lag_limit_minutes is not None
        else "-"
    )
    lines = [
        "# Phase 7 – Daily Dashboard",
        "",
        f"**Erstellt:** {erstellt}",
        f"**Fenster:** letzte {hours}h",
        f"**Timeframe:** `{timeframe}`",
        f"**Log-Quelle:** `{log_dir}`",
        f"**Watchdog-Datei:** `{watchdog_datei}`",
        f"**Gesamtampel:** **{gesamt}**",
        "",
        "## Watchdog-Überblick",
        "",
        f"- **Datei vorhanden:** {'Ja' if watchdog_datei.exists() else 'Nein'}",
        f"- **Watchdog-Daten geladen:** {'Ja' if watchdog_verfuegbar else 'Nein'}",
        f"- **Watchdog Gesamtstatus:** {watchdog_snapshot.overall_status or '-'}",
        f"- **Watchdog erstellt (UTC):** {watchdog_snapshot.generated_at_utc or '-'}",
        f"- **Watchdog Stale-Limit Min:** {stale_limit_text}",
        f"- **Watchdog Lag-Limit Min:** {lag_limit_text}",
        "",
        "",
        "| Symbol | Ampel | Watchdog | Sig Fresh | RT Fresh | Letztes Signal (UTC) | "
        "Letzter RT-Heartbeat (UTC) | Lag Sig→RT Min | Signale | Closes | TP | SL | Net PnL | PF | DD% | WR% | "
        "Ø Dauer Min | Begründung | Watchdog-Reason |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for status in statuses:
        lines.append(
            (
                "| {symbol} | {ampel} | {watchdog} | {sig_fresh} | {rt_fresh} | {last_sig} | "
                "{last_rt} | {lag} | {signale} | {closes} | {tp} | {sl} | {pnl} | "
                "{pf} | {dd} | {wr} | {dauer} | {grund} | {watchdog_reason} |"
            ).format(
                symbol=status.symbol,
                ampel=status.ampel,
                watchdog=status.watchdog_status or "-",
                sig_fresh="OK" if status.live_fresh else "STALE",
                rt_fresh="OK" if status.runtime_fresh else "STALE",
                last_sig=status.last_event_utc or "-",
                last_rt=status.last_runtime_utc or "-",
                lag=(
                    f"{status.csv_runtime_lag_min:.1f}"
                    if status.csv_runtime_lag_min is not None
                    else "-"
                ),
                signale=status.signale,
                closes=status.closes,
                tp=status.tp_closes,
                sl=status.sl_closes,
                pnl=(f"{status.net_pnl:+.2f}" if status.net_pnl is not None else "-"),
                pf=(
                    f"{status.profit_factor:.3f}"
                    if status.profit_factor is not None
                    else "-"
                ),
                dd=(
                    f"{status.max_drawdown_pct:.2f}"
                    if status.max_drawdown_pct is not None
                    else "-"
                ),
                wr=(
                    f"{status.win_rate_pct:.1f}"
                    if status.win_rate_pct is not None
                    else "-"
                ),
                dauer=(
                    f"{status.avg_dauer_min:.1f}"
                    if status.avg_dauer_min is not None
                    else "-"
                ),
                grund=status.begruendung,
                watchdog_reason=status.watchdog_reason or "-",
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **OK**: Frische Daten und keine kritischen operativen Auffälligkeiten.",
        "- **WATCH**: Daten sind frisch, aber Aktivität oder Close-KPIs sollten beobachtet werden.",
        "- **INCIDENT**: Stale Logs oder kritischer Drawdown – operativ sofort prüfen.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """CLI-Einstiegspunkt für das Phase-7-Tagesdashboard."""
    args = parse_args()
    symbols = tuple(
        sorted(
            {
                token.strip().upper()
                for token in str(args.symbols).split(",")
                if token.strip()
            }
        )
    )
    log_dir = resolve_log_dir(str(args.log_dir), symbols)
    output_dir = Path(args.output_dir)
    watchdog_snapshot = load_watchdog_snapshot(log_dir)

    statuses = [
        symbol_status_berechnen(
            symbol,
            log_dir=log_dir,
            hours=args.hours,
            timeframe=args.timeframe,
            watchdog_snapshot=watchdog_snapshot,
        )
        for symbol in symbols
    ]
    df = statuses_to_dataframe(statuses)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase7_daily_dashboard_latest.csv"
    df.to_csv(csv_path, index=False)
    md_path = markdown_dashboard_schreiben(
        statuses,
        output_dir,
        args.hours,
        args.timeframe,
        log_dir,
        watchdog_snapshot,
    )

    print("=" * 80)
    print("PHASE 7 DAILY DASHBOARD")
    print("=" * 80)
    print(f"Log-Quelle: {log_dir}")
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"CSV: {csv_path}")
    print(f"MD : {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
