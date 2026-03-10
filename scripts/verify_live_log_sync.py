"""
verify_live_log_sync.py – Verifiziert Live-Log-Sync (Windows -> Linux).

Zweck:
    Prüft pro Symbol, ob die erwarteten Log-Dateien vorhanden und frisch sind.
    Das Skript ist als schneller Operativ-Check für Phase 7 gedacht.

Beispiele:
    python scripts/verify_live_log_sync.py
    python scripts/verify_live_log_sync.py --symbols USDCAD,USDJPY --max_age_minutes 10
    python scripts/verify_live_log_sync.py --check_closes

Exit-Code:
    0 -> alle Pflichtdateien vorhanden und frisch
    1 -> mindestens eine Pflichtdatei fehlt oder ist stale
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


WATCHDOG_JSON_NAME = "live_log_watchdog_latest.json"


@dataclass
class FileCheckResult:
    """Ergebnis einer Einzeldatei-Prüfung.

    Args:
        symbol: Handelssymbol (z. B. USDCAD).
        kind: Dateityp (signals oder closes).
        path: Voller Dateipfad.
        exists: True wenn Datei existiert.
        age_minutes: Alter in Minuten, None wenn Datei fehlt.
        size_bytes: Dateigröße in Byte, None wenn Datei fehlt.
        fresh: True wenn Datei jünger/gleich max_age_minutes ist.
    """

    symbol: str
    kind: str
    path: Path
    exists: bool
    age_minutes: float | None
    mtime_age_minutes: float | None
    size_bytes: int | None
    fresh: bool
    content_timestamp_utc: datetime | None


@dataclass
class WatchdogCheckResult:
    """Ergebnis der Linux-seitigen Watchdog-Prüfung."""

    path: Path
    exists: bool
    age_minutes: float | None
    mtime_age_minutes: float | None
    fresh: bool
    overall_status: str
    generated_at_utc: datetime | None
    healthy: bool
    reason: str


def parse_args() -> argparse.Namespace:
    """Parst CLI-Argumente.

    Returns:
        Namespace mit allen Parametern.
    """
    parser = argparse.ArgumentParser(
        description="Verifiziert Frische und Existenz von *_signals.csv / *_closes.csv"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Log-Ordner auf Linux (Standard: logs)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="USDCAD,USDJPY",
        help="Komma-getrennte Symbol-Liste (Standard: USDCAD,USDJPY)",
    )
    parser.add_argument(
        "--max_age_minutes",
        type=float,
        default=10.0,
        help="Maximales Dateialter in Minuten für FRESH (Standard: 10)",
    )
    parser.add_argument(
        "--check_closes",
        action="store_true",
        help="Prüft zusätzlich *_closes.csv als Pflichtdatei",
    )
    parser.add_argument(
        "--check_watchdog",
        action="store_true",
        help=(
            "Prüft zusätzlich live_log_watchdog_latest.json auf Existenz, Frische "
            "und overall_status != INCIDENT"
        ),
    )
    return parser.parse_args()


def file_age_minutes(file_path: Path) -> float:
    """Berechnet Dateialter in Minuten.

    Args:
        file_path: Zu prüfender Dateipfad.

    Returns:
        Alter in Minuten (float).
    """
    # Letzte Modifikation in UTC lesen, damit Zeitzonen eindeutig sind.
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    now_utc = datetime.now(tz=timezone.utc)
    # Alter als Minutenwert für einfache Operativ-Grenzen zurückgeben.
    return (now_utc - mtime).total_seconds() / 60.0


def csv_content_timestamp_utc(file_path: Path) -> datetime | None:
    """Liest den letzten gültigen UTC-Zeitstempel aus der CSV-Inhaltsspalte `time`.

    Args:
        file_path: Pfad zur Signal-/Close-CSV.

    Returns:
        Letzter gültiger UTC-Zeitstempel aus dem Inhalt oder None.
    """
    last_ts: datetime | None = None
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "time" not in reader.fieldnames:
            return None

        for row in reader:
            raw_value = str(row.get("time", "")).strip()
            if not raw_value:
                continue
            try:
                parsed = datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            last_ts = parsed

    return last_ts


def timestamp_age_minutes(timestamp_utc: datetime) -> float:
    """Berechnet das Alter eines UTC-Zeitstempels in Minuten."""
    now_utc = datetime.now(tz=timezone.utc)
    return (now_utc - timestamp_utc).total_seconds() / 60.0


def json_content_timestamp_utc(file_path: Path) -> datetime | None:
    """Liest den UTC-Zeitstempel `generated_at_utc` aus der Watchdog-JSON."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None

    raw_value = str(payload.get("generated_at_utc", "")).strip()
    if not raw_value:
        return None

    try:
        return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def check_watchdog_file(log_dir: Path, max_age_minutes: float) -> WatchdogCheckResult:
    """Prüft die synchronisierte Watchdog-Datei auf Existenz, Frische und Status."""
    file_path = log_dir / WATCHDOG_JSON_NAME
    if not file_path.exists():
        return WatchdogCheckResult(
            path=file_path,
            exists=False,
            age_minutes=None,
            mtime_age_minutes=None,
            fresh=False,
            overall_status="FEHLT",
            generated_at_utc=None,
            healthy=False,
            reason="Watchdog-Datei fehlt",
        )

    mtime_age_min = file_age_minutes(file_path)
    generated_at_utc = json_content_timestamp_utc(file_path)
    age_min = (
        timestamp_age_minutes(generated_at_utc)
        if generated_at_utc is not None
        else mtime_age_min
    )
    fresh = age_min <= max_age_minutes

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return WatchdogCheckResult(
            path=file_path,
            exists=True,
            age_minutes=age_min,
            mtime_age_minutes=mtime_age_min,
            fresh=False,
            overall_status="UNGUELTIG",
            generated_at_utc=generated_at_utc,
            healthy=False,
            reason="Watchdog-Datei ist kein gültiges JSON",
        )

    overall_status = str(payload.get("overall_status", "")).upper().strip() or "-"
    if not fresh:
        return WatchdogCheckResult(
            path=file_path,
            exists=True,
            age_minutes=age_min,
            mtime_age_minutes=mtime_age_min,
            fresh=False,
            overall_status=overall_status,
            generated_at_utc=generated_at_utc,
            healthy=False,
            reason="Watchdog-Datei stale",
        )
    if overall_status == "INCIDENT":
        return WatchdogCheckResult(
            path=file_path,
            exists=True,
            age_minutes=age_min,
            mtime_age_minutes=mtime_age_min,
            fresh=True,
            overall_status=overall_status,
            generated_at_utc=generated_at_utc,
            healthy=False,
            reason="Watchdog meldet INCIDENT",
        )

    return WatchdogCheckResult(
        path=file_path,
        exists=True,
        age_minutes=age_min,
        mtime_age_minutes=mtime_age_min,
        fresh=True,
        overall_status=overall_status,
        generated_at_utc=generated_at_utc,
        healthy=True,
        reason="Watchdog ok",
    )


def check_single_file(
    symbol: str,
    kind: str,
    file_path: Path,
    max_age_minutes: float,
) -> FileCheckResult:
    """Prüft Existenz und Frische einer Log-Datei.

    Args:
        symbol: Handelssymbol.
        kind: Art der Datei (signals oder closes).
        file_path: Datei, die geprüft wird.
        max_age_minutes: Frische-Grenze in Minuten.

    Returns:
        FileCheckResult mit allen Prüffeldern.
    """
    if not file_path.exists():
        return FileCheckResult(
            symbol=symbol,
            kind=kind,
            path=file_path,
            exists=False,
            age_minutes=None,
            mtime_age_minutes=None,
            size_bytes=None,
            fresh=False,
            content_timestamp_utc=None,
        )

    mtime_age_min = file_age_minutes(file_path)
    size = file_path.stat().st_size
    content_timestamp = csv_content_timestamp_utc(file_path)
    age_min = (
        timestamp_age_minutes(content_timestamp)
        if content_timestamp is not None
        else mtime_age_min
    )
    is_fresh = age_min <= max_age_minutes

    return FileCheckResult(
        symbol=symbol,
        kind=kind,
        path=file_path,
        exists=True,
        age_minutes=age_min,
        mtime_age_minutes=mtime_age_min,
        size_bytes=size,
        fresh=is_fresh,
        content_timestamp_utc=content_timestamp,
    )


def format_row(result: FileCheckResult) -> str:
    """Formatiert ein Prüfergebnis für die Terminal-Ausgabe.

    Args:
        result: Einzelnes Dateiergebnis.

    Returns:
        Formatierte Tabellenzeile.
    """
    status = "FEHLT"
    if result.exists and result.fresh:
        status = "OK"
    elif result.exists:
        status = "DRIFT"
        if result.mtime_age_minutes is None or result.mtime_age_minutes > 10.0:
            status = "STALE"

    age_str = "-" if result.age_minutes is None else f"{result.age_minutes:6.1f}"
    mtime_str = (
        "-" if result.mtime_age_minutes is None else f"{result.mtime_age_minutes:6.1f}"
    )
    size_str = "-" if result.size_bytes is None else f"{result.size_bytes:8d}"
    return (
        f"{result.symbol:7} | {result.kind:7} | {status:6} | "
        f"{age_str:>8} | {mtime_str:>8} | {size_str:>8} | {result.path.name}"
    )


def format_watchdog_row(result: WatchdogCheckResult) -> str:
    """Formatiert das Watchdog-Ergebnis für die Terminal-Ausgabe."""
    status = "OK" if result.healthy else "INCIDENT"
    age_str = "-" if result.age_minutes is None else f"{result.age_minutes:6.1f}"
    mtime_str = (
        "-" if result.mtime_age_minutes is None else f"{result.mtime_age_minutes:6.1f}"
    )
    return (
        f"WATCHDG | global  | {status:8} | {age_str:>8} | {mtime_str:>8} | "
        f"{result.overall_status:>8} | {result.path.name} | {result.reason}"
    )


def main() -> None:
    """Startet die Sync-Verifikation und setzt passenden Exit-Code."""
    args = parse_args()

    log_dir = Path(args.log_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not symbols:
        raise SystemExit("Keine Symbole angegeben. Beispiel: --symbols USDCAD,USDJPY")

    results: list[FileCheckResult] = []

    # Für jedes Symbol immer signals prüfen.
    for symbol in symbols:
        signals_path = log_dir / f"{symbol}_signals.csv"
        results.append(
            check_single_file(
                symbol=symbol,
                kind="signals",
                file_path=signals_path,
                max_age_minutes=args.max_age_minutes,
            )
        )

        # Closes nur optional als Pflichtprüfung hinzufügen.
        if args.check_closes:
            closes_path = log_dir / f"{symbol}_closes.csv"
            results.append(
                check_single_file(
                    symbol=symbol,
                    kind="closes",
                    file_path=closes_path,
                    max_age_minutes=args.max_age_minutes,
                )
            )

    watchdog_result = (
        check_watchdog_file(log_dir=log_dir, max_age_minutes=args.max_age_minutes)
        if args.check_watchdog
        else None
    )

    print("=" * 120)
    print("LIVE LOG SYNC VERIFY")
    print(
        f"log_dir={log_dir} | symbols={','.join(symbols)} | max_age_minutes={args.max_age_minutes}"
    )
    print("=" * 120)
    print("Symbol  | Typ     | Status | EventAlter | mtimeAlt | Größe(B) | Datei")
    print("-" * 120)

    for res in results:
        print(format_row(res))

    if watchdog_result is not None:
        print("-" * 120)
        print("Quelle  | Scope   | Status   | EventAlter | mtimeAlt | Overall  | Datei | Hinweis")
        print("-" * 120)
        print(format_watchdog_row(watchdog_result))

    # Gesamtstatus: alle Pflichtdateien müssen existieren und frisch sein.
    all_ok = all(r.exists and r.fresh for r in results)
    if watchdog_result is not None:
        all_ok = all_ok and watchdog_result.healthy
    print("-" * 120)
    print("SYNC_OK" if all_ok else "SYNC_NICHT_OK")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
