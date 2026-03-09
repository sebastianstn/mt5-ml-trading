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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    size_bytes: int | None
    fresh: bool


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
            size_bytes=None,
            fresh=False,
        )

    age_min = file_age_minutes(file_path)
    size = file_path.stat().st_size
    is_fresh = age_min <= max_age_minutes

    return FileCheckResult(
        symbol=symbol,
        kind=kind,
        path=file_path,
        exists=True,
        age_minutes=age_min,
        size_bytes=size,
        fresh=is_fresh,
    )


def format_row(result: FileCheckResult) -> str:
    """Formatiert ein Prüfergebnis für die Terminal-Ausgabe.

    Args:
        result: Einzelnes Dateiergebnis.

    Returns:
        Formatierte Tabellenzeile.
    """
    status = (
        "OK"
        if result.exists and result.fresh
        else "FEHLT" if not result.exists else "STALE"
    )
    age_str = "-" if result.age_minutes is None else f"{result.age_minutes:6.1f}"
    size_str = "-" if result.size_bytes is None else f"{result.size_bytes:8d}"
    return (
        f"{result.symbol:7} | {result.kind:7} | {status:6} | "
        f"{age_str:>8} | {size_str:>8} | {result.path.name}"
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

    print("=" * 92)
    print("LIVE LOG SYNC VERIFY")
    print(
        f"log_dir={log_dir} | symbols={','.join(symbols)} | max_age_minutes={args.max_age_minutes}"
    )
    print("=" * 92)
    print("Symbol  | Typ     | Status | Alter(min) | Größe(B) | Datei")
    print("-" * 92)

    for res in results:
        print(format_row(res))

    # Gesamtstatus: alle Pflichtdateien müssen existieren und frisch sein.
    all_ok = all(r.exists and r.fresh for r in results)
    print("-" * 92)
    print("SYNC_OK" if all_ok else "SYNC_NICHT_OK")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
