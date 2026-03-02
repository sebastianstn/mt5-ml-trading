"""
verify_raw_data_copy.py – Prüft Rohdaten-Dateien für mehrere Timeframes auf Linux.

Zweck:
    Validiert, dass die CSV-Rohdaten nach dem Upload vom Windows-Laptop
    vollständig vorhanden sind und grundlegende Qualitätskriterien erfüllen.

Standard-Check:
    - Symbole: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
    - Timeframes: M15, M30, M60
    - Datei vorhanden
    - Mindestanzahl Zeilen
    - Pflichtspalten vorhanden

Verwendung (Linux-Server):
    .venv/bin/python scripts/verify_raw_data_copy.py --timeframes M15 M30 M60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
PFLICHT_SPALTEN = ["time", "open", "high", "low", "close", "volume", "spread"]


def check_file(path: Path, min_rows: int) -> tuple[bool, str]:
    """Prüft eine einzelne CSV-Datei auf Existenz und Basisqualität."""
    if not path.exists():
        return False, "fehlt"

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"lesefehler: {exc}"

    fehlende = [c for c in PFLICHT_SPALTEN if c not in df.columns]
    if fehlende:
        return False, f"fehlende_spalten: {fehlende}"

    if len(df) < min_rows:
        return False, f"zu_wenig_zeilen: {len(df)} < {min_rows}"

    return True, f"ok ({len(df)} zeilen)"


def main() -> None:
    """Startet die Gesamtprüfung für Symbol × Timeframe."""
    parser = argparse.ArgumentParser(description="Prüft hochgeladene Rohdaten-Dateien")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["M15", "M30", "M60"],
        help="Zu prüfende Timeframes (Standard: M15 M30 M60)",
    )
    parser.add_argument(
        "--min_rows",
        type=int,
        default=1000,
        help="Mindestanzahl Zeilen pro CSV (Standard: 1000)",
    )
    args = parser.parse_args()

    gesamt = 0
    ok_count = 0

    print("=" * 72)
    print("ROHDATEN-CHECK (Linux-Server)")
    print(f"Data-Verzeichnis: {DATA_DIR}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print("=" * 72)

    for sym in SYMBOLE:
        for tf in args.timeframes:
            gesamt += 1
            path = DATA_DIR / f"{sym}_{tf}.csv"
            ok, msg = check_file(path, args.min_rows)
            status = "✓" if ok else "✗"
            print(f"{status} {sym}_{tf}.csv -> {msg}")
            if ok:
                ok_count += 1

    print("-" * 72)
    print(f"Ergebnis: {ok_count}/{gesamt} Dateien OK")
    if ok_count != gesamt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
