"""
doc_drift_guard.py – Konsistenzprüfung für zentrale Projektdokumente.

Zweck:
    Verhindert, dass wichtige Markdown-Dateien inhaltlich auseinanderlaufen
    ("Doc Drift"), z.B. bei Phase-Status, operativen Symbolen oder
    Roadmap-Verweisen.

Läuft auf:
    Linux-Server (oder lokal in jeder Python-Umgebung mit Standardbibliothek)

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source .venv/bin/activate
    python reports/doc_drift_guard.py

Exit-Codes:
    0 = alle Checks bestanden
    1 = mindestens ein Check fehlgeschlagen
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Projektwurzel aus Datei-Pfad ableiten.
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class GuardResult:
    """Ein einzelnes Prüfergebnis des Guards.

    Args:
        ok: True bei bestandenem Check, sonst False.
        scope: Betroffene Datei oder Prüfkategorie.
        message: Lesbare Beschreibung des Ergebnisses.
    """

    ok: bool
    scope: str
    message: str


def _read_text(file_path: Path) -> str:
    """Liest eine Datei robust als UTF-8 Text.

    Args:
        file_path: Absoluter Dateipfad.

    Returns:
        Dateitext als String.
    """
    # UTF-8 ist im Repo konsistent gesetzt.
    return file_path.read_text(encoding="utf-8")


def _contains_all(text: str, needles: Iterable[str]) -> bool:
    """Prüft, ob alle Suchbegriffe im Text vorkommen.

    Args:
        text: Zu prüfender Text.
        needles: Liste der Pflicht-Strings.

    Returns:
        True, wenn jeder String enthalten ist.
    """
    # Alle Teilstrings müssen vorhanden sein, sonst ist der Check negativ.
    return all(needle in text for needle in needles)


def check_file_exists(file_path: Path) -> GuardResult:
    """Prüft, ob eine Datei vorhanden ist.

    Args:
        file_path: Zu prüfender Dateipfad.

    Returns:
        GuardResult mit Status und Meldung.
    """
    # Existenzprüfung ist die erste Schutzschicht gegen Drift.
    if file_path.exists():
        return GuardResult(
            True, str(file_path.relative_to(BASE_DIR)), "Datei vorhanden"
        )
    return GuardResult(False, str(file_path.relative_to(BASE_DIR)), "Datei fehlt")


def check_phase_7_present(file_path: Path) -> GuardResult:
    """Prüft, ob der Phase-7-Status enthalten ist.

    Args:
        file_path: Markdown-Datei mit Status-Information.

    Returns:
        GuardResult mit Prüfergebnis.
    """
    text = _read_text(file_path)
    # Tolerant prüfen: "Phase 7" genügt, damit Varianten wie Überschriften funktionieren.
    ok = "Phase 7" in text
    msg = "Phase 7 gefunden" if ok else "Phase 7 fehlt"
    return GuardResult(ok, str(file_path.relative_to(BASE_DIR)), msg)


def check_operational_symbols(file_path: Path) -> GuardResult:
    """Prüft, ob beide operativen Symbole dokumentiert sind.

    Args:
        file_path: Markdown-Datei mit Betriebs-Policy.

    Returns:
        GuardResult mit Prüfergebnis.
    """
    text = _read_text(file_path)
    # Beide Symbole müssen im selben Dokument vorkommen.
    ok = _contains_all(text, ["USDCAD", "USDJPY"])
    msg = (
        "Operative Symbole vorhanden (USDCAD/USDJPY)"
        if ok
        else "Operative Symbole unvollständig"
    )
    return GuardResult(ok, str(file_path.relative_to(BASE_DIR)), msg)


def check_roadmap_reference(file_path: Path) -> GuardResult:
    """Prüft den korrekten Dateinamen für Roadmap-Referenzen.

    Args:
        file_path: Markdown-Datei mit Roadmap-Verweis.

    Returns:
        GuardResult mit Prüfergebnis.
    """
    text = _read_text(file_path)
    # Uppercase-Altname soll nicht mehr verwendet werden.
    has_correct = "Roadmap.md" in text
    has_wrong = "ROADMAP.md" in text

    if has_correct and not has_wrong:
        return GuardResult(
            True, str(file_path.relative_to(BASE_DIR)), "Roadmap-Verweis korrekt"
        )
    if has_wrong:
        return GuardResult(
            False,
            str(file_path.relative_to(BASE_DIR)),
            "Veralteter Verweis 'ROADMAP.md' gefunden",
        )
    return GuardResult(
        False, str(file_path.relative_to(BASE_DIR)), "Kein Roadmap-Verweis gefunden"
    )


def check_roadmap_active_policy(file_path: Path) -> GuardResult:
    """Prüft die explizite Policy-Zeile in der Roadmap.

    Args:
        file_path: `Roadmap.md`.

    Returns:
        GuardResult mit Prüfergebnis.
    """
    text = _read_text(file_path)
    # Exakte Policy-Formulierung sichert schnelle Lesbarkeit zu Betriebsbeginn.
    pattern = r"Aktive operative Paare\s*\(Paper\)"
    ok = re.search(pattern, text) is not None
    msg = "Aktive-Paare-Policy vorhanden" if ok else "Aktive-Paare-Policy fehlt"
    return GuardResult(ok, str(file_path.relative_to(BASE_DIR)), msg)


def run_guard() -> list[GuardResult]:
    """Führt alle vordefinierten Guard-Checks aus.

    Returns:
        Liste aller Prüfergebnisse.
    """
    roadmap = BASE_DIR / "Roadmap.md"
    readme = BASE_DIR / "README.md"
    claude = BASE_DIR / "CLAUDE.md"
    copilot = BASE_DIR / ".github" / "copilot-instructions.md"
    plan_90d = BASE_DIR / "reports" / "paper_trading_90d_plan.md"

    checks: list[GuardResult] = []

    # 1) Existenz-Checks für zentrale Dateien.
    for file_path in [roadmap, readme, claude, copilot, plan_90d]:
        checks.append(check_file_exists(file_path))

    # Wenn eine Pflichtdatei fehlt, sparen wir Folgechecks und liefern klaren Fehler.
    if not all(result.ok for result in checks):
        return checks

    # 2) Inhaltliche Kern-Checks.
    checks.extend(
        [
            check_phase_7_present(readme),
            check_phase_7_present(claude),
            check_phase_7_present(copilot),
            check_operational_symbols(roadmap),
            check_operational_symbols(copilot),
            check_operational_symbols(claude),
            check_roadmap_reference(readme),
            check_roadmap_reference(claude),
            check_roadmap_reference(copilot),
            check_roadmap_reference(plan_90d),
            check_roadmap_active_policy(roadmap),
        ]
    )

    return checks


def print_report(results: list[GuardResult]) -> None:
    """Gibt einen konsolidierten Report in der Konsole aus.

    Args:
        results: Alle Prüfergebnisse.
    """
    # Übersicht am Kopf beschleunigt die Einordnung im CI/Terminal.
    total = len(results)
    failed = [result for result in results if not result.ok]
    passed = total - len(failed)

    print("=" * 72)
    print("DOC-DRIFT-GUARD – Ergebnis")
    print("=" * 72)
    print(f"Bestanden: {passed}/{total}")

    # Alle Ergebnisse ausgeben, damit auch grüne Checks sichtbar bleiben.
    for result in results:
        icon = "✅" if result.ok else "❌"
        print(f"{icon} [{result.scope}] {result.message}")

    if failed:
        print("-" * 72)
        print("Fehlgeschlagen: Bitte die oben markierten Dateien synchronisieren.")
    else:
        print("-" * 72)
        print("Alles konsistent. Kein Doc Drift erkannt.")


def parse_args() -> argparse.Namespace:
    """Parst CLI-Argumente.

    Returns:
        Namespace mit Argumentwerten.
    """
    parser = argparse.ArgumentParser(
        description="Prüft zentrale Markdown-Dateien auf Status-/Policy-Drift."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reserviert für spätere, schärfere Checks (aktuell funktional identisch).",
    )
    return parser.parse_args()


def main() -> None:
    """Startet den Guard-Prozess und setzt passenden Exit-Code."""
    # Argumente werden schon jetzt geparst, damit die CLI stabil bleibt.
    _ = parse_args()
    results = run_guard()
    print_report(results)

    # Exit-Code 1 signalisiert CI/Task-Runs zuverlässig einen Drift.
    has_failures = any(not result.ok for result in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
