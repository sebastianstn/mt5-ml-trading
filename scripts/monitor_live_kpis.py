"""
monitor_live_kpis.py – KPI-Monitoring für MT5 Live-/Paper-Trader Logs.

Ziel:
    Dieses Skript fasst die Heartbeat-/Signal-CSV-Logs aus dem Live-Trader
    pro Symbol zusammen und liefert Monitoring-KPIs für Phase 7.

Beispiel:
    python scripts/monitor_live_kpis.py --log_dir logs --hours 72 --timeframe H1
"""

# Standard-Bibliotheken
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Datenverarbeitung
import pandas as pd


# Regime-Namen (konsistent mit live_trader.py)
REGIME_NAMEN: Dict[int, str] = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

# Standard aktive Symbole laut Projekt-Policy
STANDARD_SYMBOLE: List[str] = ["USDCAD", "USDJPY"]


@dataclass
class SymbolKPI:
    """Container für berechnete KPIs eines Symbols."""

    symbol: str
    events: int
    kerzen_heartbeat: int
    signale_total: int
    signale_long: int
    signale_short: int
    signalrate_pct: float
    avg_prob_all_pct: float
    avg_prob_signale_pct: float
    last_event_utc: Optional[datetime]
    minutes_since_last: Optional[float]
    stale: bool
    regime_counts: Dict[str, int]


def parse_args() -> argparse.Namespace:
    """
    Parst CLI-Argumente für das Monitoring.

    Returns:
        Namespace mit allen Nutzerparametern.
    """
    parser = argparse.ArgumentParser(
        description=(
            "KPI-Monitoring für live_trader CSV-Logs (Heartbeat + Signale). "
            "Empfohlen für Phase 7 Überwachung."
        )
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Pfad zum Log-Ordner mit *_{signals|closes}.csv (Standard: logs)",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="_signals.csv",
        help=(
            "Datei-Suffix pro Symbol (Standard: _signals.csv). "
            "Beispiel: --file_suffix _closes.csv"
        ),
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(STANDARD_SYMBOLE),
        help=(
            "Komma-getrennte Symbole (Standard: USDCAD,USDJPY). "
            "Beispiel: --symbols USDCAD,USDJPY"
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Zeitfenster in Stunden für KPI-Berechnung (Standard: 72)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        choices=["H1", "M30", "M15", "M5_TWO_STAGE"],
        default="H1",
        help="Trader-Zeitrahmen für Stale-Check (Standard: H1)",
    )
    parser.add_argument(
        "--stale_factor",
        type=float,
        default=1.5,
        help=(
            "Stale-Faktor relativ zur Kerzenlänge. "
            "Bei H1 und 1.5 => stale ab >90 Minuten ohne Event."
        ),
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default="",
        help=(
            "Optionaler Ausgabepfad für KPI-Tabelle als CSV. "
            "Beispiel: reports/live_kpis_latest.csv"
        ),
    )
    return parser.parse_args()


def timeframe_minutes(timeframe: str) -> int:
    """
    Übersetzt Timeframe-String in Minuten.

    Args:
        timeframe: Timeframe als String (H1, M30, M15).

    Returns:
        Anzahl Minuten pro Kerze.
    """
    if timeframe == "H1":
        return 60
    if timeframe == "M30":
        return 30
    if timeframe == "M5_TWO_STAGE":
        return 5
    return 15


def load_symbol_log(log_dir: Path, symbol: str, file_suffix: str) -> pd.DataFrame:
    """
    Lädt die CSV-Logdatei eines Symbols.

    Args:
        log_dir: Ordner mit Logdateien.
        symbol: Handelssymbol (z.B. USDCAD).

    Returns:
        Bereinigtes DataFrame mit UTC-Zeitspalte.

    Raises:
        FileNotFoundError: Wenn die CSV-Datei nicht existiert.
        ValueError: Wenn Pflichtspalten fehlen.
    """
    log_file = log_dir / f"{symbol}{file_suffix}"
    if not log_file.exists():
        raise FileNotFoundError(f"Logdatei nicht gefunden: {log_file}")

    # CSV einlesen und Time-Spalte robust in UTC umwandeln.
    df = pd.read_csv(log_file)

    required_cols = {"time", "signal", "prob", "regime", "regime_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Pflichtspalten fehlen in {log_file.name}: {sorted(missing)}")

    # Zeitspalte parsen und explizit auf UTC normalisieren.
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).copy()

    # Numerische Spalten robust casten, fehlerhafte Werte zu NaN.
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")

    # Zeilen mit ungültigem Signal verwerfen (Signal ist Kernlogik).
    df = df.dropna(subset=["signal"]).copy()
    df["signal"] = df["signal"].astype(int)

    # Bei fehlendem Regime-Namen aus Mapping ergänzen.
    df["regime"] = df["regime"].fillna(-1).astype(int)
    df["regime_name"] = df["regime_name"].fillna(df["regime"].map(REGIME_NAMEN))
    df["regime_name"] = df["regime_name"].fillna("Unbekannt")

    return df


def filter_time_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Filtert ein DataFrame auf ein rückblickendes Zeitfenster.

    Args:
        df: Vollständiges Log-DataFrame.
        hours: Rückblick in Stunden.

    Returns:
        Gefiltertes DataFrame im gewünschten Zeitraum.
    """
    # Aktuelle UTC-Zeit als Referenz für reproduzierbare Monitoring-Fenster.
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    return df[df["time"] >= cutoff].copy()


def compute_symbol_kpis(
    symbol: str,
    df_window: pd.DataFrame,
    timeframe: str,
    stale_factor: float,
) -> SymbolKPI:
    """
    Berechnet Monitoring-KPIs für ein einzelnes Symbol.

    Args:
        symbol: Handelssymbol.
        df_window: Bereits auf Zeitfenster gefilterte Logdaten.
        timeframe: Timeframe (H1/M30/M15) für Stale-Schwelle.
        stale_factor: Multiplikator auf Kerzenlänge für Stale-Status.

    Returns:
        SymbolKPI mit aggregierten Metriken.
    """
    events = int(len(df_window))

    # Heartbeat/No-Signal werden als signal == 0 geloggt.
    kerzen_heartbeat = int((df_window["signal"] == 0).sum())

    # Signale sind alle Nicht-Null-Werte (-1 Short, 2 Long).
    signale_total = int((df_window["signal"] != 0).sum())
    signale_long = int((df_window["signal"] == 2).sum())
    signale_short = int((df_window["signal"] == -1).sum())

    signalrate_pct = (signale_total / events * 100.0) if events > 0 else 0.0

    avg_prob_all_pct = (
        float(df_window["prob"].dropna().mean() * 100.0)
        if not df_window["prob"].dropna().empty
        else 0.0
    )

    prob_signale = df_window.loc[df_window["signal"] != 0, "prob"].dropna()
    avg_prob_signale_pct = (
        float(prob_signale.mean() * 100.0) if not prob_signale.empty else 0.0
    )

    # Letztes Event für Stale-Überwachung bestimmen.
    last_event_utc: Optional[datetime] = None
    minutes_since_last: Optional[float] = None
    stale = True

    if events > 0:
        last_ts = df_window["time"].max()
        last_event_utc = last_ts.to_pydatetime()

        now_utc = datetime.now(timezone.utc)
        # mypy/pylance: innerhalb dieses Blocks ist last_event_utc garantiert gesetzt.
        assert last_event_utc is not None
        minutes_since_last = (now_utc - last_event_utc).total_seconds() / 60.0

        # Stale wenn letzter Event älter als timeframe_minutes * stale_factor.
        stale_limit_minutes = timeframe_minutes(timeframe) * stale_factor
        stale = bool(minutes_since_last > stale_limit_minutes)

    # Regime-Verteilung als Dictionary für schnellen Überblick.
    regime_counts_series = df_window["regime_name"].value_counts(dropna=False)
    regime_counts = {str(k): int(v) for k, v in regime_counts_series.to_dict().items()}

    return SymbolKPI(
        symbol=symbol,
        events=events,
        kerzen_heartbeat=kerzen_heartbeat,
        signale_total=signale_total,
        signale_long=signale_long,
        signale_short=signale_short,
        signalrate_pct=signalrate_pct,
        avg_prob_all_pct=avg_prob_all_pct,
        avg_prob_signale_pct=avg_prob_signale_pct,
        last_event_utc=last_event_utc,
        minutes_since_last=minutes_since_last,
        stale=stale,
        regime_counts=regime_counts,
    )


def kpis_to_dataframe(kpis: List[SymbolKPI]) -> pd.DataFrame:
    """
    Wandelt KPI-Objekte in tabellarisches DataFrame um.

    Args:
        kpis: Liste von SymbolKPI-Objekten.

    Returns:
        DataFrame mit einer Zeile pro Symbol.
    """
    rows: List[Dict[str, object]] = []
    for kpi in kpis:
        rows.append(
            {
                "symbol": kpi.symbol,
                "events": kpi.events,
                "heartbeats_signal0": kpi.kerzen_heartbeat,
                "signale_total": kpi.signale_total,
                "signale_long": kpi.signale_long,
                "signale_short": kpi.signale_short,
                "signalrate_pct": round(kpi.signalrate_pct, 2),
                "avg_prob_all_pct": round(kpi.avg_prob_all_pct, 2),
                "avg_prob_signale_pct": round(kpi.avg_prob_signale_pct, 2),
                "last_event_utc": (
                    kpi.last_event_utc.strftime("%Y-%m-%d %H:%M:%S")
                    if kpi.last_event_utc is not None
                    else ""
                ),
                "minutes_since_last": (
                    round(kpi.minutes_since_last, 1)
                    if kpi.minutes_since_last is not None
                    else None
                ),
                "status": "STALE" if kpi.stale else "OK",
                "regime_counts": ", ".join(
                    [f"{name}:{count}" for name, count in kpi.regime_counts.items()]
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    df_kpi: pd.DataFrame, hours: int, timeframe: str, stale_factor: float
) -> None:
    """
    Gibt KPI-Zusammenfassung lesbar in die Konsole aus.

    Args:
        df_kpi: KPI-Tabelle.
        hours: Rückblick-Fenster in Stunden.
        timeframe: Timeframe für Kontextanzeige.
        stale_factor: Stale-Faktor für Kontextanzeige.
    """
    print("=" * 90)
    print("LIVE KPI MONITORING (Phase 7)")
    print(
        f"Fenster: letzte {hours}h | Timeframe: {timeframe} | "
        f"Stale-Grenze: {timeframe_minutes(timeframe) * stale_factor:.0f} Minuten"
    )
    print("=" * 90)

    if df_kpi.empty:
        print("Keine KPI-Daten vorhanden.")
        return

    # Kerntabelle ausgeben.
    columns_main = [
        "symbol",
        "events",
        "heartbeats_signal0",
        "signale_total",
        "signalrate_pct",
        "avg_prob_all_pct",
        "avg_prob_signale_pct",
        "minutes_since_last",
        "status",
    ]
    print(df_kpi[columns_main].to_string(index=False))

    print("\nRegime-Verteilung pro Symbol:")
    for _, row in df_kpi.iterrows():
        print(f"- {row['symbol']}: {row['regime_counts']}")


def main() -> None:
    """Programm-Einstiegspunkt für KPI-Monitoring."""
    args = parse_args()

    # Log-Pfad auflösen und validieren.
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log-Ordner nicht gefunden: {log_dir.resolve()}")

    # Symbol-Liste robust aus CLI lesen.
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("Keine Symbole angegeben. Beispiel: --symbols USDCAD,USDJPY")

    # Pro Symbol Daten laden, filtern und KPIs berechnen.
    kpis: List[SymbolKPI] = []
    for symbol in symbols:
        try:
            df_symbol = load_symbol_log(
                log_dir=log_dir,
                symbol=symbol,
                file_suffix=args.file_suffix,
            )
            df_window = filter_time_window(df=df_symbol, hours=args.hours)
            kpi = compute_symbol_kpis(
                symbol=symbol,
                df_window=df_window,
                timeframe=args.timeframe,
                stale_factor=args.stale_factor,
            )
            kpis.append(kpi)
        except FileNotFoundError as exc:
            # Fehlende Dateien nicht hart abbrechen, sondern sichtbar melden.
            print(f"[WARN] {exc}")
        except ValueError as exc:
            # Formatprobleme pro Symbol ebenfalls als Warnung ausgeben.
            print(f"[WARN] {symbol}: {exc}")

    # Ergebnis als DataFrame aufbereiten und sortieren.
    df_kpi = kpis_to_dataframe(kpis)
    if not df_kpi.empty:
        df_kpi = df_kpi.sort_values(by=["status", "symbol"], ascending=[True, True])

    # Konsole-Zusammenfassung ausgeben.
    print_summary(
        df_kpi=df_kpi,
        hours=args.hours,
        timeframe=args.timeframe,
        stale_factor=args.stale_factor,
    )

    # Optional Export in CSV für Reports/Dashboard.
    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df_kpi.to_csv(export_path, index=False)
        print(f"\nKPI-Export geschrieben: {export_path}")


if __name__ == "__main__":
    main()
