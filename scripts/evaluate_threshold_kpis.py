"""
evaluate_threshold_kpis.py - Schwellenvergleich fuer Live-/Paper-Signal-Logs.

Ziel:
    Vergleicht mehrere Wahrscheinlichkeits-Schwellen (z. B. 0.50/0.55/0.60)
    auf Basis der bestehenden live_trader-CSV-Logs und erzeugt:
      1) detail.csv  -> symbol x threshold x regime
      2) summary.csv -> symbol x threshold inkl. Ranking

Laufort:
    Linux-Server

Beispiel:
    python scripts/evaluate_threshold_kpis.py --hours 72
    python scripts/evaluate_threshold_kpis.py --symbols USDCAD,USDJPY --thresholds 0.50,0.55,0.60
"""

from __future__ import annotations

# Standard-Bibliotheken
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Datenverarbeitung
import pandas as pd


# Regime-Mapping konsistent zu live/live_trader.py
REGIME_NAMEN: Dict[int, str] = {
    0: "Seitwaerts",
    1: "Aufwaertstrend",
    2: "Abwaertstrend",
    3: "Hohe_Volatilitaet",
}

# Standard-Symbole laut aktiver Betriebs-Policy.
STANDARD_SYMBOLE: List[str] = ["USDCAD", "USDJPY"]

# Standard-Schwellen fuer den gewuenschten Vergleich.
STANDARD_THRESHOLDS: List[float] = [0.50, 0.55, 0.60]


@dataclass(frozen=True)
class EvalConfig:
    """Konfiguration fuer die KPI-Auswertung."""

    log_dir: Path
    symbols: Tuple[str, ...]
    thresholds: Tuple[float, ...]
    threshold_operator: str
    timeframe: str
    stale_factor: float
    hours: Optional[int]
    start_utc: Optional[datetime]
    end_utc: Optional[datetime]
    output_dir: Path


def parse_args() -> argparse.Namespace:
    """
    Liest und validiert CLI-Argumente.

    Returns:
        Argument-Namespace fuer die Auswertung.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Vergleicht Signal-Schwellen fuer live_trader-Logs und erzeugt "
            "Summary/Detail-CSV fuer Symbol, Regime und Threshold."
        )
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Ordner mit *_live_trades.csv (Standard: logs)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(STANDARD_SYMBOLE),
        help="Komma-getrennte Symbole (Standard: USDCAD,USDJPY)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join([f"{v:.2f}" for v in STANDARD_THRESHOLDS]),
        help="Komma-getrennte Schwellen (Standard: 0.50,0.55,0.60)",
    )
    parser.add_argument(
        "--threshold_operator",
        choices=["ge", "gt"],
        default="ge",
        help=(
            "Vergleichsoperator fuer Schwelle: ge => >= (Live-Logik), "
            "gt => > (Backtest-Logik). Standard: ge"
        ),
    )
    parser.add_argument(
        "--timeframe",
        choices=["H1", "M30", "M15", "M5"],
        default="M5",
        help="Timeframe fuer Frequenz-/Stale-Auswertung (Standard: M5)",
    )
    parser.add_argument(
        "--stale_factor",
        type=float,
        default=1.5,
        help="Stale-Grenze als Faktor * Timeframe-Minuten (Standard: 1.5)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Rueckblick in Stunden (Standard: 72). Ignoriert wenn --start gesetzt.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help="Expliziter Startzeitpunkt in ISO-Format, z. B. 2026-03-06T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="",
        help="Explizites Enddatum in ISO-Format (optional, Default: jetzt UTC)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/threshold_eval",
        help="Ausgabeordner fuer CSV-Artefakte (Standard: reports/threshold_eval)",
    )
    return parser.parse_args()


def parse_iso_utc(raw_value: str) -> datetime:
    """
    Parst einen ISO-Zeitstempel robust und normalisiert auf UTC.

    Args:
        raw_value: Zeitwert als String.

    Returns:
        UTC-normalisierter datetime-Wert.

    Raises:
        ValueError: Wenn das Format ungueltig ist.
    """
    cleaned = raw_value.strip()
    if not cleaned:
        raise ValueError("Leerer Zeitstempel ist nicht erlaubt.")
    parsed = pd.to_datetime(cleaned, utc=True, errors="raise")
    # In Python-datetime umwandeln, damit Typen explizit bleiben.
    return parsed.to_pydatetime()


def timeframe_minutes(timeframe: str) -> int:
    """
    Liefert die Minuten pro Kerze fuer den Timeframe.

    Args:
        timeframe: Timeframe-String.

    Returns:
        Minuten pro Kerze.
    """
    mapping = {"H1": 60, "M30": 30, "M15": 15, "M5": 5}
    return mapping[timeframe]


def parse_symbols(raw_symbols: str) -> Tuple[str, ...]:
    """
    Zerlegt und validiert Symbol-Liste.

    Args:
        raw_symbols: Komma-getrennte Symbol-Strings.

    Returns:
        Normalisierte Symbol-Tupel.
    """
    symbols = tuple(sorted({s.strip().upper() for s in raw_symbols.split(",") if s.strip()}))
    if not symbols:
        raise ValueError("Keine gueltigen Symbole angegeben.")
    return symbols


def parse_thresholds(raw_thresholds: str) -> Tuple[float, ...]:
    """
    Zerlegt und validiert Threshold-Liste.

    Args:
        raw_thresholds: Komma-getrennte Schwellen als String.

    Returns:
        Sortiertes Tuple mit Schwellenwerten.
    """
    values: List[float] = []
    for raw in raw_thresholds.split(","):
        text = raw.strip()
        if not text:
            continue
        value = float(text)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Schwelle ausserhalb [0,1]: {value}")
        values.append(round(value, 4))
    if not values:
        raise ValueError("Keine gueltigen Schwellen angegeben.")
    return tuple(sorted(set(values)))


def build_config(args: argparse.Namespace) -> EvalConfig:
    """
    Baut eine typsichere Auswertungskonfiguration.

    Args:
        args: CLI-Argumente.

    Returns:
        Fertige EvalConfig.
    """
    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    hours: Optional[int] = int(args.hours)

    # Wenn Start gesetzt ist, verwenden wir Start/End statt Hours.
    if args.start:
        start_utc = parse_iso_utc(args.start)
        end_utc = parse_iso_utc(args.end) if args.end else datetime.now(timezone.utc)
        hours = None
        if end_utc < start_utc:
            raise ValueError("--end darf nicht vor --start liegen.")

    return EvalConfig(
        log_dir=Path(args.log_dir),
        symbols=parse_symbols(args.symbols),
        thresholds=parse_thresholds(args.thresholds),
        threshold_operator=str(args.threshold_operator),
        timeframe=str(args.timeframe),
        stale_factor=float(args.stale_factor),
        hours=hours,
        start_utc=start_utc,
        end_utc=end_utc,
        output_dir=Path(args.output_dir),
    )


def load_symbol_log(log_dir: Path, symbol: str) -> pd.DataFrame:
    """
    Laedt und normalisiert die Logdatei eines Symbols.

    Args:
        log_dir: Verzeichnis mit Live-CSV-Logs.
        symbol: Symbolname.

    Returns:
        Bereinigtes DataFrame mit Standardspalten.

    Raises:
        FileNotFoundError: Wenn die Datei fehlt.
        ValueError: Wenn Pflichtspalten fehlen.
    """
    log_path = log_dir / f"{symbol}_live_trades.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Logdatei fehlt: {log_path}")

    df = pd.read_csv(log_path)
    required_cols = {"time", "signal", "prob", "regime"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Pflichtspalten fehlen in {log_path.name}: {sorted(missing)}")

    # Zeit in UTC normieren.
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    # Numerische Kerndaten robust casten.
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["regime"] = pd.to_numeric(df["regime"], errors="coerce")
    df = df.dropna(subset=["time", "signal", "prob", "regime"]).copy()
    df["signal"] = df["signal"].astype(int)
    df["regime"] = df["regime"].astype(int)

    # Regime-Namen konsistent aufbauen.
    if "regime_name" not in df.columns:
        df["regime_name"] = df["regime"].map(REGIME_NAMEN)
    df["regime_name"] = df["regime_name"].fillna(df["regime"].map(REGIME_NAMEN))
    df["regime_name"] = df["regime_name"].fillna("Unbekannt")

    return df.sort_values("time").reset_index(drop=True)


def apply_window(df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """
    Filtert Daten auf das gewuenschte Zeitfenster.

    Args:
        df: Vollstaendige Logdaten.
        config: Auswertungskonfiguration.

    Returns:
        DataFrame im gewuenschten Fenster.
    """
    if config.start_utc is not None and config.end_utc is not None:
        return df[(df["time"] >= config.start_utc) & (df["time"] <= config.end_utc)].copy()
    assert config.hours is not None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.hours)
    return df[df["time"] >= cutoff].copy()


def threshold_mask(df: pd.DataFrame, threshold: float, operator: str) -> pd.Series:
    """
    Erzeugt die boolesche Signalmaske fuer ein Threshold-Szenario.

    Args:
        df: Gefilterte Logdaten.
        threshold: Zu pruefende Schwelle.
        operator: ge(>=) oder gt(>).

    Returns:
        Boolesche Maske fuer aktive Signale.
    """
    if operator == "ge":
        return (df["signal"] != 0) & (df["prob"] >= threshold)
    return (df["signal"] != 0) & (df["prob"] > threshold)


def safe_mean(values: pd.Series) -> float:
    """
    Liefert robust den Mittelwert oder 0.0 fuer leere Werte.

    Args:
        values: Werte-Container.

    Returns:
        Durchschnitt als float.
    """
    clean = values.dropna()
    return float(clean.mean()) if not clean.empty else 0.0


def compute_frequency_minutes(active_signals: pd.DataFrame) -> float:
    """
    Berechnet die mittlere Distanz zwischen aktiven Signalen.

    Args:
        active_signals: Signal-DataFrame fuer eine Schwelle.

    Returns:
        Durchschnittliche Distanz in Minuten.
    """
    if len(active_signals) < 2:
        return 0.0
    diffs = active_signals["time"].diff().dropna().dt.total_seconds() / 60.0
    return float(diffs.mean()) if not diffs.empty else 0.0


def compute_stale_gaps(
    active_signals: pd.DataFrame,
    timeframe: str,
    stale_factor: float,
) -> int:
    """
    Zaehlt Signalabstaende ueber der konfigurierten Stale-Grenze.

    Args:
        active_signals: Signal-DataFrame fuer eine Schwelle.
        timeframe: Timeframe fuer die Grenzwertableitung.
        stale_factor: Multiplikator auf Kerzenlaenge.

    Returns:
        Anzahl der erkannten Stale-Gaps.
    """
    if len(active_signals) < 2:
        return 0
    limit = timeframe_minutes(timeframe) * stale_factor
    diffs = active_signals["time"].diff().dropna().dt.total_seconds() / 60.0
    return int((diffs > limit).sum())


def optional_trade_metrics(active_signals: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Berechnet optionale Trade-KPIs falls PnL-Spalten im Log existieren.

    Args:
        active_signals: Signal-DataFrame fuer eine Schwelle.

    Returns:
        Dict mit optionalen KPIs (None falls nicht verfuegbar).
    """
    if "pnl_pct" not in active_signals.columns:
        return {
            "profit_factor": None,
            "win_rate_pct": None,
            "avg_r_pct": None,
            "max_drawdown_pct": None,
            "return_pct": None,
        }

    pnl = pd.to_numeric(active_signals["pnl_pct"], errors="coerce").dropna()
    if pnl.empty:
        return {
            "profit_factor": None,
            "win_rate_pct": None,
            "avg_r_pct": None,
            "max_drawdown_pct": None,
            "return_pct": None,
        }

    wins = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    pf = (wins / losses) if losses > 0 else float("inf")
    win_rate_pct = float((pnl > 0).mean() * 100.0)
    avg_r_pct = float(pnl.mean() * 100.0)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    max_dd_pct = float(drawdown.min() * 100.0)
    return_pct = float(pnl.sum() * 100.0)

    return {
        "profit_factor": pf,
        "win_rate_pct": win_rate_pct,
        "avg_r_pct": avg_r_pct,
        "max_drawdown_pct": max_dd_pct,
        "return_pct": return_pct,
    }


def score_summary_row(row: pd.Series) -> float:
    """
    Bewertet eine Summary-Zeile fuer die Ranking-Ausgabe.

    Args:
        row: Zeile aus dem Summary-DataFrame.

    Returns:
        Deterministischer Score (hoeher ist besser).
    """
    # Basisscore: Wahrscheinlichkeit + Aktivitaet.
    score = float(row["avg_prob_signals_pct"]) + float(row["signal_rate_pct"]) * 0.20
    # Zu viele Stale-Gaps werden abgestraft.
    score -= float(row["stale_gap_count"]) * 2.0

    # Optional: Trade-Outcomes einbeziehen wenn vorhanden.
    profit_factor = row.get("profit_factor")
    win_rate_pct = row.get("win_rate_pct")
    return_pct = row.get("return_pct")
    max_drawdown_pct = row.get("max_drawdown_pct")

    if pd.notna(profit_factor):
        # Inf wird hart gedeckelt, um Ranglisten stabil zu halten.
        score += min(float(profit_factor), 5.0) * 5.0
    if pd.notna(win_rate_pct):
        score += float(win_rate_pct) * 0.15
    if pd.notna(return_pct):
        score += float(return_pct) * 0.30
    if pd.notna(max_drawdown_pct):
        # Drawdown ist negativ; staerkere Drawdowns reduzieren den Score.
        score += float(max_drawdown_pct) * 0.20

    return round(score, 4)


def summarise_symbol_threshold(
    symbol: str,
    df_window: pd.DataFrame,
    threshold: float,
    config: EvalConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Erstellt Summary- und Detail-Metriken fuer Symbol + Schwelle.

    Args:
        symbol: Handelssymbol.
        df_window: Zeitgefilterte Logdaten.
        threshold: Schwellenwert.
        config: Globale Konfiguration.

    Returns:
        Tuple aus (summary_row, detail_rows).
    """
    signals_mask = threshold_mask(df_window, threshold, config.threshold_operator)
    active_signals = df_window[signals_mask].copy()

    events_total = int(len(df_window))
    signals_total = int(len(active_signals))
    long_count = int((active_signals["signal"] == 2).sum())
    short_count = int((active_signals["signal"] == -1).sum())
    signal_rate_pct = (signals_total / events_total * 100.0) if events_total > 0 else 0.0
    avg_prob_signals_pct = safe_mean(active_signals["prob"]) * 100.0
    avg_signal_distance_minutes = compute_frequency_minutes(active_signals)
    stale_gap_count = compute_stale_gaps(
        active_signals=active_signals,
        timeframe=config.timeframe,
        stale_factor=config.stale_factor,
    )

    # Optional vorhandene Trade-Metriken.
    trade_metrics = optional_trade_metrics(active_signals)

    # Dominantes Regime fuer schnellen Vergleich.
    if signals_total > 0:
        dominant_regime = str(active_signals["regime_name"].value_counts().idxmax())
    else:
        dominant_regime = "Kein_Signal"

    summary_row: Dict[str, object] = {
        "symbol": symbol,
        "threshold": threshold,
        "operator": config.threshold_operator,
        "events_total": events_total,
        "signals_total": signals_total,
        "signals_long": long_count,
        "signals_short": short_count,
        "signal_rate_pct": round(signal_rate_pct, 2),
        "avg_prob_signals_pct": round(avg_prob_signals_pct, 2),
        "avg_signal_distance_minutes": round(avg_signal_distance_minutes, 2),
        "stale_gap_count": stale_gap_count,
        "dominant_regime": dominant_regime,
        "profit_factor": trade_metrics["profit_factor"],
        "win_rate_pct": trade_metrics["win_rate_pct"],
        "avg_r_pct": trade_metrics["avg_r_pct"],
        "max_drawdown_pct": trade_metrics["max_drawdown_pct"],
        "return_pct": trade_metrics["return_pct"],
    }

    # Detail: immer alle bekannten Regimes ausgeben (auch wenn count=0).
    detail_rows: List[Dict[str, object]] = []
    for regime_id, regime_name in REGIME_NAMEN.items():
        regime_df = active_signals[active_signals["regime"] == regime_id]
        regime_trade_metrics = optional_trade_metrics(regime_df)
        detail_rows.append(
            {
                "symbol": symbol,
                "threshold": threshold,
                "operator": config.threshold_operator,
                "regime_id": regime_id,
                "regime_name": regime_name,
                "signals_count": int(len(regime_df)),
                "long_count": int((regime_df["signal"] == 2).sum()),
                "short_count": int((regime_df["signal"] == -1).sum()),
                "avg_prob_pct": round(safe_mean(regime_df["prob"]) * 100.0, 2),
                "profit_factor": regime_trade_metrics["profit_factor"],
                "win_rate_pct": regime_trade_metrics["win_rate_pct"],
                "avg_r_pct": regime_trade_metrics["avg_r_pct"],
                "max_drawdown_pct": regime_trade_metrics["max_drawdown_pct"],
                "return_pct": regime_trade_metrics["return_pct"],
            }
        )

    return summary_row, detail_rows


def evaluate(config: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fuehrt die gesamte Schwellen-Auswertung ueber alle Symbole aus.

    Args:
        config: Auswertungs-Konfiguration.

    Returns:
        Tuple aus (summary_df, detail_df, warnings).
    """
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    for symbol in config.symbols:
        try:
            df_symbol = load_symbol_log(config.log_dir, symbol)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"{symbol}: {exc}")
            continue

        df_window = apply_window(df_symbol, config)
        for threshold in config.thresholds:
            summary_row, details = summarise_symbol_threshold(
                symbol=symbol,
                df_window=df_window,
                threshold=threshold,
                config=config,
            )
            summary_rows.append(summary_row)
            detail_rows.extend(details)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    # Ranking nur wenn Daten vorhanden sind.
    if not summary_df.empty:
        summary_df["score"] = summary_df.apply(score_summary_row, axis=1)
        summary_df["rank_global"] = (
            summary_df["score"].rank(method="dense", ascending=False).astype(int)
        )
        summary_df["rank_symbol"] = (
            summary_df.groupby("symbol")["score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        summary_df = summary_df.sort_values(
            by=["rank_global", "symbol", "threshold"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            by=["symbol", "threshold", "regime_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    return summary_df, detail_df, warnings


def recommendation_text(summary_df: pd.DataFrame) -> str:
    """
    Erzeugt eine kompakte Empfehlung auf Basis des Top-Rankings.

    Args:
        summary_df: Aggregierte Summary-Daten.

    Returns:
        Textuelle Empfehlung fuer den Betrieb.
    """
    if summary_df.empty:
        return "Keine Empfehlung moeglich (keine Daten)."

    top = summary_df.iloc[0]
    mode = "konservativ" if float(top["threshold"]) >= 0.55 else "aktiv"
    return (
        f"Empfehlung: {mode} | bestes Setup: {top['symbol']} "
        f"bei Schwelle {float(top['threshold']):.2f} ({top['operator']})."
    )


def print_console_summary(
    config: EvalConfig,
    summary_df: pd.DataFrame,
    warnings: Sequence[str],
) -> None:
    """
    Druckt eine kompakte Konsolenausgabe fuer schnelle Bewertung.

    Args:
        config: Laufkonfiguration.
        summary_df: Ergebnisdaten.
        warnings: Warnungen aus dem Lauf.
    """
    print("=" * 100)
    print("THRESHOLD KPI EVALUATION")
    print("=" * 100)
    # Schwellen-Semantik explizit dokumentieren.
    op_text = ">=" if config.threshold_operator == "ge" else ">"
    print(
        "Threshold-Semantik: "
        f"Signal gilt als aktiv bei prob {op_text} threshold "
        f"(operator={config.threshold_operator})."
    )
    print(
        "Hinweis Vergleichbarkeit: Live-Logik nutzt typischerweise >=, "
        "Backtests oft >. Dieser Lauf verwendet exakt die oben gezeigte Semantik."
    )
    if config.hours is not None:
        print(f"Fenster: letzte {config.hours}h")
    else:
        assert config.start_utc is not None and config.end_utc is not None
        print(f"Fenster: {config.start_utc.isoformat()} bis {config.end_utc.isoformat()}")
    print(
        f"Timeframe={config.timeframe} | Stale-Grenze="
        f"{timeframe_minutes(config.timeframe) * config.stale_factor:.1f} Minuten"
    )
    print("-" * 100)

    if warnings:
        print("WARNUNGEN:")
        for warning in warnings:
            print(f"- {warning}")
        print("-" * 100)

    if summary_df.empty:
        print("Keine auswertbaren Daten gefunden.")
        return

    columns = [
        "rank_global",
        "rank_symbol",
        "symbol",
        "threshold",
        "signals_total",
        "signal_rate_pct",
        "avg_prob_signals_pct",
        "stale_gap_count",
        "score",
    ]
    print(summary_df[columns].to_string(index=False))
    print("-" * 100)
    print(recommendation_text(summary_df))


def save_reports(
    output_dir: Path,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> Tuple[Path, Path]:
    """
    Schreibt Summary- und Detail-Artefakte als CSV.

    Args:
        output_dir: Zielordner.
        summary_df: Aggregierte Summary-Daten.
        detail_df: Regime-Detaildaten.

    Returns:
        Pfade zu (summary_csv, detail_csv).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    detail_path = output_dir / "detail.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    return summary_path, detail_path


def run() -> int:
    """
    Programm-Entry mit Rückgabecode.

    Returns:
        0 bei Erfolg, 1 bei Fehlern.
    """
    try:
        args = parse_args()
        config = build_config(args)
    except ValueError as exc:
        print(f"[FEHLER] Ungueltige Parameter: {exc}")
        return 1

    if not config.log_dir.exists():
        print(f"[FEHLER] Log-Ordner nicht gefunden: {config.log_dir.resolve()}")
        return 1

    summary_df, detail_df, warnings = evaluate(config)
    print_console_summary(config, summary_df, warnings)

    summary_path, detail_path = save_reports(config.output_dir, summary_df, detail_df)
    print(f"CSV geschrieben: {summary_path}")
    print(f"CSV geschrieben: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
