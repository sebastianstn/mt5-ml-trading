"""
weekly_kpi_report.py – Automatischer Wochenreport für 2 Kernsymbole

Zweck:
        Erstellt einen kompakten KPI-Report für USDCAD und USDJPY mit
        Go/No-Go-Ampel auf Basis von:
            1) Live-Aktivitätsdaten (logs/SYMBOL_signals.csv, falls vorhanden)
      2) Profitabilitätsdaten aus Backtest-Trades (backtest/SYMBOL_trades.csv)

Wichtig:
    Die Live-Logdateien enthalten standardmäßig Signal-/Operativdaten,
    aber keinen realisierten P&L. Daher wird die Profitabilität aus den
    Backtest-Trades berechnet, bis ein echter Live-PnL-Export vorliegt.

    Wichtig für Phase 7:
    Weekly KPI-Gates werden NUR mit frischen Live-Daten bewertet.
    Fehlen frische Live-Events (stale), wird der Symbolstatus hart auf NO-GO gesetzt.

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1Tb-Data/XGBoost-LightGBM
    source .venv/bin/activate
    python reports/weekly_kpi_report.py
    python reports/weekly_kpi_report.py --tage 7

Eingabe:
    logs/USDCAD_signals.csv (optional)
    logs/USDJPY_signals.csv (optional)
    backtest/USDCAD_trades.csv
    backtest/USDJPY_trades.csv

Ausgabe:
    reports/weekly_kpi_report.md
"""

# Standard-Bibliotheken
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Datenverarbeitung
import pandas as pd

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
BACKTEST_DIR = BASE_DIR / "backtest"
REPORTS_DIR = BASE_DIR / "reports"

# Kernsymbole laut aktueller Betriebsstrategie
KERN_SYMBOLE = ["USDCAD", "USDJPY"]

# Aktuelle Live-Log-Suffixe aus live_trader.py
LIVE_SIGNAL_SUFFIX = "_signals.csv"
LIVE_CLOSE_SUFFIX = "_closes.csv"

# Persistente Historie für Wochen-Gates
KPI_HISTORY_PATH = REPORTS_DIR / "weekly_kpi_history.csv"

# KPI-Schwellenwerte (Roadmap-/Betriebsziele)
ZIEL_PROFIT_FACTOR = 1.20
ZIEL_SHARPE = 0.80
ZIEL_MAX_DD = -10.0  # in Prozent (negativ)
ZIEL_WIN_RATE = 45.0  # in Prozent
ZIEL_SIGNALE_WOCHE = 5  # Mindestaktivität
ZIEL_CLOSES_WOCHE = 5  # Mindestanzahl Closings für belastbare Live-PnL-KPIs
START_EQUITY_LIVE = 10000.0  # Referenzwert für Live-/Paper-DD aus *_closes.csv

# Stale-Logik für "frische Live-Daten erzwingen"
STALE_FACTOR = 1.5

# 3-Monats-Regel (ca. 12 Wochen): Eskalation nur bei stabilen GO-Werten
PAPER_GATE_WOCHEN = 12


@dataclass
class SymbolKPI:
    """Kapselt alle KPI-Werte für ein Symbol.

    Args:
        symbol: Währungssymbol.
        live_signale: Anzahl Live-Signale im Zeitraum.
        live_long_pct: Long-Anteil in % im Live-Log.
        live_short_pct: Short-Anteil in % im Live-Log.
        live_avg_prob: Durchschnittliche Signal-Wahrscheinlichkeit.
        profit_factor: Gewinnfaktor aus Backtest-Trades.
        sharpe_ratio: Sharpe aus Backtest-Trades.
        max_drawdown_pct: Maximaler Drawdown in % aus Backtest-Trades.
        win_rate_pct: Win-Rate in % aus Backtest-Trades.
        return_pct: Gesamtrendite in % aus Backtest-Trades.
        status: Ampelstatus (GO/NO-GO/UNKLAR).
        hinweis: Kurzbegründung zum Status.
    """

    symbol: str
    live_signale: int
    live_events: int
    live_long_pct: float
    live_short_pct: float
    live_avg_prob: float
    live_last_event_utc: str
    live_minutes_since_last: Optional[float]
    live_fresh: bool
    live_closes: int
    live_profit_factor: Optional[float]
    live_win_rate_pct: Optional[float]
    live_max_drawdown_pct: Optional[float]
    live_net_pnl: Optional[float]
    live_avg_dauer_min: Optional[float]
    metric_source: str
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    return_pct: float
    status: str
    hinweis: str


def lade_live_log(symbol: str, tage: int) -> Optional[pd.DataFrame]:
    """Lädt das Live-Log eines Symbols für die letzten N Tage.

    Args:
        symbol: Währungssymbol (z.B. "USDCAD").
        tage: Rückblick in Tagen.

    Returns:
        Gefilterter DataFrame oder None, wenn keine Datei vorhanden ist.
    """
    live_path = LOG_DIR / f"{symbol}{LIVE_SIGNAL_SUFFIX}"
    if not live_path.exists():
        logger.warning("[%s] Kein Live-Log gefunden: %s", symbol, live_path)
        return None

    df = pd.read_csv(live_path)
    if df.empty or "time" not in df.columns:
        logger.warning("[%s] Live-Log leer oder ohne 'time'-Spalte.", symbol)
        return None

    # Zeitstempel robust parsen (UTC-naiv für lokale Vergleiche)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=tage)
    return df[df["time"] >= cutoff].copy()


def lade_close_log(symbol: str, tage: int) -> Optional[pd.DataFrame]:
    """Lädt das Close-Log eines Symbols für die letzten N Tage."""
    close_path = LOG_DIR / f"{symbol}{LIVE_CLOSE_SUFFIX}"
    if not close_path.exists():
        logger.info("[%s] Kein Close-Log gefunden: %s", symbol, close_path)
        return None

    df = pd.read_csv(close_path)
    if df.empty or "time" not in df.columns:
        logger.warning("[%s] Close-Log leer oder ohne 'time'-Spalte.", symbol)
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=tage)
    return df[df["time"] >= cutoff].copy()


def timeframe_minutes(timeframe: str) -> int:
    """Übersetzt den Timeframe-String in Minuten.

    Args:
        timeframe: Timeframe aus CLI.

    Returns:
        Minuten pro Kerze.
    """
    mapping = {
        "H1": 60,
        "M60": 60,
        "M30": 30,
        "M15": 15,
        "M5_TWO_STAGE": 5,
        "H4": 240,
    }
    return mapping.get(timeframe, 60)


def live_kpis_berechnen(symbol: str, tage: int, timeframe: str = "H1") -> dict:
    """Berechnet operative Live-KPIs für ein Symbol.

    Args:
        symbol: Währungssymbol.
        tage: Zeitraum in Tagen.

    Returns:
        Dict mit Live-Aktivitätskennzahlen.
    """
    df = lade_live_log(symbol, tage)
    if df is None or df.empty:
        return {
            "live_signale": 0,
            "live_events": 0,
            "live_long_pct": 0.0,
            "live_short_pct": 0.0,
            "live_avg_prob": 0.0,
            "live_last_event_utc": "",
            "live_minutes_since_last": None,
            "live_fresh": False,
        }

    # Richtungsspalte: "Long" / "Short" / ggf. andere Werte
    n = len(df)
    n_long = int((df["richtung"] == "Long").sum()) if "richtung" in df.columns else 0
    n_short = int((df["richtung"] == "Short").sum()) if "richtung" in df.columns else 0
    avg_prob = float(df["prob"].mean()) if "prob" in df.columns else 0.0
    n_signale = int((df["signal"] != 0).sum()) if "signal" in df.columns else 0

    last_event_utc = ""
    minutes_since_last: Optional[float] = None
    live_fresh = False
    if n > 0:
        last_ts = pd.to_datetime(df["time"], errors="coerce").max()
        if pd.notna(last_ts):
            now_ts = (
                pd.Timestamp.now(tz=last_ts.tz)
                if getattr(last_ts, "tz", None)
                else pd.Timestamp.now()
            )
            minutes_since_last = float((now_ts - last_ts).total_seconds() / 60.0)
            stale_limit = timeframe_minutes(timeframe) * STALE_FACTOR
            live_fresh = minutes_since_last <= stale_limit
            last_event_utc = str(last_ts)

    return {
        "live_signale": n_signale,
        "live_events": n,
        "live_long_pct": (n_long / n * 100) if n > 0 else 0.0,
        "live_short_pct": (n_short / n * 100) if n > 0 else 0.0,
        "live_avg_prob": avg_prob,
        "live_last_event_utc": last_event_utc,
        "live_minutes_since_last": minutes_since_last,
        "live_fresh": live_fresh,
    }


def backtest_kpis_berechnen(symbol: str, timeframe: str = "H1") -> dict:
    """Berechnet Profitabilitäts-KPIs aus gespeicherten Backtest-Trades.

    Args:
        symbol: Währungssymbol.

    Returns:
        Dict mit Profitabilitätskennzahlen.

    Raises:
        FileNotFoundError: Wenn die Trade-Datei fehlt.
        ValueError: Wenn erforderliche Spalten fehlen.
    """
    if timeframe == "H1":
        trade_path = BACKTEST_DIR / f"{symbol}_trades.csv"
    elif timeframe == "M5_TWO_STAGE":
        trade_path = BACKTEST_DIR / f"{symbol}_M5_two_stage_trades.csv"
    else:
        trade_path = BACKTEST_DIR / f"{symbol}_{timeframe}_trades.csv"
    if not trade_path.exists():
        raise FileNotFoundError(f"Trades-Datei fehlt: {trade_path}")

    df = pd.read_csv(trade_path, index_col=0, parse_dates=True)
    if "pnl_pct" not in df.columns:
        raise ValueError(f"'pnl_pct' fehlt in {trade_path}")

    pnl = df["pnl_pct"].astype(float)
    if pnl.empty:
        return {
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "return_pct": 0.0,
        }

    # Gewinnfaktor
    gewinne = float(pnl[pnl > 0].sum())
    verluste = abs(float(pnl[pnl < 0].sum()))
    profit_factor = (gewinne / verluste) if verluste > 0 else float("inf")

    # Sharpe (trades-basiert, annualisierte Näherung)
    sharpe = 0.0
    if len(pnl) > 1 and float(pnl.std()) > 0:
        sharpe = float((pnl.mean() / pnl.std()) * (252**0.5))

    # Max Drawdown in %
    equity = pnl.cumsum()
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd_pct = float(drawdown.min() * 100)

    # Win-Rate und Rendite
    win_rate_pct = float((pnl > 0).mean() * 100)
    return_pct = float(pnl.sum() * 100)

    return {
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": win_rate_pct,
        "return_pct": return_pct,
    }


def profit_factor_from_money(pnl_series: pd.Series) -> Optional[float]:
    """Berechnet Gewinnfaktor aus einer Geld-PnL-Serie."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty:
        return None
    wins = float(clean[clean > 0].sum())
    losses_abs = abs(float(clean[clean < 0].sum()))
    if losses_abs <= 0.0:
        return float("inf") if wins > 0 else None
    return wins / losses_abs


def max_drawdown_pct_from_money(
    pnl_series: pd.Series,
    start_equity: float = START_EQUITY_LIVE,
) -> Optional[float]:
    """Berechnet maximalen Drawdown in Prozent aus kumulierter Geld-PnL."""
    clean = pd.to_numeric(pnl_series, errors="coerce").dropna()
    if clean.empty or start_equity <= 0.0:
        return None
    equity = start_equity + clean.cumsum()
    drawdown_money = equity - equity.cummax()
    dd_pct = (drawdown_money / start_equity) * 100.0
    return float(dd_pct.min())


def live_close_kpis_berechnen(symbol: str, tage: int) -> dict:
    """Berechnet realisierte Live-/Paper-KPIs aus *_closes.csv."""
    df = lade_close_log(symbol, tage)
    if df is None or df.empty:
        return {
            "live_closes": 0,
            "live_profit_factor": None,
            "live_win_rate_pct": None,
            "live_max_drawdown_pct": None,
            "live_net_pnl": None,
            "live_avg_dauer_min": None,
        }

    pnl_money = (
        pd.to_numeric(df.get("pnl_money", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(float)
    )
    dauer_series = (
        pd.to_numeric(df.get("dauer_min", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(float)
    )

    live_pf = profit_factor_from_money(pnl_money)
    live_win_rate = (
        float((pnl_money > 0).mean() * 100.0) if not pnl_money.empty else None
    )
    live_dd = max_drawdown_pct_from_money(pnl_money) if not pnl_money.empty else None
    live_net_pnl = float(pnl_money.sum()) if not pnl_money.empty else None
    live_avg_dauer = float(dauer_series.mean()) if not dauer_series.empty else None

    return {
        "live_closes": int(len(df)),
        "live_profit_factor": live_pf,
        "live_win_rate_pct": live_win_rate,
        "live_max_drawdown_pct": live_dd,
        "live_net_pnl": live_net_pnl,
        "live_avg_dauer_min": live_avg_dauer,
    }


def status_bewerten(kpi: dict) -> tuple[str, str]:
    """Bewertet KPI-Werte als GO/NO-GO/UNKLAR.

    Args:
        kpi: Kombinierte KPI-Werte eines Symbols.

    Returns:
        Tuple aus (Status, Hinweis).
    """
    if not bool(kpi.get("live_fresh", False)):
        return "NO-GO", "Keine frischen Live-Daten (stale/fehlend)"

    # Mindestaktivität: sonst keine belastbare Entscheidung
    if int(kpi["live_signale"]) < ZIEL_SIGNALE_WOCHE:
        return "UNKLAR", "Zu wenige Live-Signale im Zeitraum"

    if int(kpi.get("live_closes", 0)) >= ZIEL_CLOSES_WOCHE:
        checks = {
            "PF": (kpi.get("live_profit_factor") is not None)
            and float(kpi["live_profit_factor"]) >= ZIEL_PROFIT_FACTOR,
            "DD": (kpi.get("live_max_drawdown_pct") is not None)
            and float(kpi["live_max_drawdown_pct"]) >= ZIEL_MAX_DD,
            "WinRate": (kpi.get("live_win_rate_pct") is not None)
            and float(kpi["live_win_rate_pct"]) >= ZIEL_WIN_RATE,
        }

        if all(checks.values()):
            return "GO", "Live-Close-KPIs erfüllen alle Gates"

        failed = [name for name, ok in checks.items() if not ok]
        return "NO-GO", f"Live-Close-KPI unter Ziel: {', '.join(failed)}"

    if int(kpi.get("live_closes", 0)) > 0:
        return (
            "UNKLAR",
            "Live-Closes vorhanden, aber noch zu wenige für belastbare PnL-Gates",
        )

    checks = {
        "PF": float(kpi["profit_factor"]) >= ZIEL_PROFIT_FACTOR,
        "Sharpe": float(kpi["sharpe_ratio"]) >= ZIEL_SHARPE,
        "DD": float(kpi["max_drawdown_pct"]) >= ZIEL_MAX_DD,
        "WinRate": float(kpi["win_rate_pct"]) >= ZIEL_WIN_RATE,
    }

    if all(checks.values()):
        return "GO", "Backtest-KPIs erfüllt, Live-Close-Daten noch nicht belastbar"

    failed = [name for name, ok in checks.items() if not ok]
    return "NO-GO", f"Backtest-KPI unter Ziel: {', '.join(failed)}"


def symbol_report_erstellen(symbol: str, tage: int, timeframe: str = "H1") -> SymbolKPI:
    """Erstellt alle KPIs und die Statusbewertung für ein Symbol.

    Args:
        symbol: Währungssymbol.
        tage: Zeitraum in Tagen für Live-Aktivität.

    Returns:
        Vollständiges SymbolKPI-Objekt.
    """
    live = live_kpis_berechnen(symbol, tage, timeframe=timeframe)
    live_closes = live_close_kpis_berechnen(symbol, tage)
    try:
        bt = backtest_kpis_berechnen(symbol, timeframe=timeframe)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("[%s] Backtest-KPIs nicht verfügbar: %s", symbol, exc)
        bt = {
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "return_pct": 0.0,
        }

    kombiniert = {**live, **live_closes, **bt}
    status, hinweis = status_bewerten(kombiniert)

    metric_source = (
        "LIVE_CLOSES"
        if int(kombiniert.get("live_closes", 0)) >= ZIEL_CLOSES_WOCHE
        else "BACKTEST_FALLBACK"
    )

    return SymbolKPI(
        symbol=symbol,
        live_signale=int(kombiniert["live_signale"]),
        live_events=int(kombiniert["live_events"]),
        live_long_pct=float(kombiniert["live_long_pct"]),
        live_short_pct=float(kombiniert["live_short_pct"]),
        live_avg_prob=float(kombiniert["live_avg_prob"]),
        live_last_event_utc=str(kombiniert.get("live_last_event_utc", "")),
        live_minutes_since_last=(
            float(kombiniert["live_minutes_since_last"])
            if kombiniert.get("live_minutes_since_last") is not None
            else None
        ),
        live_fresh=bool(kombiniert.get("live_fresh", False)),
        live_closes=int(kombiniert.get("live_closes", 0)),
        live_profit_factor=(
            float(kombiniert["live_profit_factor"])
            if kombiniert.get("live_profit_factor") is not None
            else None
        ),
        live_win_rate_pct=(
            float(kombiniert["live_win_rate_pct"])
            if kombiniert.get("live_win_rate_pct") is not None
            else None
        ),
        live_max_drawdown_pct=(
            float(kombiniert["live_max_drawdown_pct"])
            if kombiniert.get("live_max_drawdown_pct") is not None
            else None
        ),
        live_net_pnl=(
            float(kombiniert["live_net_pnl"])
            if kombiniert.get("live_net_pnl") is not None
            else None
        ),
        live_avg_dauer_min=(
            float(kombiniert["live_avg_dauer_min"])
            if kombiniert.get("live_avg_dauer_min") is not None
            else None
        ),
        metric_source=metric_source,
        profit_factor=float(kombiniert["profit_factor"]),
        sharpe_ratio=float(kombiniert["sharpe_ratio"]),
        max_drawdown_pct=float(kombiniert["max_drawdown_pct"]),
        win_rate_pct=float(kombiniert["win_rate_pct"]),
        return_pct=float(kombiniert["return_pct"]),
        status=status,
        hinweis=hinweis,
    )


def markdown_bericht_schreiben(
    kpis: list[SymbolKPI],
    tage: int,
    timeframe: str = "H1",
) -> Path:
    """Schreibt den Wochenreport als Markdown-Datei.

    Args:
        kpis: KPI-Objekte der Symbole.
        tage: betrachteter Zeitraum in Tagen.

    Returns:
        Pfad zur erzeugten Report-Datei.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if timeframe == "H1":
        out_path = REPORTS_DIR / "weekly_kpi_report.md"
    else:
        out_path = REPORTS_DIR / f"weekly_kpi_report_{timeframe}.md"
    erstellt = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Gesamteinschätzung aus Einzelstatus ableiten
    status_liste = [k.status for k in kpis]
    if all(s == "GO" for s in status_liste):
        gesamt = "GO"
    elif any(s == "NO-GO" for s in status_liste):
        gesamt = "NO-GO"
    else:
        gesamt = "UNKLAR"

    # Historie fortschreiben und 3-Monats-Gate auswerten
    gate_info = historie_aktualisieren_und_gate_pruefen(kpis, gesamt)

    lines = [
        "# Wöchentlicher KPI-Report (USDCAD / USDJPY)",
        "",
        f"**Erstellt:** {erstellt}",
        f"**Zeitraum (Live-Aktivität):** letzte {tage} Tage",
        f"**Gesamtstatus:** **{gesamt}**",
        f"**Timeframe:** `{timeframe}`",
        f"**Paper-Gate (12 Wochen):** **{gate_info['paper_gate_status']}**",
        f"**Konsekutive GO-Wochen:** {gate_info['consecutive_go_weeks']} / {PAPER_GATE_WOCHEN}",
        "",
        "## KPI-Zielwerte",
        "",
        f"- Profit Factor > {ZIEL_PROFIT_FACTOR}",
        f"- Sharpe > {ZIEL_SHARPE}",
        f"- Max Drawdown > {ZIEL_MAX_DD:.1f}%",
        f"- Win-Rate > {ZIEL_WIN_RATE:.1f}%",
        f"- Mindest-Live-Signale/Woche: {ZIEL_SIGNALE_WOCHE}",
        f"- Mindest-Live-Closes/Woche: {ZIEL_CLOSES_WOCHE}",
        "",
        "## Ergebnis je Symbol",
        "",
        "| Symbol | Status | Quelle | Live Fresh | Letztes Event (UTC) | Min seit Event | "
        "Events | Signale | Closes | Ø Prob | Live PnL | Live PF | Live DD% | Live WR% | Ø Dauer Min | PF (BT) | Sharpe (BT) | Hinweis |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for k in kpis:
        lines.append(
            "| {symbol} | {status} | {source} | {fresh} | {last} | {mins} | {events} | {live_signale} | {closes} | "
            "{prob:.3f} | {live_pnl} | {live_pf} | {live_dd} | {live_wr} | {live_dauer} | {pf:.3f} | {sh:.3f} | {hint} |".format(
                symbol=k.symbol,
                status=k.status,
                source=k.metric_source,
                fresh="OK" if k.live_fresh else "STALE",
                last=k.live_last_event_utc if k.live_last_event_utc else "-",
                mins=(
                    f"{k.live_minutes_since_last:.1f}"
                    if k.live_minutes_since_last is not None
                    else "-"
                ),
                events=k.live_events,
                live_signale=k.live_signale,
                closes=k.live_closes,
                prob=k.live_avg_prob,
                live_pnl=(
                    f"{k.live_net_pnl:+.2f}" if k.live_net_pnl is not None else "-"
                ),
                live_pf=(
                    f"{k.live_profit_factor:.3f}"
                    if k.live_profit_factor is not None
                    else "-"
                ),
                live_dd=(
                    f"{k.live_max_drawdown_pct:.2f}"
                    if k.live_max_drawdown_pct is not None
                    else "-"
                ),
                live_wr=(
                    f"{k.live_win_rate_pct:.1f}"
                    if k.live_win_rate_pct is not None
                    else "-"
                ),
                live_dauer=(
                    f"{k.live_avg_dauer_min:.1f}"
                    if k.live_avg_dauer_min is not None
                    else "-"
                ),
                pf=k.profit_factor,
                sh=k.sharpe_ratio,
                hint=k.hinweis,
            )
        )

    lines += [
        "",
        "## 3-Monats Paper-Gate (stabile Werte)",
        "",
        f"- Letzte {PAPER_GATE_WOCHEN} Wochen insgesamt erfasst: {gate_info['available_weeks']}",
        f"- Konsekutive GO-Wochen aktuell: {gate_info['consecutive_go_weeks']}",
        f"- Entscheidungsstatus: **{gate_info['paper_gate_status']}**",
        f"- Begründung: {gate_info['paper_gate_reason']}",
        "",
        "## Interpretation",
        "",
        "- **Live-Freshness** ist ein hartes Gate: ohne frische Events wird der Status auf NO-GO gesetzt.",
        "- **Live-Signale** messen operative Aktivität und Freshness.",
        "- **Live-Closes** liefern realisierte Paper-/Live-PnL-KPIs und werden bevorzugt, sobald genug Daten vorliegen.",
        (
            "- **Backtest-KPIs** bleiben Fallback, solange noch nicht genug Live-Closes vorhanden sind "
            "(`backtest/SYMBOL_trades.csv`)."
            if timeframe == "H1"
            else (
                "- **Backtest-KPIs** bleiben Fallback, solange noch nicht genug Live-Closes vorhanden sind "
                "(`backtest/SYMBOL_M5_two_stage_trades.csv`)."
                if timeframe == "M5_TWO_STAGE"
                else (
                    "- **Backtest-KPIs** bleiben Fallback, solange noch nicht genug Live-Closes vorhanden sind "
                    f"(`backtest/SYMBOL_{timeframe}_trades.csv`)."
                )
            )
        ),
        "",
        "## Nächste Schritte",
        "",
        "1. Bei **NO-GO**: Schwellen, Regime oder Overtrading prüfen und nur im Paper-Modus weiterlaufen lassen.",
        "2. Bei **UNKLAR**: mehr Live-Closes sammeln (Trader weiterlaufen lassen) oder Zeitraum verlängern.",
        f"3. Eskalation Richtung Live nur bei **{PAPER_GATE_WOCHEN} konsekutiven GO-Wochen**.",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wochenreport geschrieben: %s", out_path)
    return out_path


def historie_aktualisieren_und_gate_pruefen(
    kpis: list[SymbolKPI],
    gesamtstatus: str,
) -> dict:
    """Aktualisiert die Wochenhistorie und prüft das 3-Monats-Paper-Gate.

    Gate-Logik:
        - Eskalation nur bei stabilen Werten, d.h. `PAPER_GATE_WOCHEN`
          konsekutive Wochen mit Gesamtstatus == GO.
        - Sobald NO-GO oder UNKLAR auftritt, wird die GO-Serie unterbrochen.

    Args:
        kpis: KPI-Objekte der Kernsymbole.
        gesamtstatus: Aggregierter Status der aktuellen Woche (GO/NO-GO/UNKLAR).

    Returns:
        Dict mit Gate-Status und Stabilitätskennzahlen.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    heute = pd.Timestamp.now().normalize()

    # Historie laden oder neu anlegen
    if KPI_HISTORY_PATH.exists():
        hist = pd.read_csv(KPI_HISTORY_PATH, parse_dates=["date"])
    else:
        hist = pd.DataFrame(
            columns=[
                "date",
                "overall_status",
                "USDCAD_status",
                "USDJPY_status",
                "USDCAD_pf",
                "USDJPY_pf",
                "USDCAD_sharpe",
                "USDJPY_sharpe",
                "USDCAD_dd",
                "USDJPY_dd",
                "USDCAD_winrate",
                "USDJPY_winrate",
                "USDCAD_live_events",
                "USDJPY_live_events",
                "USDCAD_live_fresh",
                "USDJPY_live_fresh",
                "USDCAD_live_signale",
                "USDJPY_live_signale",
            ]
        )

    kpi_map = {k.symbol: k for k in kpis}
    usdcad = kpi_map.get("USDCAD")
    usdjpy = kpi_map.get("USDJPY")

    neuer_eintrag = {
        "date": heute,
        "overall_status": gesamtstatus,
        "USDCAD_status": usdcad.status if usdcad else "UNKLAR",
        "USDJPY_status": usdjpy.status if usdjpy else "UNKLAR",
        "USDCAD_pf": usdcad.profit_factor if usdcad else 0.0,
        "USDJPY_pf": usdjpy.profit_factor if usdjpy else 0.0,
        "USDCAD_sharpe": usdcad.sharpe_ratio if usdcad else 0.0,
        "USDJPY_sharpe": usdjpy.sharpe_ratio if usdjpy else 0.0,
        "USDCAD_dd": usdcad.max_drawdown_pct if usdcad else 0.0,
        "USDJPY_dd": usdjpy.max_drawdown_pct if usdjpy else 0.0,
        "USDCAD_winrate": usdcad.win_rate_pct if usdcad else 0.0,
        "USDJPY_winrate": usdjpy.win_rate_pct if usdjpy else 0.0,
        "USDCAD_live_events": usdcad.live_events if usdcad else 0,
        "USDJPY_live_events": usdjpy.live_events if usdjpy else 0,
        "USDCAD_live_fresh": int(usdcad.live_fresh) if usdcad else 0,
        "USDJPY_live_fresh": int(usdjpy.live_fresh) if usdjpy else 0,
        "USDCAD_live_signale": usdcad.live_signale if usdcad else 0,
        "USDJPY_live_signale": usdjpy.live_signale if usdjpy else 0,
    }

    # Pro Woche nur ein Eintrag (heutiges Datum ersetzen, nicht duplizieren)
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
        hist = hist[hist["date"] != heute]

    if hist.empty:
        hist = pd.DataFrame([neuer_eintrag])
    else:
        hist = pd.concat([hist, pd.DataFrame([neuer_eintrag])], ignore_index=True)
    hist = hist.sort_values("date")
    hist.to_csv(KPI_HISTORY_PATH, index=False)

    # Konsekutive GO-Wochen von hinten zählen
    consecutive_go_weeks = 0
    for status in reversed(hist["overall_status"].tolist()):
        if status == "GO":
            consecutive_go_weeks += 1
        else:
            break

    available_weeks = int(len(hist))
    if consecutive_go_weeks >= PAPER_GATE_WOCHEN:
        gate_status = "ESCALATION_CANDIDATE"
        gate_reason = (
            f"{consecutive_go_weeks} konsekutive GO-Wochen erreicht "
            "(stabile Werte über 3 Monate)."
        )
    else:
        gate_status = "PAPER_ONLY"
        rest = PAPER_GATE_WOCHEN - consecutive_go_weeks
        gate_reason = (
            f"Noch {rest} GO-Woche(n) ohne Unterbrechung nötig "
            "für kontrollierte Eskalation."
        )

    return {
        "paper_gate_status": gate_status,
        "paper_gate_reason": gate_reason,
        "consecutive_go_weeks": consecutive_go_weeks,
        "available_weeks": available_weeks,
    }


def main() -> None:
    """Startet die KPI-Berichterstellung für die Kernsymbole."""
    parser = argparse.ArgumentParser(
        description="Automatischer Wochenreport für USDCAD/USDJPY"
    )
    parser.add_argument(
        "--tage",
        type=int,
        default=7,
        help="Zeitraum für Live-Aktivitätsauswertung in Tagen (Standard: 7)",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "M60", "M30", "M15", "H4", "M5_TWO_STAGE"],
        help=(
            "Timeframe für Backtest-KPI-Dateien (Standard: H1). "
            "Für M15/M30/M60 werden backtest/SYMBOL_TIMEFRAME_trades.csv ausgewertet. "
            "Für M5_TWO_STAGE werden backtest/SYMBOL_M5_two_stage_trades.csv ausgewertet."
        ),
    )
    args = parser.parse_args()

    logger.info("Starte Wochenreport für Kernsymbole: %s", ", ".join(KERN_SYMBOLE))
    logger.info("Timeframe fuer Backtest-KPIs: %s", args.timeframe)
    kpis = [
        symbol_report_erstellen(symbol, args.tage, timeframe=args.timeframe)
        for symbol in KERN_SYMBOLE
    ]
    pfad = markdown_bericht_schreiben(kpis, args.tage, timeframe=args.timeframe)

    print("=" * 70)
    print("WÖCHENTLICHER KPI-REPORT ERSTELLT")
    print("=" * 70)
    print(f"Datei: {pfad}")
    for k in kpis:
        print(
            f"- {k.symbol}: {k.status} | Quelle={k.metric_source} | "
            f"Signale={k.live_signale} | Closes={k.live_closes} | "
            f"LivePnL={k.live_net_pnl if k.live_net_pnl is not None else '-'} | "
            f"PF(BT)={k.profit_factor:.3f} | Sharpe(BT)={k.sharpe_ratio:.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
