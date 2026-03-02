"""
daily_performance_email.py – Täglicher Performance-Report per E-Mail.

Zweck:
    Erstellt eine kompakte Tageszusammenfassung für die aktiven Paper-Symbole
    (USDCAD, USDJPY) und versendet sie per SMTP.

Läuft auf:
    Linux-Server (oder Windows, solange Pfade/Logs vorhanden sind).

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source .venv/bin/activate
    python reports/daily_performance_email.py --tage 1 --dry_run
    python reports/daily_performance_email.py --tage 1

Benötigte Umgebungsvariablen (.env):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM, SMTP_TO
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# .env laden, damit SMTP-Konfiguration zentral gepflegt wird.
load_dotenv()

# Logging einheitlich und terminalfreundlich konfigurieren.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Projektpfade zentral definieren.
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
KPI_HISTORY_PATH = REPORTS_DIR / "weekly_kpi_history.csv"

# Operative Symbole gemäß aktueller Policy.
AKTIVE_SYMBOLE = ["USDCAD", "USDJPY"]


@dataclass
class EmailConfig:
    """SMTP-Konfiguration für den E-Mail-Versand.

    Args:
        host: SMTP-Serveradresse.
        port: SMTP-Port (typisch 587).
        user: SMTP-Benutzername.
        password: SMTP-Passwort.
        sender: Absenderadresse.
        recipient: Empfängeradresse.
    """

    host: str
    port: int
    user: str
    password: str
    sender: str
    recipient: str


@dataclass
class DailySymbolSnapshot:
    """Kapselt tägliche Live- und Referenzkennzahlen je Symbol.

    Args:
        symbol: Währungssymbol.
        signals_24h: Anzahl Signale im gewählten Zeitraum.
        long_pct: Long-Anteil in Prozent.
        short_pct: Short-Anteil in Prozent.
        avg_prob: Durchschnittliche Modellwahrscheinlichkeit.
        backtest_pf: Profit-Factor aus KPI-Historie (letzter Stand).
        backtest_sharpe: Sharpe aus KPI-Historie (letzter Stand).
        backtest_dd: Max Drawdown in Prozent aus KPI-Historie.
        backtest_wr: Win-Rate in Prozent aus KPI-Historie.
        gate_status: Letzter Gate-Status (GO/NO-GO/UNKLAR).
    """

    symbol: str
    signals_24h: int
    long_pct: float
    short_pct: float
    avg_prob: float
    backtest_pf: float
    backtest_sharpe: float
    backtest_dd: float
    backtest_wr: float
    gate_status: str


def parse_args() -> argparse.Namespace:
    """Parst CLI-Argumente für den Tagesreport.

    Returns:
        Namespace mit Argumentwerten.
    """
    parser = argparse.ArgumentParser(description="Täglichen KPI-Report per SMTP senden")
    parser.add_argument(
        "--tage",
        type=int,
        default=1,
        help="Rückblick für Live-Signale in Tagen (Standard: 1).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Erzeugt Mail-Inhalt ohne Versand (nur Terminal-Ausgabe).",
    )
    return parser.parse_args()


def load_email_config() -> EmailConfig:
    """Lädt und validiert SMTP-Einstellungen aus Umgebungsvariablen.

    Returns:
        Valides EmailConfig-Objekt.

    Raises:
        ValueError: Wenn Pflichtfelder fehlen oder ungültig sind.
    """
    host = os.getenv("SMTP_HOST", "").strip()
    port_raw = os.getenv("SMTP_PORT", "587").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    recipient = os.getenv("SMTP_TO", "").strip()

    fehlend = [
        name
        for name, value in [
            ("SMTP_HOST", host),
            ("SMTP_USER", user),
            ("SMTP_PASSWORD", password),
            ("SMTP_FROM", sender),
            ("SMTP_TO", recipient),
        ]
        if not value
    ]
    if fehlend:
        raise ValueError(f"Fehlende SMTP-Variablen: {', '.join(fehlend)}")

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"SMTP_PORT ist keine Zahl: {port_raw}") from exc

    return EmailConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        sender=sender,
        recipient=recipient,
    )


def _load_live_df(symbol: str, tage: int) -> Optional[pd.DataFrame]:
    """Lädt Live-Trades eines Symbols für den gewünschten Zeitraum.

    Args:
        symbol: Währungssymbol.
        tage: Rückblick in Tagen.

    Returns:
        Gefilterter DataFrame oder None, falls Datei fehlt/leer ist.
    """
    path = LOGS_DIR / f"{symbol}_live_trades.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    cutoff = pd.Timestamp.now(tz=timezone.utc).tz_localize(None) - pd.Timedelta(days=tage)
    return df[df["time"] >= cutoff].copy()


def _live_metrics(symbol: str, tage: int) -> tuple[int, float, float, float]:
    """Berechnet kompakte Live-Metriken aus Logs.

    Args:
        symbol: Währungssymbol.
        tage: Rückblick in Tagen.

    Returns:
        Tuple aus (signal_count, long_pct, short_pct, avg_prob).
    """
    df = _load_live_df(symbol, tage)
    if df is None or df.empty:
        return 0, 0.0, 0.0, 0.0

    total = len(df)
    long_count = int((df["richtung"] == "Long").sum()) if "richtung" in df.columns else 0
    short_count = int((df["richtung"] == "Short").sum()) if "richtung" in df.columns else 0
    avg_prob = float(df["prob"].mean()) if "prob" in df.columns else 0.0

    long_pct = (long_count / total) * 100 if total > 0 else 0.0
    short_pct = (short_count / total) * 100 if total > 0 else 0.0
    return total, long_pct, short_pct, avg_prob


def _latest_kpi_row(symbol: str) -> Optional[pd.Series]:
    """Liest den neuesten KPI-Historieneintrag für ein Symbol.

    Args:
        symbol: Währungssymbol.

    Returns:
        Letzte Zeile als Series oder None, wenn keine Historie vorhanden.
    """
    if not KPI_HISTORY_PATH.exists():
        return None

    df = pd.read_csv(KPI_HISTORY_PATH)
    if df.empty or "symbol" not in df.columns:
        return None

    sym_df = df[df["symbol"] == symbol].copy()
    if sym_df.empty:
        return None

    if "report_date" in sym_df.columns:
        sym_df["report_date"] = pd.to_datetime(sym_df["report_date"], errors="coerce")
        sym_df = sym_df.sort_values("report_date")
    else:
        sym_df = sym_df.reset_index(drop=True)

    return sym_df.iloc[-1]


def build_daily_snapshot(symbol: str, tage: int) -> DailySymbolSnapshot:
    """Kombiniert Live-Metriken mit letztem KPI-Referenzstand.

    Args:
        symbol: Währungssymbol.
        tage: Rückblick in Tagen.

    Returns:
        DailySymbolSnapshot mit robusten Fallback-Werten.
    """
    signals, long_pct, short_pct, avg_prob = _live_metrics(symbol, tage)
    latest = _latest_kpi_row(symbol)

    if latest is None:
        return DailySymbolSnapshot(
            symbol=symbol,
            signals_24h=signals,
            long_pct=long_pct,
            short_pct=short_pct,
            avg_prob=avg_prob,
            backtest_pf=0.0,
            backtest_sharpe=0.0,
            backtest_dd=0.0,
            backtest_wr=0.0,
            gate_status="UNKLAR",
        )

    return DailySymbolSnapshot(
        symbol=symbol,
        signals_24h=signals,
        long_pct=long_pct,
        short_pct=short_pct,
        avg_prob=avg_prob,
        backtest_pf=float(latest.get("profit_factor", 0.0)),
        backtest_sharpe=float(latest.get("sharpe_ratio", 0.0)),
        backtest_dd=float(latest.get("max_drawdown_pct", 0.0)),
        backtest_wr=float(latest.get("win_rate_pct", 0.0)),
        gate_status=str(latest.get("status", "UNKLAR")),
    )


def render_email_body(snapshots: list[DailySymbolSnapshot], tage: int) -> str:
    """Erzeugt den Klartext-Mailinhalt für den Daily Report.

    Args:
        snapshots: Symbol-Snapshots für den Report.
        tage: Rückblick in Tagen.

    Returns:
        Vollständiger Mailtext als String.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("MT5 Daily Performance Report")
    lines.append("=" * 36)
    lines.append(f"Zeitpunkt: {now_utc}")
    lines.append(f"Zeitraum Live-Signale: letzte {tage} Tag(e)")
    lines.append("Aktive Symbole: USDCAD, USDJPY")
    lines.append("")

    for snap in snapshots:
        lines.append(f"[{snap.symbol}] Status: {snap.gate_status}")
        lines.append(
            f"  Live: Signale={snap.signals_24h}, Long={snap.long_pct:.1f}%, "
            f"Short={snap.short_pct:.1f}%, AvgProb={snap.avg_prob:.3f}"
        )
        lines.append(
            f"  KPI-Referenz: PF={snap.backtest_pf:.3f}, Sharpe={snap.backtest_sharpe:.3f}, "
            f"DD={snap.backtest_dd:.2f}%, WR={snap.backtest_wr:.1f}%"
        )
        lines.append("")

    lines.append("Hinweis: Live-PnL wird aktuell nicht aus MT5-Closed-Trades aggregiert.")
    lines.append("         Profitabilitätsreferenz basiert auf dem letzten Backtest/KPI-Stand.")
    return "\n".join(lines)


def send_email(cfg: EmailConfig, subject: str, body: str) -> None:
    """Versendet die E-Mail über SMTP mit STARTTLS.

    Args:
        cfg: SMTP-Konfiguration.
        subject: Mail-Betreff.
        body: Klartext-Inhalt.

    Raises:
        smtplib.SMTPException: Bei Versandfehlern.
    """
    msg = MIMEText(body, _subtype="plain", _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient

    with smtplib.SMTP(cfg.host, cfg.port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(cfg.user, cfg.password)
        server.sendmail(cfg.sender, [cfg.recipient], msg.as_string())


def main() -> None:
    """Hauptablauf: Snapshots bauen, Mail rendern, optional versenden."""
    args = parse_args()

    snapshots = [build_daily_snapshot(symbol, args.tage) for symbol in AKTIVE_SYMBOLE]
    body = render_email_body(snapshots, args.tage)
    subject = f"[MT5 Paper] Daily KPI Update {datetime.now().strftime('%Y-%m-%d')}"

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN – kein Versand")
        print("=" * 72)
        print(f"Subject: {subject}")
        print(body)
        logger.info("Dry-Run abgeschlossen.")
        return

    cfg = load_email_config()
    send_email(cfg, subject, body)
    logger.info("Daily-Report erfolgreich versendet an %s", cfg.recipient)


if __name__ == "__main__":
    main()
