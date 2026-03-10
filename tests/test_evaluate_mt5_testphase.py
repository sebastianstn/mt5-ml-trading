"""Tests für scripts/evaluate_mt5_testphase.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from reports import (
    daily_phase7_dashboard as dph,
)  # pylint: disable=wrong-import-position
from reports import weekly_kpi_report as wk  # pylint: disable=wrong-import-position
from scripts import (
    evaluate_mt5_testphase as emt,
)  # pylint: disable=wrong-import-position
from scripts import verify_live_log_sync as vls  # pylint: disable=wrong-import-position


def test_close_grund_counts_leer_bei_fehlender_spalte() -> None:
    """Ohne close_grund-Spalte soll ein leeres Dict zurückkommen."""
    df = pd.DataFrame({"pnl_money": [1.0, -1.0]})

    result = emt.close_grund_counts(df)

    assert result == {}


def test_close_grund_counts_zaehlt_tp_und_sl() -> None:
    """Die Verteilung der Close-Gründe soll robust aggregiert werden."""
    df = pd.DataFrame(
        {
            "close_grund": ["TP", "SL", "TP", None, "MANUAL"],
        }
    )

    result = emt.close_grund_counts(df)

    assert result["TP"] == 2
    assert result["SL"] == 1
    assert result["MANUAL"] == 1
    assert result["UNBEKANNT"] == 1


def test_max_drawdown_pct_from_money_berechnet_negativen_drawdown() -> None:
    """Der maximale Drawdown soll als negativer Prozentwert zurückgegeben werden."""
    pnl_series = pd.Series([100.0, -50.0, -300.0, 200.0])

    result = emt.max_drawdown_pct_from_money(pnl_series, start_equity=10000.0)

    assert result is not None
    assert round(result, 2) == -3.5


def test_weekly_status_bevorzugt_live_closes() -> None:
    """Bei genug Live-Closes sollen die Live-KPIs die Statuslogik treiben."""
    status, hinweis = wk.status_bewerten(
        {
            "live_fresh": True,
            "live_signale": 12,
            "live_closes": 6,
            "live_profit_factor": 1.5,
            "live_max_drawdown_pct": -4.0,
            "live_win_rate_pct": 55.0,
            "profit_factor": 0.7,
            "sharpe_ratio": 0.1,
            "max_drawdown_pct": -25.0,
            "win_rate_pct": 30.0,
        }
    )

    assert status == "GO"
    assert "Live-Close-KPIs" in hinweis


def test_weekly_status_unklar_bei_zu_wenigen_live_closes() -> None:
    """Ein paar Live-Closes sind gut, aber noch nicht belastbar genug für GO/NO-GO."""
    status, hinweis = wk.status_bewerten(
        {
            "live_fresh": True,
            "live_signale": 12,
            "live_closes": 2,
            "live_profit_factor": 2.0,
            "live_max_drawdown_pct": -1.0,
            "live_win_rate_pct": 100.0,
            "profit_factor": 1.4,
            "sharpe_ratio": 1.0,
            "max_drawdown_pct": -5.0,
            "win_rate_pct": 50.0,
        }
    )

    assert status == "UNKLAR"
    assert "zu wenige" in hinweis


def test_daily_status_incident_bei_stale_logs() -> None:
    """Ohne frische Daten muss die Tagesampel INCIDENT zeigen."""
    status, reason = dph.status_bewerten(
        {
            "live_fresh": False,
            "signale": 4,
            "closes": 2,
            "net_pnl": 5.0,
            "profit_factor": 1.2,
            "max_drawdown_pct": -2.0,
        }
    )

    assert status == "INCIDENT"
    assert "frischen" in reason


def test_daily_status_watch_bei_negativer_pnl() -> None:
    """Frische Daten mit negativen Close-KPIs sollen WATCH ergeben."""
    status, reason = dph.status_bewerten(
        {
            "live_fresh": True,
            "signale": 5,
            "closes": 3,
            "net_pnl": -12.0,
            "profit_factor": 0.9,
            "max_drawdown_pct": -3.0,
        }
    )

    assert status == "WATCH"
    assert "Negative Tages-PnL" in reason


def test_daily_status_ok_bei_frischer_aktivitaet_ohne_closes() -> None:
    """Frische Signale ohne Closes sind operativ okay, aber noch ohne PnL-Stichprobe."""
    status, reason = dph.status_bewerten(
        {
            "live_fresh": True,
            "signale": 4,
            "closes": 0,
            "net_pnl": None,
            "profit_factor": None,
            "max_drawdown_pct": None,
        }
    )

    assert status == "OK"
    assert "noch keine Closes" in reason


def test_resolve_log_dir_auto_bevorzugt_frischsten_signal_ordner(
    tmp_path: Path,
) -> None:
    """Auto soll den jüngsten aktiven Signal-Ordner statt eines stale Root-Ordners wählen."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    paper_dir = logs_dir / "paper_test128"
    paper_dir.mkdir()
    train_dir = logs_dir / "train"
    train_dir.mkdir()

    root_signal = logs_dir / "USDCAD_signals.csv"
    root_signal.write_text("time,signal\n2026-03-10 10:00:00,2\n", encoding="utf-8")
    paper_signal = paper_dir / "USDCAD_signals.csv"
    paper_signal.write_text("time,signal\n2026-03-10 12:00:00,2\n", encoding="utf-8")

    os.utime(root_signal, (1_700_000_000, 1_700_000_000))
    os.utime(paper_signal, (1_800_000_000, 1_800_000_000))

    original_log_dir = dph.LOG_DIR
    try:
        dph.LOG_DIR = logs_dir
        result = dph.resolve_log_dir("auto", ("USDCAD", "USDJPY"))
    finally:
        dph.LOG_DIR = original_log_dir

    assert result == paper_dir


def test_resolve_log_dir_expliziter_pfad_bleibt_erhalten(tmp_path: Path) -> None:
    """Ein explizit gesetzter Pfad darf nicht durch Auto-Erkennung überschrieben werden."""
    custom_dir = tmp_path / "my_logs"
    custom_dir.mkdir()

    result = dph.resolve_log_dir(str(custom_dir), ("USDCAD",))

    assert result == custom_dir


def test_daily_dashboard_erkennt_runtime_csv_drift(tmp_path: Path) -> None:
    """Frischer Runtime-Heartbeat bei stale Signal-CSV soll als Logging-/Sync-Drift sichtbar werden."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    signal_time = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    signal_path = log_dir / "USDCAD_signals.csv"
    signal_path.write_text(
        "time,signal,prob,regime,regime_name\n"
        f"{signal_time},2,0.51,1,Aufwärtstrend\n",
        encoding="utf-8",
    )

    runtime_time = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=2)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    (log_dir / "live_trader.log").write_text(
        f"[USDCAD] Neue M5-Kerze | {runtime_time} UTC\n",
        encoding="utf-8",
    )

    status = dph.symbol_status_berechnen(
        symbol="USDCAD",
        log_dir=log_dir,
        hours=24,
        timeframe="M5_TWO_STAGE",
    )

    assert status.runtime_fresh is True
    assert status.live_fresh is False
    assert status.csv_runtime_lag_min is not None
    assert status.csv_runtime_lag_min > 20.0
    assert status.ampel == "INCIDENT"
    assert "Signal-CSV" in status.begruendung


def test_verify_live_log_sync_nutzt_csv_inhalt_statt_nur_mtime(tmp_path: Path) -> None:
    """Ein frisch kopiertes, aber inhaltlich altes CSV muss als nicht frisch gelten."""
    signal_path = tmp_path / "USDCAD_signals.csv"
    old_time = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=90)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    signal_path.write_text(
        "time,signal,prob,regime,regime_name\n" f"{old_time},2,0.55,1,Aufwärtstrend\n",
        encoding="utf-8",
    )
    os.utime(signal_path, None)

    result = vls.check_single_file(
        symbol="USDCAD",
        kind="signals",
        file_path=signal_path,
        max_age_minutes=10.0,
    )

    assert result.exists is True
    assert result.fresh is False
    assert result.age_minutes is not None
    assert result.age_minutes > 60.0
    assert result.mtime_age_minutes is not None
    assert result.mtime_age_minutes < 1.0
    assert result.content_timestamp_utc is not None


def test_watchdog_check_erkennt_fehlende_datei(tmp_path: Path) -> None:
    """Fehlende Watchdog-Datei soll klar als ungesund markiert werden."""
    result = vls.check_watchdog_file(log_dir=tmp_path, max_age_minutes=10.0)

    assert result.exists is False
    assert result.healthy is False
    assert result.reason == "Watchdog-Datei fehlt"


def test_watchdog_check_erkennt_incident_payload(tmp_path: Path) -> None:
    """Eine frische Watchdog-Datei mit overall_status INCIDENT soll die Prüfung rot machen."""
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "INCIDENT",
        "symbols": [],
    }
    (tmp_path / vls.WATCHDOG_JSON_NAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = vls.check_watchdog_file(log_dir=tmp_path, max_age_minutes=10.0)

    assert result.exists is True
    assert result.fresh is True
    assert result.healthy is False
    assert result.overall_status == "INCIDENT"
    assert result.reason == "Watchdog meldet INCIDENT"


def test_watchdog_snapshot_wird_aus_json_geladen(tmp_path: Path) -> None:
    """Das Dashboard soll synchronisierte Watchdog-Infos pro Symbol aus JSON laden."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    payload = {
        "overall_status": "INCIDENT",
        "symbols": [
            {
                "symbol": "USDCAD",
                "status": "INCIDENT",
                "reason": "Signal-CSV hinkt hinter Runtime-Heartbeat",
            },
            {
                "symbol": "USDJPY",
                "status": "OK",
                "reason": "Alles frisch",
            },
        ],
    }
    (log_dir / dph.WATCHDOG_JSON_NAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    result = dph.load_watchdog_snapshot(log_dir)

    assert result.overall_status == "INCIDENT"
    assert result.symbols["USDCAD"]["status"] == "INCIDENT"
    assert "Runtime" in result.symbols["USDCAD"]["reason"]
    assert result.symbols["USDJPY"]["status"] == "OK"


def test_daily_dashboard_watchdog_incident_eskaliert_ampel(tmp_path: Path) -> None:
    """Ein Watchdog-INCIDENT soll die Dashboard-Ampel direkt auf INCIDENT setzen."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    signal_time = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=2)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    (log_dir / "USDCAD_signals.csv").write_text(
        "time,signal,prob,regime,regime_name\n"
        f"{signal_time},2,0.61,1,Aufwärtstrend\n",
        encoding="utf-8",
    )
    (log_dir / "live_trader.log").write_text(
        f"[USDCAD] Neue M5-Kerze | {signal_time} UTC\n",
        encoding="utf-8",
    )

    status = dph.symbol_status_berechnen(
        symbol="USDCAD",
        log_dir=log_dir,
        hours=24,
        timeframe="M5_TWO_STAGE",
        watchdog_snapshot=dph.WatchdogSnapshot(
            overall_status="INCIDENT",
            generated_at_utc="2026-03-10 14:00:00",
            stale_limit_minutes=7.5,
            lag_limit_minutes=15.0,
            symbols={
                "USDCAD": {
                    "status": "INCIDENT",
                    "reason": "Windows-Watchdog meldet Drift",
                }
            },
        ),
    )

    assert status.watchdog_status == "INCIDENT"
    assert status.ampel == "INCIDENT"
    assert status.begruendung == "Windows-Watchdog meldet Drift"


def test_markdown_dashboard_enthaelt_watchdog_ueberblick(tmp_path: Path) -> None:
    """Der Markdown-Report soll eine kompakte Watchdog-Zusammenfassung oberhalb der Tabelle zeigen."""
    output_dir = tmp_path / "reports"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    watchdog_path = log_dir / dph.WATCHDOG_JSON_NAME
    watchdog_path.write_text("{}", encoding="utf-8")

    status = dph.DailySymbolStatus(
        symbol="USDCAD",
        ampel="WATCH",
        begruendung="Test-Begründung",
        live_fresh=True,
        last_event_utc="2026-03-10 13:55:09+00:00",
        minutes_since_last=2.0,
        runtime_fresh=True,
        last_runtime_utc="2026-03-10 13:55:00+00:00",
        minutes_since_runtime=2.0,
        csv_runtime_lag_min=0.1,
        watchdog_status="WATCH",
        watchdog_reason="Signalfluss ok, aber noch keine Close-CSV vorhanden",
        events=5,
        signale=2,
        closes=0,
        long_signale=1,
        short_signale=1,
        tp_closes=0,
        sl_closes=0,
        net_pnl=None,
        profit_factor=None,
        win_rate_pct=None,
        max_drawdown_pct=None,
        avg_prob_pct=52.3,
        avg_dauer_min=None,
    )
    snapshot = dph.WatchdogSnapshot(
        overall_status="WATCH",
        generated_at_utc="2026-03-10 14:00:00",
        stale_limit_minutes=7.5,
        lag_limit_minutes=15.0,
        symbols={
            "USDCAD": {
                "status": "WATCH",
                "reason": "Signalfluss ok, aber noch keine Close-CSV vorhanden",
            }
        },
    )

    md_path = dph.markdown_dashboard_schreiben(
        statuses=[status],
        output_dir=output_dir,
        hours=24,
        timeframe="M5_TWO_STAGE",
        log_dir=log_dir,
        watchdog_snapshot=snapshot,
    )

    content = md_path.read_text(encoding="utf-8")
    assert "## Watchdog-Überblick" in content
    assert "**Watchdog Gesamtstatus:** WATCH" in content
    assert "**Watchdog erstellt (UTC):** 2026-03-10 14:00:00" in content
    assert "**Watchdog Lag-Limit Min:** 15.0" in content
