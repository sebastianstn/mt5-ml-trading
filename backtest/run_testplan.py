#!/usr/bin/env python3
"""
run_testplan.py - Führt 30 systematische Backtests durch

Liest testplan_30configs.csv und führt für jede Konfiguration einen
kompletten Backtest auf USDCAD + USDJPY durch.

Workflow:
1. Testplan einlesen
2. Für jede Config: two_stage_backtest.py via subprocess aufrufen
3. CSV-Ergebnisse sammeln
4. Gesamt-Ranking erstellen

Läuft auf: Linux-Server (benötigt ~15-30 Min für 30 Backtests)
"""

import argparse
import subprocess
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import pandas as pd

# Projekt-Root
BASE_DIR = Path(__file__).parent.parent
BACKTEST_DIR = BASE_DIR / "backtest"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def testplan_einlesen(csv_pfad: Path) -> pd.DataFrame:
    """Testplan-CSV einlesen und validieren."""
    if not csv_pfad.exists():
        raise FileNotFoundError(f"Testplan nicht gefunden: {csv_pfad}")

    df = pd.read_csv(csv_pfad)
    logger.info(f"Testplan geladen: {len(df)} Konfigurationen")

    # Pflicht-Spalten prüfen
    required = [
        "test_id",
        "schwelle",
        "tp_pct",
        "sl_pct",
        "cooldown_bars",
        "regime_filter",
        "atr_faktor",
        "horizon",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im Testplan: {missing}")

    return df


def regime_filter_parsen(filter_str: Any) -> str:
    """Parse '0|1|2' → '0,1,2' für CLI."""
    if pd.isna(filter_str):
        return ""

    cleaned = str(filter_str).strip()
    if cleaned.lower() == "alle":
        return ""

    return cleaned.replace("|", ",")


def backtest_durchfuehren(
    test_id: int,
    schwelle: float,
    tp_pct: float,
    sl_pct: float,
    cooldown_bars: int,
    regime_filter: str,
    atr_faktor: float,
    horizon: int,
    output_dir: Path,
) -> Optional[Path]:
    """
    Führt einen Two-Stage-Backtest via subprocess durch.

    Returns:
        Path zur CSV-Datei mit Ergebnissen oder None bei Fehler
    """
    try:
        expected_summary = BACKTEST_DIR / "two_stage_backtest_summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(BACKTEST_DIR / "two_stage_backtest.py"),
            "--symbol",
            "USDCAD",
            "USDJPY",
            "--version",
            "v4",
            "--ltf_timeframe",
            "M5",
            "--schwelle",
            str(schwelle),
            "--tp_pct",
            str(tp_pct),
            "--sl_pct",
            str(sl_pct),
            "--horizon",
            str(horizon),
            "--atr_faktor",
            str(atr_faktor),
            "--cooldown_bars",
            str(cooldown_bars),
        ]

        if regime_filter:
            cmd.extend(["--regime_filter", regime_filter])

        logger.info(f"[Test {test_id}] Running backtest...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 Min pro Backtest
            cwd=str(BASE_DIR),
        )

        if result.returncode != 0:
            logger.error(
                f"[Test {test_id}] Backtest fehlgeschlagen:\n"
                f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
            )
            return None

        # Erwartete Summary-Datei prüfen (stabiler als glob über Alt-Dateien).
        if not expected_summary.exists():
            logger.error(f"[Test {test_id}] Keine Summary-CSV gefunden")
            return None

        # Pro Test-ID separat kopieren (Original bleibt für den nächsten Lauf erhalten).
        renamed = output_dir / f"test_{test_id:03d}_summary.csv"
        if renamed.exists():
            renamed.unlink()
        shutil.copy2(expected_summary, renamed)

        logger.info(f"[Test {test_id}] Ergebnisse gespeichert: {renamed.name}")
        return renamed

    except subprocess.TimeoutExpired:
        logger.error(f"[Test {test_id}] Timeout nach 5 Min")
        return None
    except Exception as e:
        logger.error(f"[Test {test_id}] Fehler: {e}")
        return None


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Gibt die erste vorhandene Spalte aus der Kandidatenliste zurück."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main() -> None:
    """Hauptfunktion: Testplan durcharbeiten und Ergebnisse sammeln."""

    parser = argparse.ArgumentParser(
        description="Führt einen Testplan mit Two-Stage-Backtests aus"
    )
    parser.add_argument(
        "--testplan",
        type=Path,
        default=BACKTEST_DIR / "testplan_30configs.csv",
        help="Pfad zur Testplan-CSV (Standard: backtest/testplan_30configs.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=BACKTEST_DIR,
        help="Zielordner für Summary-/Ranking-/Ergebnisdateien (Standard: backtest)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TESTPLAN-RUNNER")
    logger.info("=" * 70)

    # Testplan laden
    testplan_pfad = args.testplan
    if not testplan_pfad.is_absolute():
        testplan_pfad = (BASE_DIR / testplan_pfad).resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (BASE_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    testplan = testplan_einlesen(testplan_pfad)
    anzahl_tests = len(testplan)
    logger.info("Aktiver Testplan: %s", testplan_pfad.name)
    logger.info("Konfigurationen im Plan: %s", anzahl_tests)
    logger.info("Zielordner: %s", output_dir)

    # Ergebnisse sammeln
    all_results = []

    for idx, row in testplan.iterrows():
        test_id = int(row["test_id"])
        regime_filter = regime_filter_parsen(row["regime_filter"])

        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {test_id}/{anzahl_tests}: {row['description']}")
        logger.info(
            f"Config: schwelle={row['schwelle']:.2f}, TP/SL={row['tp_pct']:.1%}/{row['sl_pct']:.1%} "
            f"(RRR {row['rrr']:.1f}:1), cooldown={int(row['cooldown_bars'])}, "
            f"regime={row['regime_filter']}, ATR={row['atr_faktor']:.1f}x, horizon={int(row['horizon'])}"
        )
        logger.info(f"{'='*70}")

        summary_pfad = backtest_durchfuehren(
            test_id=test_id,
            schwelle=row["schwelle"],
            tp_pct=row["tp_pct"],
            sl_pct=row["sl_pct"],
            cooldown_bars=int(row["cooldown_bars"]),
            regime_filter=regime_filter,
            atr_faktor=row["atr_faktor"],
            horizon=int(row["horizon"]),
            output_dir=output_dir,
        )

        if summary_pfad is None:
            logger.warning(f"Test {test_id} fehlgeschlagen – übersprungen")
            continue

        # CSV einlesen und test_id + config hinzufügen
        try:
            summary_df = pd.read_csv(summary_pfad)
            summary_df["test_id"] = test_id
            summary_df["description"] = row["description"]
            summary_df["schwelle"] = row["schwelle"]
            summary_df["tp_pct"] = row["tp_pct"]
            summary_df["sl_pct"] = row["sl_pct"]
            summary_df["rrr"] = row["rrr"]
            summary_df["cooldown_bars"] = row["cooldown_bars"]
            summary_df["regime_filter"] = row["regime_filter"]
            summary_df["atr_faktor"] = row["atr_faktor"]
            summary_df["horizon"] = row["horizon"]
            all_results.append(summary_df)

            # Kurz-Status loggen
            for _, r in summary_df.iterrows():
                symbol = r.get("symbol", "N/A")
                if "TWO_STAGE" in str(r.get("modell", "")):
                    ret = r.get("two_stage_return_pct", 0)
                    sharpe = r.get("two_stage_sharpe", 0)
                    trades = r.get("two_stage_n_trades", 0)
                    logger.info(
                        f"  [{symbol}] Return={ret:.2f}% | Sharpe={sharpe:.2f} | Trades={trades}"
                    )
        except Exception as e:
            logger.error(f"Fehler beim Einlesen von {summary_pfad}: {e}")

    if not all_results:
        logger.error("Keine Ergebnisse gesammelt!")
        return

    # Alle Ergebnisse zusammenführen
    results_df = pd.concat(all_results, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pfad = output_dir / f"testplan_results_{timestamp}.csv"
    results_df.to_csv(output_pfad, index=False, float_format="%.6f")
    logger.info(f"\n✓ Alle Ergebnisse gespeichert: {output_pfad.name}")

    # Symlink auf "latest"
    latest_pfad = output_dir / "testplan_results_latest.csv"
    if latest_pfad.exists():
        latest_pfad.unlink()
    latest_pfad.symlink_to(output_pfad.name)

    # Ranking erstellen (Two-Stage Sharpe Ratio)
    logger.info("\n" + "=" * 70)
    logger.info("RANKING NACH TWO-STAGE SHARPE RATIO")
    logger.info("=" * 70)

    # Filter: nur Two-Stage Zeilen
    modell_col = _first_existing_col(results_df, ["modell"])
    if modell_col is not None:
        ts_df = results_df[
            results_df[modell_col].astype(str).str.contains("TWO_STAGE", na=False)
        ].copy()
    else:
        # Summary aus two_stage_backtest.py enthält nur Two-Stage/Baseline-Metriken pro Symbol.
        ts_df = results_df.copy()

    if ts_df.empty:
        logger.warning("Keine Two-Stage Ergebnisse zum Ranken!")
        return

    sharpe_col = _first_existing_col(ts_df, ["two_stage_sharpe"])
    return_col = _first_existing_col(ts_df, ["two_stage_return_pct"])
    maxdd_col = _first_existing_col(
        ts_df, ["two_stage_max_dd_pct", "two_stage_maxdd_pct"]
    )
    trades_col = _first_existing_col(ts_df, ["two_stage_n_trades"])
    pf_col = _first_existing_col(ts_df, ["two_stage_profit_factor", "two_stage_pf"])

    required_metric_cols = [
        ("two_stage_sharpe", sharpe_col),
        ("two_stage_return_pct", return_col),
        ("two_stage_n_trades", trades_col),
    ]
    missing_metrics = [name for name, col in required_metric_cols if col is None]
    if missing_metrics:
        logger.error(f"Für Ranking fehlen Pflichtspalten: {missing_metrics}")
        return

    # Pro Test-ID: Durchschnitt über beide Symbole
    agg_results = []
    for test_id in ts_df["test_id"].unique():
        test_data = ts_df[ts_df["test_id"] == test_id]

        agg_results.append(
            {
                "test_id": int(test_id),
                "description": test_data.iloc[0]["description"],
                "schwelle": test_data.iloc[0]["schwelle"],
                "rrr": test_data.iloc[0]["rrr"],
                "cooldown_bars": test_data.iloc[0]["cooldown_bars"],
                "regime_filter": test_data.iloc[0]["regime_filter"],
                "atr_faktor": test_data.iloc[0]["atr_faktor"],
                "horizon": test_data.iloc[0]["horizon"],
                "avg_sharpe": test_data[sharpe_col].mean(),
                "avg_return": test_data[return_col].mean(),
                "avg_max_dd": test_data[maxdd_col].mean() if maxdd_col else 0.0,
                "total_trades": test_data[trades_col].sum(),
                "avg_profit_factor": test_data[pf_col].mean() if pf_col else 0.0,
            }
        )

    agg_df = pd.DataFrame(agg_results)
    agg_df = agg_df.sort_values("avg_sharpe", ascending=False)

    # Ranking speichern
    ranking_pfad = output_dir / f"testplan_ranking_{timestamp}.csv"
    agg_df.to_csv(ranking_pfad, index=False, float_format="%.4f")
    logger.info(f"\n✓ Ranking gespeichert: {ranking_pfad.name}")

    # Symlink
    ranking_latest = output_dir / "testplan_ranking_latest.csv"
    if ranking_latest.exists():
        ranking_latest.unlink()
    ranking_latest.symlink_to(ranking_pfad.name)

    # Top-10 anzeigen
    logger.info("\nTOP-10 KONFIGURATIONEN:")
    logger.info("-" * 100)
    logger.info(
        f"{'#':<3} {'Test':<5} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'PF':>5} {'Description':<30}"
    )
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(agg_df.head(10).iterrows(), start=1):
        logger.info(
            f"{rank:<3} {int(row['test_id']):<5} "
            f"{row['avg_sharpe']:>7.2f} {row['avg_return']:>7.2f}% "
            f"{row['avg_max_dd']:>6.2f}% {int(row['total_trades']):>7} "
            f"{row['avg_profit_factor']:>5.2f} {row['description']:<30}"
        )

    logger.info("\n" + "=" * 70)
    logger.info("TESTLAUF ABGESCHLOSSEN")
    logger.info(f"Vollständige Ergebnisse: {output_pfad.name}")
    logger.info(f"Ranking: {ranking_pfad.name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
