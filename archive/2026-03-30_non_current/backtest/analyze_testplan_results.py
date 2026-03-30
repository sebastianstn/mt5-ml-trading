#!/usr/bin/env python3
"""
analyze_testplan_results.py - Analysiert die Ergebnisse der 30 Tests

Liest testplan_results_latest.csv und erstellt ein Ranking nach Sharpe Ratio.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Projekt-Root
BASE_DIR = Path(__file__).parent.parent
BACKTEST_DIR = BASE_DIR / "backtest"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Gibt die erste vorhandene Spalte aus der Kandidatenliste zurück."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _symbol_metric(
    test_data: pd.DataFrame,
    symbol: str,
    metric_col: Optional[str],
    default: float = 0.0,
) -> float:
    """Liest eine Kennzahl für ein Symbol sicher aus dem Test-Subset."""
    if metric_col is None:
        return default

    symbol_rows = test_data[test_data["symbol"] == symbol]
    if symbol_rows.empty:
        return default

    value = symbol_rows.iloc[0].get(metric_col, default)
    return float(value) if pd.notna(value) else default


def main():
    """Ranking aus existierenden Ergebnissen erstellen."""

    # CSV laden
    results_pfad = BACKTEST_DIR / "testplan_results_latest.csv"
    if not results_pfad.exists():
        results_pfad = BACKTEST_DIR / "testplan_results_20260309_121528.csv"

    if not results_pfad.exists():
        logger.error("Keine Ergebnisse vorhanden!")
        return

    logger.info("Lade Ergebnisse aus: %s", results_pfad.name)
    results_df = pd.read_csv(results_pfad)

    logger.info("Gefunden: %s Zeilen (je 2 Symbole pro Test)", len(results_df))

    sharpe_col = _first_existing_col(results_df, ["two_stage_sharpe"])
    return_col = _first_existing_col(results_df, ["two_stage_return_pct"])
    maxdd_col = _first_existing_col(
        results_df, ["two_stage_max_dd_pct", "two_stage_maxdd_pct"]
    )
    trades_col = _first_existing_col(results_df, ["two_stage_n_trades"])
    pf_col = _first_existing_col(
        results_df, ["two_stage_profit_factor", "two_stage_pf"]
    )

    required_metric_cols = [
        ("two_stage_sharpe", sharpe_col),
        ("two_stage_return_pct", return_col),
        ("two_stage_n_trades", trades_col),
    ]
    missing_metrics = [name for name, col in required_metric_cols if col is None]
    if missing_metrics:
        logger.error("Für die Analyse fehlen Pflichtspalten: %s", missing_metrics)
        return

    # Pro Test-ID: Durchschnitt über beide Symbole berechnen
    agg_results = []
    for test_id in sorted(results_df["test_id"].unique()):
        test_data = results_df[results_df["test_id"] == test_id]

        if len(test_data) < 2:
            logger.warning("Test %s: nur %s Symbole gefunden", test_id, len(test_data))
            continue

        agg_results.append(
            {
                "test_id": int(test_id),
                "description": test_data.iloc[0]["description"],
                "schwelle": test_data.iloc[0]["schwelle"],
                "rrr": test_data.iloc[0]["rrr"],
                "cooldown_bars": int(test_data.iloc[0]["cooldown_bars"]),
                "regime_filter": test_data.iloc[0]["regime_filter"],
                "atr_faktor": test_data.iloc[0]["atr_faktor"],
                "horizon": int(test_data.iloc[0]["horizon"]),
                "avg_sharpe": test_data[sharpe_col].mean(),
                "avg_return": test_data[return_col].mean(),
                "avg_max_dd": test_data[maxdd_col].mean() if maxdd_col else 0.0,
                "total_trades": test_data[trades_col].sum(),
                "avg_profit_factor": test_data[pf_col].mean() if pf_col else 0.0,
                "usdcad_sharpe": _symbol_metric(test_data, "USDCAD", sharpe_col),
                "usdjpy_sharpe": _symbol_metric(test_data, "USDJPY", sharpe_col),
                "usdcad_return": _symbol_metric(test_data, "USDCAD", return_col),
                "usdjpy_return": _symbol_metric(test_data, "USDJPY", return_col),
            }
        )

    agg_df = pd.DataFrame(agg_results)
    agg_df = agg_df.sort_values("avg_sharpe", ascending=False)

    # Ranking speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ranking_pfad = BACKTEST_DIR / f"testplan_ranking_{timestamp}.csv"
    agg_df.to_csv(ranking_pfad, index=False, float_format="%.4f")
    logger.info("\n✓ Ranking gespeichert: %s", ranking_pfad.name)

    # Symlink
    ranking_latest = BACKTEST_DIR / "testplan_ranking_latest.csv"
    if ranking_latest.exists():
        ranking_latest.unlink()
    ranking_latest.symlink_to(ranking_pfad.name)

    # Top-10 anzeigen
    logger.info("\n%s", "=" * 100)
    logger.info("TOP-10 KONFIGURATIONEN (sortiert nach Ø Sharpe Ratio)")
    logger.info("%s", "=" * 100)
    logger.info(
        "%s",
        f"{'#':<3} {'Test':<5} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'PF':>5} {'Description':<35}",
    )
    logger.info("-" * 100)

    for rank, (_, row) in enumerate(agg_df.head(10).iterrows(), start=1):
        logger.info(
            "%s",
            f"{rank:<3} {int(row['test_id']):<5} "
            f"{row['avg_sharpe']:>7.2f} {row['avg_return']:>7.2f}% "
            f"{row['avg_max_dd']:>6.2f}% {int(row['total_trades']):>7} "
            f"{row['avg_profit_factor']:>5.2f} {row['description']:<35}",
        )

    # Details zu Top-3
    logger.info("\n%s", "=" * 100)
    logger.info("TOP-3 DETAILS (pro Symbol)")
    logger.info("%s", "=" * 100)

    for rank, (_, row) in enumerate(agg_df.head(3).iterrows(), start=1):
        logger.info("\n#%s Test %s: %s", rank, int(row["test_id"]), row["description"])
        logger.info(
            "%s",
            f"   Config: schwelle={row['schwelle']:.2f}, RRR={row['rrr']:.1f}:1, cooldown={int(row['cooldown_bars'])}, "
            f"regime={row['regime_filter']}, ATR={row['atr_faktor']:.1f}x, horizon={int(row['horizon'])}",
        )
        logger.info(
            "   USDCAD: Sharpe=%.2f, Return=%.2f%%",
            row["usdcad_sharpe"],
            row["usdcad_return"],
        )
        logger.info(
            "   USDJPY: Sharpe=%.2f, Return=%.2f%%",
            row["usdjpy_sharpe"],
            row["usdjpy_return"],
        )
        logger.info(
            "   Gesamt: Ø Sharpe=%.2f, Ø Return=%.2f%%, Ø MaxDD=%.2f%%, Total Trades=%s",
            row["avg_sharpe"],
            row["avg_return"],
            row["avg_max_dd"],
            int(row["total_trades"]),
        )

    logger.info("\n%s", "=" * 100)
    logger.info("ANALYSE ABGESCHLOSSEN")
    logger.info("Ranking-Datei: %s", ranking_pfad.name)
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
