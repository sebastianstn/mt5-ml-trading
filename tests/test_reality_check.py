"""Tests für reports/reality_check.py (Fallback- und Robustheitsfälle)."""

import numpy as np
import pandas as pd

from reports import reality_check as rc  # pylint: disable=wrong-import-position


def test_wahrscheinlichkeit_ohne_prob_spalte() -> None:
    """Wenn 'prob' fehlt, sollen neutrale Standardwerte zurückkommen."""
    df = pd.DataFrame({"richtung": ["Long", "Short"]})
    result = rc.wahrscheinlichkeit_analysieren(df)

    assert result["avg_prob"] == 0.0
    assert result["median_prob"] == 0.0
    assert result["min_prob"] == 0.0


def test_regime_verteilung_ohne_spalte() -> None:
    """Wenn 'market_regime' fehlt, soll ein leeres Dict zurückkommen."""
    df = pd.DataFrame({"richtung": ["Long"]})
    result = rc.regime_verteilung_analysieren(df)

    assert result == {}


def test_exit_typen_ohne_spalte() -> None:
    """Wenn 'exit_grund' fehlt, soll ein leeres Dict zurückkommen."""
    df = pd.DataFrame({"richtung": ["Long", "Short"]})
    result = rc.exit_typen_analysieren(df)

    assert result == {}


def test_monatlicher_pnl_ohne_gewinn_hat_nan_winrate() -> None:
    """Ohne 'gewinn'-Spalte wird win_rate erzeugt und bleibt NaN."""
    index = pd.to_datetime(["2026-01-10", "2026-01-20", "2026-02-02"])
    df = pd.DataFrame({"pnl_pct": [0.01, -0.02, 0.03]}, index=index)

    result = rc.monatlicher_pnl_analysieren(df)

    assert not result.empty
    assert "win_rate" in result.columns
    assert result["win_rate"].isna().all()


def test_monatlicher_pnl_mit_gewinn_wird_prozent() -> None:
    """Mit 'gewinn'-Spalte wird die Win-Rate als Prozent skaliert."""
    index = pd.to_datetime(["2026-01-10", "2026-01-20", "2026-01-21"])
    df = pd.DataFrame(
        {
            "pnl_pct": [0.01, -0.02, 0.03],
            "gewinn": [1, 0, 1],
        },
        index=index,
    )

    result = rc.monatlicher_pnl_analysieren(df)

    assert len(result) == 1
    # 2 von 3 Trades gewonnen => 66.666...%
    assert np.isclose(result.loc[0, "win_rate"], (2 / 3) * 100, atol=1e-9)
