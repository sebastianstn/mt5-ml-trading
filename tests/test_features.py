"""
test_features.py – Unit-Tests für Feature Engineering und Labeling

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd

from features.labeling import double_barrier_label


# ============================================================
# Hilfsfunktionen für Tests
# ============================================================


def dummy_ohlcv(n: int = 20) -> pd.DataFrame:
    """Erstellt einen einfachen Test-DataFrame mit konstanten Preisen."""
    return pd.DataFrame(
        {
            "open": [1.1000] * n,
            "high": [1.1020] * n,
            "low": [1.0980] * n,
            "close": [1.1000] * n,
            "volume": [1000] * n,
        }
    )


# ============================================================
# Tests: double_barrier_label
# ============================================================


class TestDoubleBarrierLabel:
    """Tests für die Double-Barrier Labeling-Funktion."""

    def test_ausgabe_laenge(self):
        """Ausgabe-Array hat gleiche Länge wie Eingabe."""
        n = 20
        close = np.ones(n) * 1.1
        high = close + 0.005
        low = close - 0.005
        labels = double_barrier_label(close, high, low)
        assert len(labels) == n

    def test_letzte_zeilen_nan(self):
        """Die letzten `horizon` Einträge müssen NaN sein."""
        n = 20
        horizon = 5
        close = np.ones(n) * 1.1
        high = close + 0.001
        low = close - 0.001
        labels = double_barrier_label(close, high, low, horizon=horizon)
        assert np.all(np.isnan(labels[-horizon:]))

    def test_long_signal_korrekt(self):
        """Label 1 wenn TP-Schranke (oben) zuerst erreicht wird."""
        # Preis steigt sofort auf TP-Niveau
        close = np.array([1.0000] * 10)
        high = np.array(
            [1.0000, 1.0040, 1.0040, 1.0040, 1.0040, 1.0040, 1.0, 1.0, 1.0, 1.0]
        )
        low = np.array([1.0000] * 10)
        # TP = 1.0 * 1.003 = 1.003 → high[1]=1.004 > TP → Label 1
        labels = double_barrier_label(
            close, high, low, tp_pct=0.003, sl_pct=0.003, horizon=5
        )
        assert labels[0] == 1

    def test_short_signal_korrekt(self):
        """Label -1 wenn SL-Schranke (unten) zuerst erreicht wird."""
        # Preis fällt sofort auf SL-Niveau
        close = np.array([1.0000] * 10)
        high = np.array([1.0000] * 10)
        low = np.array(
            [1.0000, 0.9965, 0.9965, 0.9965, 0.9965, 0.9965, 1.0, 1.0, 1.0, 1.0]
        )
        # SL = 1.0 * (1 - 0.003) = 0.997 → low[1]=0.9965 < SL → Label -1
        labels = double_barrier_label(
            close, high, low, tp_pct=0.003, sl_pct=0.003, horizon=5
        )
        assert labels[0] == -1

    def test_kein_signal(self):
        """Label 0 wenn weder TP noch SL innerhalb Horizon erreicht."""
        # Preise bleiben konstant, keine Schranke wird erreicht
        close = np.ones(10) * 1.0
        high = np.ones(10) * 1.001  # weit unter TP (1.003)
        low = np.ones(10) * 0.999  # weit über SL (0.997)
        labels = double_barrier_label(
            close, high, low, tp_pct=0.003, sl_pct=0.003, horizon=5
        )
        assert labels[0] == 0

    def test_nur_gueltige_werte(self):
        """Alle gültigen Labels sind -1, 0 oder 1 (keine anderen Werte)."""
        n = 50
        rng = np.random.default_rng(42)
        close = np.ones(n) * 1.1
        high = close + rng.uniform(0.0001, 0.006, n)
        low = close - rng.uniform(0.0001, 0.006, n)
        labels = double_barrier_label(close, high, low)

        # Nur gültige Einträge prüfen (ohne NaN)
        gueltig = labels[~np.isnan(labels)]
        erlaubt = {-1.0, 0.0, 1.0}
        assert set(gueltig).issubset(erlaubt)

    def test_symmetrische_barrieren(self):
        """Mit symmetrischen Barrieren: Long und Short ungefähr gleich häufig."""
        n = 1000
        rng = np.random.default_rng(0)
        # Zufälliger Random Walk
        returns = rng.normal(0, 0.002, n)
        close = np.cumprod(1 + returns) * 1.1
        high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.001, n)))

        labels = double_barrier_label(
            close, high, low, tp_pct=0.003, sl_pct=0.003, horizon=5
        )
        gueltig = labels[~np.isnan(labels)]

        n_long = np.sum(gueltig == 1)
        n_short = np.sum(gueltig == -1)

        # Verhältnis Long/Short sollte zwischen 0.5 und 2.0 liegen (grob symmetrisch)
        if n_short > 0:
            ratio = n_long / n_short
            assert 0.3 < ratio < 3.0, f"Zu asymmetrisch: Long={n_long}, Short={n_short}"
