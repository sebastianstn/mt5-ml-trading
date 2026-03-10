"""
two_stage_signal.py – Inferenz-Helfer für Option 1 (HTF -> LTF)

Dieses Modul kombiniert zwei bereits trainierte Modelle:
    1) HTF-Bias-Modell (H1)
    2) LTF-Entry-Modell (M5/M15)

Es ist als Baustein für live_trader.py gedacht und kann zunächst
im Paper-Trading getestet werden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol, Tuple, cast

import joblib
import numpy as np
import pandas as pd


@dataclass
class ZweiStufenSignal:
    """Ergebniscontainer für das kombinierte Zwei-Stufen-Signal.

    Attributes:
        signal: -1 (Short), 0 (Neutral), 2 (Long).
        prob: Eintrittswahrscheinlichkeit des gewählten Signals.
        htf_bias_klasse: HTF-Bias-Klasse als 0/1/2.
        htf_bias_proba: HTF-Bias-Wahrscheinlichkeiten je Klasse.
        ltf_klasse: Rohklasse des LTF-Modells als 0/1/2.
        ltf_signal_vor_filter: LTF-Signal vor Schwellenwertfilter.
        ltf_entry_erlaubt: True wenn der LTF-Schwellenfilter den Entry erlaubt.
        ltf_proba: LTF-Wahrscheinlichkeiten je Klasse.
    """

    signal: int
    prob: float
    htf_bias_klasse: int
    htf_bias_proba: Dict[str, float]
    ltf_klasse: int
    ltf_signal_vor_filter: int
    ltf_entry_erlaubt: bool
    ltf_proba: Dict[str, float]


class WahrscheinlichkeitsModell(Protocol):
    """Minimales Protokoll für Modelle mit predict_proba."""

    def predict_proba(self, x_features: pd.DataFrame) -> Any:
        """Gibt Klassenwahrscheinlichkeiten für die übergebenen Features zurück."""


def _modell_predict_proba(modell: object, x_features: pd.DataFrame) -> np.ndarray:
    """Ruft predict_proba typrobust auf und liefert ein NumPy-Array zurück."""
    modell_any = cast(Any, modell)
    return np.asarray(modell_any.predict_proba(x_features), dtype=float)


def _letzte_geschlossene_zeile(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt die letzte geschlossene Kerze als 1-Zeilen-DataFrame zurück.

    Konvention aus dem Projekt: `iloc[-2]` ist die letzte geschlossene Kerze,
    da `iloc[-1]` die laufende Kerze sein kann.

    Args:
        df: DataFrame mit zeitlich sortierten Kerzen.

    Returns:
        1-Zeilen-DataFrame mit letzter geschlossener Kerze.
    """
    if len(df) < 2:
        raise ValueError(
            "Zu wenige Zeilen für geschlossene Kerze (mindestens 2 nötig)."
        )
    return df.iloc[[-2]].copy()


def _klasse_zu_signal(klasse: int) -> int:
    """Mappt Klassenindex 0/1/2 auf Trading-Signal -1/0/2.

    Args:
        klasse: Klassenindex des LTF-Modells.

    Returns:
        Trading-Signal im Projektformat.
    """
    if klasse == 0:
        return -1
    if klasse == 2:
        return 2
    return 0


def modelle_laden(
    models_dir: Path,
    symbol: str,
    ltf_timeframe: str,
    version: str,
) -> Tuple[object, object]:
    """Lädt HTF- und LTF-Modell für ein Symbol.

    Args:
        models_dir: Pfad zu `models/`.
        symbol: Handelssymbol (USDCAD, USDJPY ...).
        ltf_timeframe: LTF-Zeitrahmen (M5/M15).
        version: Versionssuffix (v1, v2 ...).

    Returns:
        Tuple (htf_model, ltf_model).
    """
    htf_path = models_dir / f"lgbm_htf_bias_{symbol.lower()}_H1_{version}.pkl"
    ltf_path = (
        models_dir / f"lgbm_ltf_entry_{symbol.lower()}_{ltf_timeframe}_{version}.pkl"
    )

    if not htf_path.exists():
        raise FileNotFoundError(f"HTF-Modell nicht gefunden: {htf_path}")
    if not ltf_path.exists():
        raise FileNotFoundError(f"LTF-Modell nicht gefunden: {ltf_path}")

    return joblib.load(htf_path), joblib.load(ltf_path)


def zwei_stufen_signal(
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    htf_model: object,
    ltf_model: object,
    htf_feature_spalten: list[str],
    ltf_feature_spalten: list[str],
    schwelle: float = 0.55,
) -> ZweiStufenSignal:
    """Erzeugt ein kombiniertes Signal aus HTF-Bias und LTF-Entry.

    Ablauf:
        1) HTF-Bias auf letzter geschlossener H1-Kerze vorhersagen.
        2) Bias als Zusatzfeatures in LTF-Featurevektor eintragen.
        3) LTF-Entry vorhersagen und Schwellenwertfilter anwenden.

    Args:
        htf_df: H1-Feature-DataFrame.
        ltf_df: LTF-Feature-DataFrame (M5/M15).
        htf_model: Trainiertes HTF-Modell.
        ltf_model: Trainiertes LTF-Modell.
        htf_feature_spalten: Featureliste des HTF-Modells.
        ltf_feature_spalten: Featureliste des LTF-Modells.
        schwelle: Mindestwahrscheinlichkeit für Short/Long.

    Returns:
        ZweiStufenSignal mit Signal und Diagnoseinformationen.
    """
    # 1) HTF-Bias bestimmen.
    htf_row = _letzte_geschlossene_zeile(htf_df)
    htf_x = htf_row.reindex(columns=htf_feature_spalten).copy()
    htf_x = htf_x.fillna(htf_x.median(numeric_only=True)).fillna(0.0)

    htf_proba_arr = _modell_predict_proba(htf_model, htf_x)[0]
    htf_klasse = int(np.argmax(htf_proba_arr))

    # 2) LTF-Features vorbereiten und HTF-Bias anhängen.
    ltf_row = _letzte_geschlossene_zeile(ltf_df)
    ltf_x = ltf_row.reindex(columns=ltf_feature_spalten).copy()

    if "htf_bias_class" in ltf_x.columns:
        ltf_x.loc[:, "htf_bias_class"] = htf_klasse
    if "htf_bias_prob_short" in ltf_x.columns:
        ltf_x.loc[:, "htf_bias_prob_short"] = float(htf_proba_arr[0])
    if "htf_bias_prob_neutral" in ltf_x.columns:
        ltf_x.loc[:, "htf_bias_prob_neutral"] = float(htf_proba_arr[1])
    if "htf_bias_prob_long" in ltf_x.columns:
        ltf_x.loc[:, "htf_bias_prob_long"] = float(htf_proba_arr[2])

    ltf_x = ltf_x.fillna(ltf_x.median(numeric_only=True)).fillna(0.0)

    # 3) LTF-Signal erzeugen.
    ltf_proba_arr = _modell_predict_proba(ltf_model, ltf_x)[0]
    ltf_klasse = int(np.argmax(ltf_proba_arr))
    signal = _klasse_zu_signal(ltf_klasse)
    ltf_signal_vor_filter = signal
    ltf_entry_erlaubt = True

    # Schwellenwert nur für aktive Trades anwenden.
    if signal == 2 and float(ltf_proba_arr[2]) < schwelle:
        signal = 0
        ltf_entry_erlaubt = False
    if signal == -1 and float(ltf_proba_arr[0]) < schwelle:
        signal = 0
        ltf_entry_erlaubt = False

    if signal == 2:
        prob = float(ltf_proba_arr[2])
    elif signal == -1:
        prob = float(ltf_proba_arr[0])
    else:
        prob = float(np.max(ltf_proba_arr))

    return ZweiStufenSignal(
        signal=signal,
        prob=prob,
        htf_bias_klasse=htf_klasse,
        htf_bias_proba={
            "short": float(htf_proba_arr[0]),
            "neutral": float(htf_proba_arr[1]),
            "long": float(htf_proba_arr[2]),
        },
        ltf_klasse=ltf_klasse,
        ltf_signal_vor_filter=ltf_signal_vor_filter,
        ltf_entry_erlaubt=ltf_entry_erlaubt,
        ltf_proba={
            "short": float(ltf_proba_arr[0]),
            "neutral": float(ltf_proba_arr[1]),
            "long": float(ltf_proba_arr[2]),
        },
    )
