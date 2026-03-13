"""
signal_engine.py – Signal-Generierung (Modell-Vorhersage)

Bewertet Features mit dem LightGBM-Modell und generiert Trade-Signale.
Unterstützt Single-Stage (H1) und Two-Stage Shadow-Mode (H1 + M5).

Läuft auf: Windows 11 Laptop
"""

import logging
from typing import Any, Optional, Tuple, cast

import numpy as np
import pandas as pd

try:
    from live import config  # Import als Paket (z.B. pytest, externe Aufrufe)
except ImportError:
    import config  # Import direkt aus live/-Verzeichnis

logger = logging.getLogger(__name__)


def _modell_feature_namen(modell: object) -> list[str]:
    """Liest Feature-Namen robust vom Modell oder fällt auf Standard-Features zurück."""
    modell_any = cast(Any, modell)
    namen = getattr(modell_any, "feature_name_", None)
    if namen:
        return list(namen)
    return config.FEATURE_SPALTEN


def _modell_predict_proba(modell: object, x_features: pd.DataFrame) -> np.ndarray:
    """Ruft predict_proba typrobust auf und liefert ein NumPy-Array zurück."""
    modell_any = cast(Any, modell)
    return np.asarray(modell_any.predict_proba(x_features), dtype=float)


def signal_generieren(
    df: pd.DataFrame,
    modell: object,
    schwelle: float = 0.60,
    short_schwelle: Optional[float] = None,
    decision_mapping: str = "class",
    regime_spalte: str = "market_regime",
    regime_erlaubt: Optional[list] = None,
) -> Tuple[int, float, int, float]:
    """
    Generiert ein Trade-Signal für die letzte abgeschlossene Kerze.

    Args:
        df:              Feature-DataFrame (alle 45 Features vorhanden)
        modell:          Geladenes LightGBM-Modell
        schwelle:        Mindest-Wahrscheinlichkeit für Trade-Ausführung
        short_schwelle:  Optionale Short-Schwelle
        decision_mapping:
            "class"     => Long wenn proba_long >= schwelle, Short wenn proba_short >= short_schwelle
            "long_prob" => Long wenn proba_long >= schwelle, Short wenn proba_long <= short_schwelle
        regime_spalte:   Welche Regime-Spalte genutzt wird
        regime_erlaubt:  Erlaubte Regime-Nummern (None = alle)

    Returns:
        Tuple (signal, prob, regime, atr_pct):
            signal:  2=Long, -1=Short, 0=Kein Trade
            prob:    Wahrscheinlichkeit des Signals (0–1)
            regime:  Aktuelles Markt-Regime (0–3)
            atr_pct: ATR_14 als Prozent vom Close (für ATR-SL Berechnung)
    """
    # Letzte vollständige Kerze (-2: letzte geschlossene Kerze)
    letzte_kerze = df.iloc[[-2]]

    # Aktuelles Regime lesen
    regime_spalte_eff = (
        regime_spalte if regime_spalte in letzte_kerze.columns else "market_regime"
    )
    if regime_spalte_eff != regime_spalte:
        logger.warning(
            f"Regime-Spalte '{regime_spalte}' nicht vorhanden – fallback auf '{regime_spalte_eff}'"
        )
    aktuelles_regime = int(letzte_kerze[regime_spalte_eff].iloc[0])

    # ATR als Prozent vom Close (für dynamisches Stop-Loss)
    atr_pct = 0.0
    if "atr_14" in letzte_kerze.columns and "close" in df.columns:
        atr_abs = float(letzte_kerze["atr_14"].iloc[0])
        close_preis = float(letzte_kerze["close"].iloc[0])
        if close_preis > 0:
            atr_pct = atr_abs / close_preis

    # Regime-Filter prüfen
    if regime_erlaubt is not None and aktuelles_regime not in regime_erlaubt:
        regime_name = config.REGIME_NAMEN.get(aktuelles_regime, "?")
        logger.info(
            f"Signal übersprungen: Regime '{regime_name}' nicht in "
            f"{[config.REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]}"
        )
        return 0, 0.0, aktuelles_regime, atr_pct

    # Features für Modell vorbereiten
    modell_features = _modell_feature_namen(modell)
    verfuegbare = [f for f in modell_features if f in df.columns]
    fehlende = [f for f in modell_features if f not in df.columns]
    if fehlende:
        logger.warning(f"Fehlende Features: {fehlende} – werden mit 0 gefüllt")
        for feat in fehlende:
            letzte_kerze[feat] = 0.0

    x_features = letzte_kerze[verfuegbare].copy()

    if x_features.isna().any().any():
        logger.warning("NaN-Werte in Features – werden mit Median der letzten 50 Kerzen gefüllt")
        nan_fill = df[verfuegbare].iloc[-50:].median()
        x_features = x_features.fillna(nan_fill)

    # Modell-Vorhersage: proba[:,0]=Short, proba[:,1]=Neutral, proba[:,2]=Long
    proba = _modell_predict_proba(modell, x_features)[0]
    raw_pred = int(np.argmax(proba))
    long_prob = float(proba[2])
    short_prob = float(proba[0])
    short_schwelle_eff = (
        float(short_schwelle)
        if short_schwelle is not None
        else float(1.0 - schwelle if decision_mapping == "long_prob" else schwelle)
    )

    logger.info(
        f"Modell-Output: Short={proba[0]:.1%}, Neutral={proba[1]:.1%}, Long={proba[2]:.1%} | "
        f"raw_pred={raw_pred} | Mapping={decision_mapping} | "
        f"Long-Schwelle={schwelle:.1%} | Short-Schwelle={short_schwelle_eff:.1%}"
    )

    if decision_mapping == "long_prob":
        if long_prob >= schwelle:
            logger.info(f"→ Long-Signal ausgelöst (proba_long={long_prob:.1%} >= {schwelle:.1%})")
            return 2, long_prob, aktuelles_regime, atr_pct
        if long_prob <= short_schwelle_eff:
            logger.info(f"→ Short-Signal ausgelöst (proba_long={long_prob:.1%} <= {short_schwelle_eff:.1%})")
            return -1, 1.0 - long_prob, aktuelles_regime, atr_pct
    else:
        if raw_pred == 2 and long_prob >= schwelle:
            logger.info(f"→ Long-Signal ausgelöst (proba_long={long_prob:.1%} >= {schwelle:.1%})")
            return 2, long_prob, aktuelles_regime, atr_pct
        if raw_pred == 0 and short_prob >= short_schwelle_eff:
            logger.info(f"→ Short-Signal ausgelöst (proba_short={short_prob:.1%} >= {short_schwelle_eff:.1%})")
            return -1, short_prob, aktuelles_regime, atr_pct

    logger.info(
        f"→ Kein Signal (raw_pred={raw_pred}, höchste Prob={max(proba):.1%}, Schwelle nicht erfüllt)"
    )
    return 0, float(max(proba)), aktuelles_regime, atr_pct


def shadow_signal_generieren(
    symbol: str,
    df: pd.DataFrame,
    modell: object,
    schwelle: float = 0.60,
    short_schwelle: Optional[float] = None,
    decision_mapping: str = "class",
    regime_spalte: str = "market_regime",
    two_stage_kongruenz: bool = True,
    regime_erlaubt: Optional[list] = None,
    two_stage_config: Optional[dict] = None,
) -> Tuple[int, float, int, float]:
    """
    Shadow-Mode: Routet zwischen Single-Stage und Two-Stage.

    Bei jedem Fehler im Two-Stage-Pfad: Hard Fallback zu Single-Stage.

    Args:
        symbol:           Handelssymbol (USDCAD, USDJPY, ...)
        df:               Feature-DataFrame
        modell:           Single-Stage-Modell (Fallback)
        schwelle:         Wahrscheinlichkeits-Schwelle
        short_schwelle:   Optionale Short-Schwelle
        decision_mapping: "class" oder "long_prob"
        regime_spalte:    Regime-Quelle
        two_stage_kongruenz: True=Kongruenzfilter aktiv, False=aggressiver
        regime_erlaubt:   Erlaubte Regime oder None
        two_stage_config: Two-Stage-Konfiguration (enable, ltf_timeframe, version, htf_df, ltf_df)

    Returns:
        Tuple (signal, prob, regime, atr_pct) – identisch zu signal_generieren()
    """
    # Single-Stage als Baseline (immer berechnen)
    baseline_signal, baseline_prob, baseline_regime, baseline_atr = signal_generieren(
        df=df, modell=modell, schwelle=schwelle, short_schwelle=short_schwelle,
        decision_mapping=decision_mapping, regime_spalte=regime_spalte, regime_erlaubt=regime_erlaubt,
    )

    baseline_prob_label = (
        "proba_long" if baseline_signal == 2
        else "short_score(1-proba_long)" if baseline_signal == -1
        else "score"
    ) if decision_mapping == "long_prob" else "proba_class"

    # Two-Stage nur wenn explizit enabled und Symbol freigegeben
    TWO_STAGE_APPROVED = {"USDCAD", "USDJPY"}
    if not two_stage_config or not two_stage_config.get("enable", False):
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr

    if symbol.upper() not in TWO_STAGE_APPROVED:
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr

    try:
        if not config.TWO_STAGE_VERFUEGBAR:
            logger.warning(f"[{symbol}] Two-Stage-Modul nicht verfügbar – Fallback Single-Stage")
            return baseline_signal, baseline_prob, baseline_regime, baseline_atr

        # Modelle laden (lazy)
        if "htf_model" not in two_stage_config or "ltf_model" not in two_stage_config:
            ltf_tf = two_stage_config.get("ltf_timeframe", "M5")
            version = two_stage_config.get("version", "v4")
            htf_model, ltf_model = config.two_stage_modelle_laden(
                models_dir=config.MODEL_DIR, symbol=symbol, ltf_timeframe=ltf_tf, version=version,
            )
            two_stage_config["htf_model"] = htf_model
            two_stage_config["ltf_model"] = ltf_model
            logger.info(f"[{symbol}] Two-Stage-Modelle geladen: H1 HTF + {ltf_tf} LTF ({version})")

        htf_df = two_stage_config.get("htf_df")
        ltf_df = two_stage_config.get("ltf_df")
        if htf_df is None or ltf_df is None:
            logger.warning(f"[{symbol}] HTF/LTF DataFrames fehlen – Fallback Single-Stage")
            return baseline_signal, baseline_prob, baseline_regime, baseline_atr

        ts_signal = config.zwei_stufen_signal(
            htf_df=htf_df, ltf_df=ltf_df,
            htf_model=two_stage_config["htf_model"], ltf_model=two_stage_config["ltf_model"],
            htf_feature_spalten=two_stage_config.get("htf_features", config.FEATURE_SPALTEN),
            ltf_feature_spalten=two_stage_config.get("ltf_features", config.FEATURE_SPALTEN),
            schwelle=schwelle,
        )

        logger.info(
            f"[{symbol}] TWO-STAGE DEBUG | "
            f"HTF={config.KLASSEN_NAMEN.get(ts_signal.htf_bias_klasse, str(ts_signal.htf_bias_klasse))} "
            f"(S={ts_signal.htf_bias_proba['short']:.1%}, N={ts_signal.htf_bias_proba['neutral']:.1%}, "
            f"L={ts_signal.htf_bias_proba['long']:.1%}) | "
            f"LTF-Signal-vor-Filter={ts_signal.ltf_signal_vor_filter} "
            f"(S={ts_signal.ltf_proba['short']:.1%}, N={ts_signal.ltf_proba['neutral']:.1%}, "
            f"L={ts_signal.ltf_proba['long']:.1%})"
        )

        # Kongruenz-Filter
        htf_bias = ts_signal.htf_bias_klasse
        ltf_signal = ts_signal.signal
        two_stage_config["last_htf_bias"] = htf_bias
        two_stage_config["last_ltf_signal"] = ltf_signal

        allow_neutral_htf = bool(
            two_stage_config.get("allow_neutral_htf_entries", config.STANDARD_TWO_STAGE_ALLOW_NEUTRAL_HTF)
        )
        kongruent = True

        if ltf_signal != 0:
            if htf_bias == 1:
                if not allow_neutral_htf:
                    kongruent = False
            elif htf_bias == 0 and ltf_signal != -1:
                kongruent = False
            elif htf_bias == 2 and ltf_signal != 2:
                kongruent = False

        if two_stage_kongruenz and not kongruent:
            logger.info(
                f"[{symbol}] KONGRUENZ-FILTER | LTF={ltf_signal} BLOCKIERT (HTF={htf_bias}) | "
                f"LTF-Prob={ts_signal.prob:.1%} | Baseline={baseline_signal}"
            )
            return 0, ts_signal.prob, baseline_regime, baseline_atr

        if ts_signal.signal != baseline_signal:
            logger.info(
                f"[{symbol}] SHADOW-DIVERGENZ | Two-Stage={ts_signal.signal} | Baseline={baseline_signal}"
            )
        else:
            logger.info(f"[{symbol}] SHADOW-KONGRUENZ | Signal={ts_signal.signal}")

        return ts_signal.signal, ts_signal.prob, baseline_regime, baseline_atr

    except FileNotFoundError as e:
        logger.warning(f"[{symbol}] Two-Stage-Modelle nicht gefunden: {e} – Fallback Single-Stage")
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"[{symbol}] Two-Stage-Fehler: {e} – Fallback Single-Stage", exc_info=True)
        return baseline_signal, baseline_prob, baseline_regime, baseline_atr
