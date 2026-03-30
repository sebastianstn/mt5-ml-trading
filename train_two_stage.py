"""
train_two_stage.py – Zwei-Stufen-Modell (Option 1) für HTF/LTF

Architektur:
    Stufe 1 (HTF): Bias-Klassifikator auf H1
        Input: H1-Features (inkl. H4/D1 abgeleitete Features)
        Output: Bias-Klassen {Short, Neutral, Long} + Wahrscheinlichkeiten

    Stufe 2 (LTF): Entry-Timing auf M15/M5
        Input: LTF-Features + HTF-Bias-Features aus Stufe 1
        Output: Signal-Klassen {Short, Neutral, Long} + Wahrscheinlichkeiten

WICHTIG – Look-Ahead-Bias Prävention:
    - HTF-Bias wird vor der Projektion auf LTF um 1 H1-Kerze verzögert (.shift(1)).
    - Dadurch sieht eine LTF-Kerze niemals den Bias derselben (noch laufenden) H1-Kerze.

Läuft auf:
    Linux-Server

Verwendung:
    cd /mnt/1Tb-Data/XGBoost-LightGBM
    source .venv/bin/activate
    python train_two_stage.py --symbol USDCAD --ltf_timeframe M5 --version v4
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train_two_stage.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


AUSSCHLUSS_SPALTEN: set[str] = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "atr_14",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "obv",
    "pdh",
    "pdl",
    "pwh",
    "pwl",
    "label",
}


@dataclass
class SplitData:
    """Container für zeitlich aufgeteilte Daten."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def labeled_pfad(symbol: str, timeframe: str, version: str) -> Path:
    """Gibt den Pfad zur gelabelten Datei zurück.

    Args:
        symbol: Handelssymbol, z.B. "USDCAD".
        timeframe: Zeitrahmen, z.B. "H1" oder "M5".
        version: Versionssuffix, z.B. "v4".

    Returns:
        Vollständiger Dateipfad zur gelabelten CSV.
    """
    if version == "v1":
        return DATA_DIR / f"{symbol}_{timeframe}_labeled.csv"
    return DATA_DIR / f"{symbol}_{timeframe}_labeled_{version}.csv"


def daten_laden(symbol: str, timeframe: str, version: str) -> pd.DataFrame:
    """Lädt gelabelte Daten für Symbol/Zeitrahmen.

    Args:
        symbol: Handelssymbol.
        timeframe: Zeitrahmen (H1, M15, M30, M5 ...).
        version: Versionssuffix.

    Returns:
        Geladener DataFrame mit DatetimeIndex.

    Raises:
        FileNotFoundError: Wenn Datei fehlt.
    """
    pfad = labeled_pfad(symbol=symbol, timeframe=timeframe, version=version)
    if not pfad.exists():
        raise FileNotFoundError(
            f"Datei fehlt: {pfad}. Bitte zuerst feature_engineering + labeling ausführen."
        )

    df = pd.read_csv(pfad, index_col="time", parse_dates=True)
    df.sort_index(inplace=True)
    logger.info(
        "Geladen %s: %s Zeilen (%s bis %s)",
        pfad.name,
        f"{len(df):,}",
        df.index[0],
        df.index[-1],
    )
    return df


def labels_map(df: pd.DataFrame) -> pd.Series:
    """Mappt Label von {-1,0,1} auf {0,1,2}.

    Args:
        df: DataFrame mit Spalte `label`.

    Returns:
        Gemappte Labels als Integer-Serie.

    Raises:
        ValueError: Bei ungültigen Labelwerten.
    """
    y = df["label"].map({-1: 0, 0: 1, 1: 2})
    if y.isna().any():
        ungueltig = sorted(df.loc[y.isna(), "label"].unique().tolist())
        raise ValueError(
            "Ungültige Labelwerte gefunden. Erwartet nur -1,0,1. "
            f"Gefunden: {ungueltig}"
        )
    return y.astype(int)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert Feature-Matrix ohne Rohpreis-/Label-Spalten.

    Args:
        df: Input-DataFrame.

    Returns:
        Feature-DataFrame.
    """
    spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    x = df[spalten].copy()
    if x.isna().any().any():
        x = x.fillna(x.median())
    return x


def zeitlich_splitten(
    x: pd.DataFrame,
    y: pd.Series,
    train_bis: str = "2021-12-31",
    val_bis: str = "2022-12-31",
) -> SplitData:
    """Zeitliche Aufteilung ohne Shuffle.

    Args:
        x: Feature-Matrix mit DatetimeIndex.
        y: Zielvariable mit identischem Index.
        train_bis: Ende Trainingszeitraum.
        val_bis: Ende Validierungszeitraum.

    Returns:
        SplitData mit Train/Val/Test.
    """
    train_cut = pd.Timestamp(train_bis)
    val_cut = pd.Timestamp(val_bis)
    if getattr(x.index, "tz", None) is not None:
        train_cut = train_cut.tz_localize(x.index.tz)
        val_cut = val_cut.tz_localize(x.index.tz)

    train_mask = x.index <= train_cut
    val_mask = (x.index > train_cut) & (x.index <= val_cut)
    test_mask = x.index > val_cut

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    # Fallback für jüngere Datensätze ohne alte Historie.
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        n = len(x)
        train_end = max(1, int(n * 0.70))
        val_end = min(max(train_end + 1, int(n * 0.85)), n - 1)
        x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        x_val, y_val = x.iloc[train_end:val_end], y.iloc[train_end:val_end]
        x_test, y_test = x.iloc[val_end:], y.iloc[val_end:]
        logger.warning("Fallback-Split 70/15/15 aktiv (chronologisch, ohne Shuffle).")

    return SplitData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def gewichte(y_train: pd.Series) -> np.ndarray:
    """Berechnet Klassengewichte für 3-Klassen-Problem.

    Args:
        y_train: Trainingslabels als 0/1/2.

    Returns:
        Sample-Gewichte.
    """
    class_weights = {0: 3.0, 1: 1.0, 2: 3.0}
    return compute_sample_weight(class_weight=class_weights, y=y_train.astype(int))


def lgbm_trainieren(
    split: SplitData,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Trainiert ein LightGBM-Multiclass-Modell.

    Args:
        split: Zeitlich gesplittete Daten.
        random_state: Seed.

    Returns:
        Trainiertes Modell.
    """
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )

    sample_w = gewichte(split.y_train)
    model.fit(
        split.x_train,
        split.y_train,
        sample_weight=sample_w,
        eval_set=[(split.x_val, split.y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def evaluieren(name: str, model: lgb.LGBMClassifier, split: SplitData) -> None:
    """Loggt F1-Macro für Val/Test und Klassifikationsbericht.

    Args:
        name: Anzeigename des Modells.
        model: Trainiertes Modell.
        split: Zeitlich gesplittete Daten.
    """
    pred_val = model.predict(split.x_val)
    pred_test = model.predict(split.x_test)
    f1_val = f1_score(split.y_val, pred_val, average="macro")
    f1_test = f1_score(split.y_test, pred_test, average="macro")

    logger.info("%s | F1-Macro Val: %.4f | Test: %.4f", name, f1_val, f1_test)
    logger.info(
        "%s | Report (Val):\n%s",
        name,
        classification_report(
            split.y_val,
            pred_val,
            target_names=["Short", "Neutral", "Long"],
        ),
    )


def htf_bias_features_ableiten(
    h1_df: pd.DataFrame,
    htf_model: lgb.LGBMClassifier,
) -> pd.DataFrame:
    """Erzeugt HTF-Bias-Features aus dem H1-Modell (leakage-sicher).

    Args:
        h1_df: H1-DataFrame mit Features + Label.
        htf_model: Trainiertes HTF-Modell.

    Returns:
        DataFrame mit Bias-Klasse und Bias-Wahrscheinlichkeiten auf H1-Index,
        bereits um 1 H1-Kerze verzögert (.shift(1)).
    """
    x_h1 = feature_matrix(h1_df)
    proba = htf_model.predict_proba(x_h1)
    pred_class = np.argmax(proba, axis=1)

    htf_bias = pd.DataFrame(
        {
            "htf_bias_class": pred_class.astype(int),
            "htf_bias_prob_short": proba[:, 0],
            "htf_bias_prob_neutral": proba[:, 1],
            "htf_bias_prob_long": proba[:, 2],
        },
        index=h1_df.index,
    )

    # Kritisch gegen Look-Ahead: LTF bekommt nur abgeschlossenen H1-Bias.
    htf_bias = htf_bias.shift(1)
    return htf_bias


def htf_bias_auf_ltf_projizieren(
    ltf_df: pd.DataFrame,
    htf_bias_df: pd.DataFrame,
) -> pd.DataFrame:
    """Projiziert verzögerten H1-Bias auf LTF-Zeitstempel.

    Args:
        ltf_df: LTF-DataFrame (M15/M5) mit DatetimeIndex.
        htf_bias_df: H1-Bias-Features (bereits .shift(1)).

    Returns:
        LTF-DataFrame mit HTF-Bias-Spalten.
    """
    ltf_reset = ltf_df.reset_index().rename(columns={"time": "timestamp"})
    htf_reset = htf_bias_df.reset_index().rename(columns={"time": "timestamp"})

    # merge_asof nimmt jeweils den letzten bekannten HTF-Wert <= LTF-Zeitpunkt.
    merged = pd.merge_asof(
        ltf_reset.sort_values("timestamp"),
        htf_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp")
    merged.index.name = "time"
    return merged


def ltf_training_set_erstellen(
    ltf_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Erstellt X/y für LTF-Training inklusive HTF-Bias-Spalten.

    Args:
        ltf_df: LTF-DataFrame nach HTF-Projektion.

    Returns:
        Tuple aus (X, y, feature_namen).
    """
    erwartete = {
        "htf_bias_class",
        "htf_bias_prob_short",
        "htf_bias_prob_neutral",
        "htf_bias_prob_long",
    }
    fehlend = erwartete - set(ltf_df.columns)
    if fehlend:
        raise ValueError(f"Fehlende HTF-Bias-Spalten in LTF-Daten: {sorted(fehlend)}")

    # Zeilen mit noch nicht verfügbarem H1-Bias entfernen (Startphase).
    ltf_clean = ltf_df.dropna(subset=list(erwartete)).copy()
    y = labels_map(ltf_clean)

    x = feature_matrix(ltf_clean)
    # Sicherstellen, dass Bias-Klasse als Integer geführt wird.
    x["htf_bias_class"] = x["htf_bias_class"].astype(int)

    return x, y, list(x.columns)


def artefakte_speichern(
    symbol: str,
    ltf_timeframe: str,
    version: str,
    htf_model: lgb.LGBMClassifier,
    ltf_model: lgb.LGBMClassifier,
    htf_features: list[str],
    ltf_features: list[str],
) -> None:
    """Speichert Modelle und Metadaten.

    Args:
        symbol: Handelssymbol.
        ltf_timeframe: LTF-Zeitrahmen.
        version: Versionssuffix.
        htf_model: Trainiertes HTF-Modell.
        ltf_model: Trainiertes LTF-Modell.
        htf_features: Featureliste Stufe 1.
        ltf_features: Featureliste Stufe 2.
    """
    htf_model_path = MODEL_DIR / f"lgbm_htf_bias_{symbol.lower()}_H1_{version}.pkl"
    ltf_model_path = (
        MODEL_DIR / f"lgbm_ltf_entry_{symbol.lower()}_{ltf_timeframe}_{version}.pkl"
    )
    meta_path = MODEL_DIR / f"two_stage_{symbol.lower()}_{ltf_timeframe}_{version}.json"

    joblib.dump(htf_model, htf_model_path)
    joblib.dump(ltf_model, ltf_model_path)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "htf_timeframe": "H1",
        "ltf_timeframe": ltf_timeframe,
        "version": version,
        "htf_model": htf_model_path.name,
        "ltf_model": ltf_model_path.name,
        "htf_features": htf_features,
        "ltf_features": ltf_features,
        "notes": "Option 1 Zwei-Stufen-Modell (HTF Bias -> LTF Entry)",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Gespeichert: %s", htf_model_path)
    logger.info("Gespeichert: %s", ltf_model_path)
    logger.info("Gespeichert: %s", meta_path)


def train_two_stage(symbol: str, ltf_timeframe: str, version: str) -> None:
    """Trainiert das Zwei-Stufen-Modell für ein Symbol.

    Args:
        symbol: Handelssymbol, z.B. "USDCAD".
        ltf_timeframe: LTF-Zeitrahmen, z.B. "M15" oder "M5".
        version: Versionssuffix, z.B. "v1".
    """
    logger.info("=" * 70)
    logger.info(
        "Zwei-Stufen-Training startet | Symbol=%s | HTF=H1 | LTF=%s | Version=%s",
        symbol,
        ltf_timeframe,
        version,
    )
    logger.info("=" * 70)

    # 1) HTF-Daten laden und HTF-Modell trainieren.
    h1_df = daten_laden(symbol=symbol, timeframe="H1", version=version)
    x_h1 = feature_matrix(h1_df)
    y_h1 = labels_map(h1_df)
    split_h1 = zeitlich_splitten(x_h1, y_h1)

    htf_model = lgbm_trainieren(split=split_h1)
    evaluieren(name="HTF-Bias (H1)", model=htf_model, split=split_h1)

    # 2) HTF-Bias-Features ableiten und auf LTF projizieren.
    htf_bias_h1 = htf_bias_features_ableiten(h1_df=h1_df, htf_model=htf_model)

    ltf_df = daten_laden(symbol=symbol, timeframe=ltf_timeframe, version=version)
    ltf_df_mit_bias = htf_bias_auf_ltf_projizieren(
        ltf_df=ltf_df, htf_bias_df=htf_bias_h1
    )

    # 3) LTF-Trainingsset erstellen und LTF-Modell trainieren.
    x_ltf, y_ltf, ltf_features = ltf_training_set_erstellen(ltf_df=ltf_df_mit_bias)
    split_ltf = zeitlich_splitten(x_ltf, y_ltf)
    ltf_model = lgbm_trainieren(split=split_ltf)
    evaluieren(name=f"LTF-Entry ({ltf_timeframe})", model=ltf_model, split=split_ltf)

    # 4) Artefakte speichern.
    artefakte_speichern(
        symbol=symbol,
        ltf_timeframe=ltf_timeframe,
        version=version,
        htf_model=htf_model,
        ltf_model=ltf_model,
        htf_features=list(x_h1.columns),
        ltf_features=ltf_features,
    )

    logger.info("Zwei-Stufen-Training abgeschlossen: %s (%s)", symbol, ltf_timeframe)


def main() -> None:
    """CLI-Einstiegspunkt für Zwei-Stufen-Training."""
    parser = argparse.ArgumentParser(
        description="Option 1: Zwei-Stufen-Modell trainieren"
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Handelssymbol, z.B. USDCAD oder USDJPY",
    )
    parser.add_argument(
        "--ltf_timeframe",
        default="M5",
        choices=["M15", "M30", "M60", "M5"],
        help="LTF-Zeitrahmen für Entry-Modell (Standard: M5)",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Versionssuffix der gelabelten Daten/Modelle (Standard: v4)",
    )
    args = parser.parse_args()

    train_two_stage(
        symbol=args.symbol.upper(),
        ltf_timeframe=args.ltf_timeframe.upper(),
        version=args.version,
    )


if __name__ == "__main__":
    main()
