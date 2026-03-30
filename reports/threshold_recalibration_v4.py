"""
threshold_recalibration_v4.py – M2 Schwellen-Neukalibrierung für v4-Modelle.

Zweck:
    Evaluiert auf dem Validierungsfenster (2022) verschiedene Schwellen
    für beide Mapping-Modi (class / long_prob) und schreibt eine
    reproduzierbare Empfehlung pro Symbol.

Läuft auf:
    Linux-Server (Training/Analyse)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "reports" / "threshold_eval"

SYMBOLE: List[str] = ["USDCAD", "USDJPY"]
VERSION = "v4"
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2022-12-31")

AUSSCHLUSS_SPALTEN = {
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
    "label",
}


def _data_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}_H1_labeled_{VERSION}.csv"


def _model_path(symbol: str) -> Path:
    return MODEL_DIR / f"lgbm_{symbol.lower()}_{VERSION}.pkl"


def _prepare(symbol: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(_data_path(symbol), index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]

    val_mask = (df.index > TRAIN_END) & (df.index <= VAL_END)
    val_df = df.loc[val_mask].copy()

    x = val_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)

    # Label-Mapping wie im Training: -1->0, 0->1, 1->2
    y = val_df["label"].map({-1: 0, 0: 1, 1: 2}).astype(int).to_numpy()
    return x, y


def _predict_class_mapping(proba: np.ndarray, thr: float) -> np.ndarray:
    pred = np.full(proba.shape[0], 1, dtype=int)  # default Neutral
    pred[(proba[:, 2] >= thr) & (proba[:, 2] >= proba[:, 0])] = 2
    pred[(proba[:, 0] >= thr) & (proba[:, 0] > proba[:, 2])] = 0
    return pred


def _predict_long_prob_mapping(
    proba: np.ndarray, long_thr: float, short_thr: float
) -> np.ndarray:
    pred = np.full(proba.shape[0], 1, dtype=int)
    pred[proba[:, 2] >= long_thr] = 2
    pred[proba[:, 2] <= short_thr] = 0
    return pred


def _trade_count(pred: np.ndarray) -> int:
    return int(np.sum((pred == 0) | (pred == 2)))


def evaluate_symbol(symbol: str) -> pd.DataFrame:
    logger.info("[%s] Lade Daten + Modell", symbol)
    x_val, y_val = _prepare(symbol)
    model = joblib.load(_model_path(symbol))
    proba = np.asarray(model.predict_proba(x_val), dtype=float)

    rows: List[Dict[str, float | str | int]] = []
    grid = np.round(np.arange(0.35, 0.66, 0.01), 2)

    for thr in grid:
        pred_class = _predict_class_mapping(proba, float(thr))
        rows.append(
            {
                "symbol": symbol,
                "mapping": "class",
                "long_thr": float(thr),
                "short_thr": float(thr),
                "f1_macro": float(f1_score(y_val, pred_class, average="macro")),
                "trades": _trade_count(pred_class),
            }
        )

    for long_thr in grid:
        for short_thr in grid:
            if short_thr > long_thr:
                continue
            pred_lp = _predict_long_prob_mapping(
                proba, float(long_thr), float(short_thr)
            )
            rows.append(
                {
                    "symbol": symbol,
                    "mapping": "long_prob",
                    "long_thr": float(long_thr),
                    "short_thr": float(short_thr),
                    "f1_macro": float(f1_score(y_val, pred_lp, average="macro")),
                    "trades": _trade_count(pred_lp),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[pd.DataFrame] = []

    for symbol in SYMBOLE:
        result = evaluate_symbol(symbol)
        all_results.append(result)
        result.to_csv(
            OUT_DIR / f"recalibration_{symbol.lower()}_{VERSION}.csv", index=False
        )

    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(OUT_DIR / f"recalibration_all_{VERSION}.csv", index=False)

    # Empfehlung: max F1, bei Gleichstand mehr Trades
    best = (
        full.sort_values(
            ["symbol", "f1_macro", "trades"], ascending=[True, False, False]
        )
        .groupby("symbol", as_index=False)
        .first()
    )
    best.to_csv(OUT_DIR / f"recalibration_recommendation_{VERSION}.csv", index=False)

    logger.info(
        "Fertig. Empfehlung gespeichert: %s",
        OUT_DIR / f"recalibration_recommendation_{VERSION}.csv",
    )


if __name__ == "__main__":
    main()
