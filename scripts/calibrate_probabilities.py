"""
calibrate_probabilities.py - Kalibriert Klassifikations-Wahrscheinlichkeiten.

Ziel:
    Prueft und kalibriert Modellwahrscheinlichkeiten (Platt/Sigmoid oder Isotonic),
    damit Schwellwerte wie 0.55 robuster mit echter Edge zusammenhaengen.

Laufort:
    Linux-Server

Beispiel:
    python scripts/calibrate_probabilities.py \
      --model_path models/lgbm_usdjpy_v4.pkl \
      --data_csv data/processed/USDJPY_H1_regime_labelled.csv \
      --label_col label \
      --method sigmoid \
      --output_model models/lgbm_usdjpy_v4_calibrated_sigmoid.pkl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

try:
    # sklearn >= 1.6: Ersatz für cv='prefit'.
    from sklearn.frozen import FrozenEstimator

    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False


@dataclass(frozen=True)
class CalibrationConfig:
    """Konfiguration fuer den Kalibrierungs-Run."""

    model_path: Path
    data_csv: Path
    label_col: str
    method: str
    output_model: Path
    report_path: Path
    test_size: float
    random_state: int


def parse_args() -> argparse.Namespace:
    """Parst CLI-Argumente."""
    parser = argparse.ArgumentParser(
        description=(
            "Kalibriert Modell-Wahrscheinlichkeiten mit Platt (sigmoid) oder "
            "Isotonic und schreibt Metrik-Report."
        )
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Pfad zum .pkl-Modell"
    )
    parser.add_argument(
        "--data_csv",
        required=True,
        type=str,
        help="CSV mit Features + Label-Spalte",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name der Label-Spalte (Standard: label)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Kalibrierungsmethode (Platt=sigmoid oder isotonic)",
    )
    parser.add_argument(
        "--output_model",
        required=True,
        type=str,
        help="Pfad fuer kalibriertes Modell (.pkl)",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="reports/threshold_eval/calibration_report.json",
        help="Pfad fuer JSON-Report",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Anteil fuer Holdout-Set (Standard: 0.3)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Zufallsseed fuer reproduzierbares Splitten (Standard: 42)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CalibrationConfig:
    """Baut typsichere Konfiguration."""
    return CalibrationConfig(
        model_path=Path(args.model_path),
        data_csv=Path(args.data_csv),
        label_col=str(args.label_col),
        method=str(args.method),
        output_model=Path(args.output_model),
        report_path=Path(args.report_path),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )


def multiclass_brier_score(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> float:
    """
    Berechnet Brier-Score fuer Multi-Class (mean squared error auf One-Hot).

    Args:
        y_true: Wahre Klassenlabels.
        proba: Vorhergesagte Klassenwahrscheinlichkeiten.
        classes: Klassenreihenfolge von predict_proba.

    Returns:
        Multi-Class Brier-Score (kleiner = besser).
    """
    n_samples = y_true.shape[0]
    one_hot = np.zeros((n_samples, len(classes)), dtype=float)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    for row_idx, label in enumerate(y_true):
        label_int = int(label)
        if label_int in class_to_idx:
            one_hot[row_idx, class_to_idx[label_int]] = 1.0
    return float(np.mean(np.sum((one_hot - proba) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """
    Schaetzt ECE ueber max-Proba und Trefferquote.

    Args:
        y_true: Wahre Labels.
        proba: Klassifikationswahrscheinlichkeiten.
        n_bins: Anzahl Bins.

    Returns:
        ECE-Wert (kleiner = besser).
    """
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        # Letzter Bin inkl. oberer Grenze.
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(accuracies[mask]))
        weight = float(np.mean(mask.astype(float)))
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def load_dataset_for_model(
    model: Any, data_csv: Path, label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Laedt Datensatz und selektiert Feature-Spalten passend zum Modell.

    Args:
        model: Geladenes Klassifikationsmodell.
        data_csv: Eingabe-CSV.
        label_col: Labelspaltenname.

    Returns:
        Tuple (X, y).
    """
    df = pd.read_csv(data_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label-Spalte fehlt: {label_col}")

    if hasattr(model, "feature_name_") and model.feature_name_:
        feature_cols = [col for col in model.feature_name_ if col in df.columns]
    elif hasattr(model, "feature_names_in_"):
        feature_cols = [col for col in model.feature_names_in_ if col in df.columns]
    else:
        # Fallback: alles ausser Label.
        feature_cols = [col for col in df.columns if col != label_col]

    if not feature_cols:
        raise ValueError("Keine passenden Feature-Spalten fuer das Modell gefunden.")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[label_col], errors="coerce")
    valid_mask = y.notna()
    X = X[valid_mask].copy()
    y = y[valid_mask].astype(int)

    # Robustes NaN-Handling fuer Kalibrierungs-Run.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def evaluate_probs(
    y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray
) -> Dict[str, float]:
    """Berechnet Kernmetriken fuer Kalibrierungsqualitaet."""
    return {
        "log_loss": float(log_loss(y_true, probs, labels=list(classes))),
        "brier_multiclass": multiclass_brier_score(y_true, probs, classes),
        "ece": expected_calibration_error(y_true, probs, n_bins=10),
    }


def run_calibration(config: CalibrationConfig) -> Dict[str, Any]:
    """
    Fuehrt Kalibrierung inkl. Vorher/Nachher-Metriken durch.

    Args:
        config: Laufkonfiguration.

    Returns:
        Report-Dict mit Ergebnissen.
    """
    if not config.model_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {config.model_path}")
    if not config.data_csv.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {config.data_csv}")

    model = joblib.load(config.model_path)
    if not hasattr(model, "predict_proba"):
        raise ValueError("Modell unterstuetzt kein predict_proba().")

    X, y = load_dataset_for_model(model, config.data_csv, config.label_col)

    # Label-Kodierung robust an Modellklassen anpassen.
    # Typischer Fall im Projekt: CSV-Labels = {-1,0,1}, Modellklassen = {0,1,2}.
    label_mapping: Dict[int, int] = {}
    if hasattr(model, "classes_"):
        model_classes_sorted = sorted([int(c) for c in model.classes_])
        data_classes_sorted = sorted([int(c) for c in y.unique()])
        if data_classes_sorted != model_classes_sorted:
            if len(data_classes_sorted) != len(model_classes_sorted):
                raise ValueError(
                    "Label-Klassen in Daten und Modell sind inkompatibel: "
                    f"data={data_classes_sorted}, model={model_classes_sorted}"
                )
            label_mapping = {
                old: new for old, new in zip(data_classes_sorted, model_classes_sorted)
            }
            y = y.map(label_mapping).astype(int)

    if len(X) < 100:
        raise ValueError(
            "Zu wenige Daten fuer stabile Kalibrierung (mind. 100 Zeilen empfohlen)."
        )

    # Achtung Zeitreihe: shuffle=False.
    split_idx = int(len(X) * (1.0 - config.test_size))
    split_idx = max(1, min(split_idx, len(X) - 1))
    X_cal, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_cal, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Vorher-Metriken.
    probs_before = model.predict_proba(X_test)
    classes = (
        np.array(model.classes_)
        if hasattr(model, "classes_")
        else np.arange(probs_before.shape[1])
    )
    before = evaluate_probs(y_test.to_numpy(), probs_before, classes)

    # Kalibrierung mit vortrainiertem Modell.
    # sklearn >= 1.6 entfernt cv='prefit' -> FrozenEstimator + cv=None.
    if HAS_FROZEN_ESTIMATOR:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method=config.method, cv=None
        )
    else:
        calibrated = CalibratedClassifierCV(
            estimator=model, method=config.method, cv="prefit"
        )
    calibrated.fit(X_cal, y_cal)
    probs_after = calibrated.predict_proba(X_test)
    after = evaluate_probs(y_test.to_numpy(), probs_after, classes)

    # Artefakte schreiben.
    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, config.output_model)

    report: Dict[str, Any] = {
        "model_path": str(config.model_path),
        "data_csv": str(config.data_csv),
        "label_col": config.label_col,
        "method": config.method,
        "output_model": str(config.output_model),
        "n_rows_total": int(len(X)),
        "n_rows_calibration": int(len(X_cal)),
        "n_rows_test": int(len(X_test)),
        "label_mapping": label_mapping,
        "metrics_before": before,
        "metrics_after": after,
        "delta": {
            "log_loss": round(after["log_loss"] - before["log_loss"], 6),
            "brier_multiclass": round(
                after["brier_multiclass"] - before["brier_multiclass"], 6
            ),
            "ece": round(after["ece"] - before["ece"], 6),
        },
    }

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """CLI-Einstiegspunkt."""
    try:
        args = parse_args()
        config = build_config(args)
        report = run_calibration(config)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[FEHLER] {exc}")
        return 1

    print("Kalibrierung abgeschlossen.")
    print(f"Output-Modell: {report['output_model']}")
    print(f"Report: {config.report_path}")
    print(
        "Metriken: "
        f"logloss {report['metrics_before']['log_loss']:.5f} -> {report['metrics_after']['log_loss']:.5f}, "
        f"ECE {report['metrics_before']['ece']:.5f} -> {report['metrics_after']['ece']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
