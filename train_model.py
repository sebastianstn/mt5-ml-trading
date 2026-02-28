"""
train_model.py – XGBoost & LightGBM Training mit Optuna-Optimierung

Trainiert einen 3-Klassen-Klassifikator auf H1 Forex-Daten:
    Klasse 0 = Short-Signal   (war: label -1)
    Klasse 1 = Kein Signal    (war: label  0)
    Klasse 2 = Long-Signal    (war: label +1)

Ablauf:
    1. Daten laden (EURUSD_H1_labeled.csv)
    2. Features auswählen (scale-invariante Indikatoren)
    3. Zeitliche Train/Val/Test-Aufteilung (NIEMALS shuffle!)
    4. XGBoost Baseline trainieren
    5. LightGBM Baseline trainieren
    6. Optuna Hyperparameter-Tuning (50 Trials je Modell)
    7. Bestes Modell evaluieren (F1-Macro auf Validierung)
    8. Modell als .pkl speichern
    9. Feature Importance visualisieren

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python train_model.py [--symbol EURUSD] [--trials 50]

Eingabe:  data/SYMBOL_H1_labeled.csv
Ausgabe:  models/lgbm_SYMBOL_v1.pkl
          models/xgb_SYMBOL_v1.pkl
          plots/SYMBOL_feature_importance.png
          plots/SYMBOL_confusion_matrix.png
"""

# Standard-Bibliotheken
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Datenverarbeitung
import numpy as np
import pandas as pd

# ML-Modelle
import xgboost as xgb
import lightgbm as lgb

# Scikit-learn Hilfsmittel
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

# Hyperparameter-Optimierung
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Nur Warnungen im Log

# Modell speichern
import joblib

# Visualisierung
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_model.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

# Klassen-Beschriftungen (für Berichte und Plots)
KLASSEN_NAMEN = {0: "Short", 1: "Neutral", 2: "Long"}

# ============================================================
# Feature-Liste (scale-invariant, für cross-symbol geeignet)
# ============================================================

# Diese Spalten sind roh/skalenabhängig und werden NICHT als Features genutzt
AUSSCHLUSS_SPALTEN = {
    "open",
    "high",
    "low",
    "close",  # Rohe Preise
    "volume",
    "spread",  # Roh-Volumen und Spread
    "sma_20",
    "sma_50",
    "sma_200",  # Absolute SMA-Level
    "ema_12",
    "ema_26",  # Absolute EMA-Level
    "atr_14",  # ATR in Preis-Einheiten
    "bb_upper",
    "bb_mid",
    "bb_lower",  # Absolute BB-Level
    "obv",  # Kumulativer OBV (nicht normiert)
    "label",  # Zielvariable
}


# ============================================================
# 1. Daten laden
# ============================================================


def labeled_pfad(symbol: str, version: str = "v1", timeframe: str = "H1") -> Path:
    """
    Gibt den Eingabe-Pfad für das gelabelte CSV zurück.

    H1 (Standard):
        v1 → data/SYMBOL_H1_labeled.csv        (Original, rückwärtskompatibel)
        v2 → data/SYMBOL_H1_labeled_v2.csv
        v3 → data/SYMBOL_H1_labeled_v3.csv
    H4:
        v1 → data/SYMBOL_H4_labeled.csv
        v2 → data/SYMBOL_H4_labeled_v2.csv

    Args:
        symbol:    Handelssymbol (z.B. "EURUSD")
        version:   Versions-String (Standard: "v1")
        timeframe: Zeitrahmen der Daten – "H1" oder "H4" (Standard: "H1")

    Returns:
        Path zum gelabelten CSV
    """
    if timeframe == "H4":
        if version == "v1":
            return DATA_DIR / f"{symbol}_H4_labeled.csv"
        return DATA_DIR / f"{symbol}_H4_labeled_{version}.csv"

    # H1 (Standard, rückwärtskompatibel)
    if version == "v1":
        return DATA_DIR / f"{symbol}_H1_labeled.csv"
    return DATA_DIR / f"{symbol}_H1_labeled_{version}.csv"


def daten_laden(
    symbol: str, version: str = "v1", timeframe: str = "H1"
) -> pd.DataFrame:
    """
    Lädt den gelabelten Feature-DataFrame für ein Symbol.

    Args:
        symbol:    Handelssymbol (z.B. "EURUSD")
        version:   Versions-String für den Datei-Pfad (Standard: "v1")
        timeframe: Zeitrahmen – "H1" oder "H4" (Standard: "H1")

    Returns:
        DataFrame mit Features und 'label'-Spalte.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    pfad = labeled_pfad(symbol, version, timeframe)
    if not pfad.exists():
        hilfe = "h4_pipeline.py" if timeframe == "H4" else f"labeling.py --version {version}"
        raise FileNotFoundError(
            f"Datei nicht gefunden: {pfad}\n"
            f"Zuerst {hilfe} ausführen!"
        )

    logger.info(f"Lade {pfad.name} ...")
    df = pd.read_csv(pfad, index_col="time", parse_dates=True)
    logger.info(f"Geladen: {len(df):,} Kerzen | {len(df.columns)} Spalten")
    logger.info(f"Zeitraum: {df.index[0].date()} bis {df.index[-1].date()}")
    return df


# ============================================================
# 2. Features und Zielvariable aufbereiten
# ============================================================


def features_und_ziel(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Trennt Features (X) und Zielvariable (y) und kodiert Labels.

    Label-Kodierung: -1 → 0 (Short), 0 → 1 (Neutral), 1 → 2 (Long)
    XGBoost und LightGBM benötigen Klassen als 0, 1, 2.

    Args:
        df: Gelabelter DataFrame

    Returns:
        Tuple (X, y) – Features als DataFrame, Labels als Series (0/1/2)
    """
    # Features: alle Spalten außer den Ausschluss-Spalten
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    X = df[feature_spalten].copy()

    # Labels umkodieren: {-1, 0, 1} → {0, 1, 2}
    y = df["label"].map({-1: 0, 0: 1, 1: 2})

    logger.info(f"Features: {len(feature_spalten)} Spalten")
    logger.info(f"Feature-Spalten: {feature_spalten}")

    # NaN-Check (sollte nach feature_engineering bereits bereinigt sein)
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"{nan_count} NaN-Werte in Features – werden mit Median gefüllt")
        X = X.fillna(X.median())

    return X, y


# ============================================================
# 3. Zeitliche Train/Val/Test-Aufteilung
# ============================================================


def daten_aufteilen(
    X: pd.DataFrame,
    y: pd.Series,
    train_bis: str = "2021-12-31",
    val_bis: str = "2022-12-31",
) -> tuple:
    """
    Teilt Daten ZEITLICH auf (niemals shuffle!).

    WICHTIG: Test-Set NUR am Ende des Projekts verwenden!
    Das Test-Set repräsentiert echte, zukünftige Marktbedingungen.

    Aufteilung:
        Training:    2018–2021 (historische Muster lernen)
        Validierung: 2022      (Modell-Selektion & Tuning)
        Test:        2023+     (finale Bewertung, NIE ANFASSEN!)

    Args:
        X: Feature-DataFrame mit DatetimeIndex
        y: Label-Series mit DatetimeIndex
        train_bis: Ende des Trainingszeitraums (inklusiv)
        val_bis: Ende des Validierungszeitraums (inklusiv)

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Zeitstempel-Grenzen
    train_maske = X.index <= train_bis
    val_maske = (X.index > train_bis) & (X.index <= val_bis)
    test_maske = X.index > val_bis

    X_train, y_train = X[train_maske], y[train_maske]
    X_val, y_val = X[val_maske], y[val_maske]
    X_test, y_test = X[test_maske], y[test_maske]

    # Aufteilung protokollieren
    gesamt = len(X)
    for name, X_teil, y_teil in [
        ("Training  ", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test      ", X_test, y_test),
    ]:
        anteil = len(X_teil) / gesamt * 100
        von = X_teil.index[0].date() if len(X_teil) > 0 else "–"
        bis = X_teil.index[-1].date() if len(X_teil) > 0 else "–"
        logger.info(
            f"  {name}: {len(X_teil):6,} Kerzen ({anteil:.0f}%) | " f"{von} bis {bis}"
        )

    logger.info(f"  TEST-SET: NICHT anfassen bis zur finalen Evaluation!")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 4. Klassen-Gewichte (für Ungleichgewicht)
# ============================================================


def gewichte_berechnen(y_train: pd.Series) -> np.ndarray:
    """
    Berechnet Sample-Gewichte um Klassen-Ungleichgewicht auszugleichen.

    Bei unseren Daten ist Klasse 1 (Neutral) mit ~70% dominant.
    'balanced' gewichtet seltene Klassen höher damit das Modell sie
    nicht ignoriert.

    Args:
        y_train: Trainings-Labels (0, 1, 2)

    Returns:
        Sample-Gewicht-Array (ein Wert pro Trainingsbeispiel)
    """
    gewichte = compute_sample_weight(class_weight="balanced", y=y_train)

    # Verteilung protokollieren
    verteilung = y_train.value_counts().sort_index()
    logger.info("Klassen-Verteilung Training:")
    for klasse, anzahl in verteilung.items():
        name = KLASSEN_NAMEN[klasse]
        anteil = anzahl / len(y_train)
        mittl_gewicht = gewichte[y_train == klasse].mean()
        logger.info(
            f"  Klasse {klasse} ({name:7s}): {anzahl:6,} ({anteil:.1%}) "
            f"→ Gewicht: {mittl_gewicht:.2f}"
        )
    return gewichte


# ============================================================
# 5. XGBoost Baseline
# ============================================================


def xgboost_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gewichte: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Trainiert XGBoost mit vernünftigen Basisparametern.

    Args:
        X_train, y_train: Trainingsdaten
        X_val, y_val: Validierungsdaten (für Early Stopping)
        gewichte: Sample-Gewichte für Klassen-Ausgleich

    Returns:
        Trainiertes XGBoost-Modell
    """
    logger.info("Trainiere XGBoost Baseline ...")

    modell = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",  # Schnellster Algorithmus (auch für CPU)
        early_stopping_rounds=30,
        eval_metric="mlogloss",
        verbosity=0,
    )

    modell.fit(
        X_train,
        y_train,
        sample_weight=gewichte,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    beste_runde = modell.best_iteration
    logger.info(f"XGBoost Baseline: beste Runde = {beste_runde}")
    return modell


# ============================================================
# 6. LightGBM Baseline
# ============================================================


def lightgbm_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gewichte: np.ndarray,
) -> lgb.LGBMClassifier:
    """
    Trainiert LightGBM mit vernünftigen Basisparametern.

    LightGBM ist für tabellarische Daten oft schneller und besser als XGBoost.

    Args:
        X_train, y_train: Trainingsdaten
        X_val, y_val: Validierungsdaten (für Early Stopping)
        gewichte: Sample-Gewichte für Klassen-Ausgleich

    Returns:
        Trainiertes LightGBM-Modell
    """
    logger.info("Trainiere LightGBM Baseline ...")

    modell = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )

    modell.fit(
        X_train,
        y_train,
        sample_weight=gewichte,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    logger.info(f"LightGBM Baseline: {modell.best_iteration_} Bäume")
    return modell


# ============================================================
# 7. Optuna Hyperparameter-Tuning – XGBoost
# ============================================================


def optuna_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gewichte: np.ndarray,
    n_trials: int = 50,
) -> xgb.XGBClassifier:
    """
    Optimiert XGBoost Hyperparameter mit Optuna.

    Optimierungsziel: F1-Macro auf dem Validierungs-Set.
    (F1-Macro gewichtet alle Klassen gleich, gut bei Ungleichgewicht)

    Args:
        X_train, y_train: Trainingsdaten
        X_val, y_val: Validierungsdaten
        gewichte: Sample-Gewichte
        n_trials: Anzahl Optuna-Versuche (Standard: 50)

    Returns:
        Bestes XGBoost-Modell
    """
    logger.info(f"Starte Optuna-Tuning XGBoost ({n_trials} Trials) ...")

    def objective(trial: optuna.Trial) -> float:
        """Optuna-Zielfunktion: maximiere F1-Macro auf Validation."""
        params = {
            "objective": "multi:softmax",
            "num_class": 3,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 30,
            # Hyperparameter-Suchraum
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

        modell = xgb.XGBClassifier(**params)
        modell.fit(
            X_train,
            y_train,
            sample_weight=gewichte,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = modell.predict(X_val)
        return f1_score(y_val, y_pred, average="macro")

    # Optuna-Studie erstellen (maximize F1)
    studie = optuna.create_study(
        direction="maximize",
        study_name="xgboost_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    studie.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    beste_params = studie.best_params
    bestes_f1 = studie.best_value
    logger.info(f"XGBoost Optuna: bestes F1-Macro = {bestes_f1:.4f}")
    logger.info(f"Beste Parameter: {beste_params}")

    # Bestes Modell final trainieren
    beste_params.update(
        {
            "objective": "multi:softmax",
            "num_class": 3,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 30,
        }
    )

    bestes_modell = xgb.XGBClassifier(**beste_params)
    bestes_modell.fit(
        X_train,
        y_train,
        sample_weight=gewichte,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return bestes_modell


# ============================================================
# 8. Optuna Hyperparameter-Tuning – LightGBM
# ============================================================


def optuna_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gewichte: np.ndarray,
    n_trials: int = 50,
) -> lgb.LGBMClassifier:
    """
    Optimiert LightGBM Hyperparameter mit Optuna.

    Optimierungsziel: F1-Macro auf dem Validierungs-Set.

    Args:
        X_train, y_train: Trainingsdaten
        X_val, y_val: Validierungsdaten
        gewichte: Sample-Gewichte
        n_trials: Anzahl Optuna-Versuche (Standard: 50)

    Returns:
        Bestes LightGBM-Modell
    """
    logger.info(f"Starte Optuna-Tuning LightGBM ({n_trials} Trials) ...")

    def objective(trial: optuna.Trial) -> float:
        """Optuna-Zielfunktion: maximiere F1-Macro auf Validation."""
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": 1000,  # Wird durch Early Stopping reduziert
            # Hyperparameter-Suchraum
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        modell = lgb.LGBMClassifier(**params)
        modell.fit(
            X_train,
            y_train,
            sample_weight=gewichte,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        y_pred = modell.predict(X_val)
        return f1_score(y_val, y_pred, average="macro")

    # Optuna-Studie erstellen
    studie = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    studie.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    beste_params = studie.best_params
    bestes_f1 = studie.best_value
    logger.info(f"LightGBM Optuna: bestes F1-Macro = {bestes_f1:.4f}")
    logger.info(f"Beste Parameter: {beste_params}")

    # Bestes Modell final trainieren
    beste_params.update(
        {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": 1000,
        }
    )

    bestes_modell = lgb.LGBMClassifier(**beste_params)
    bestes_modell.fit(
        X_train,
        y_train,
        sample_weight=gewichte,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return bestes_modell


# ============================================================
# 9. Modell evaluieren
# ============================================================


def modell_evaluieren(
    modell,
    X: pd.DataFrame,
    y: pd.Series,
    modell_name: str,
    datensatz: str = "Validierung",
) -> float:
    """
    Berechnet und protokolliert Evaluierungsmetriken.

    Args:
        modell: Trainiertes Modell (XGBoost oder LightGBM)
        X: Feature-Matrix
        y: Wahre Labels (0, 1, 2)
        modell_name: Name für Logging (z.B. "XGBoost Optuna")
        datensatz: Name des Datensatzes (z.B. "Validierung")

    Returns:
        F1-Macro-Score
    """
    y_pred = modell.predict(X)

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    logger.info(f"\n{'─' * 50}")
    logger.info(f"{modell_name} – Ergebnisse ({datensatz})")
    logger.info(f"{'─' * 50}")
    logger.info(f"  Accuracy:    {acc:.4f} ({acc * 100:.2f}%)")
    logger.info(f"  F1-Macro:    {f1_macro:.4f}  ← Hauptmetrik")
    logger.info(f"  F1-Weighted: {f1_weighted:.4f}")
    logger.info(
        f"\n{classification_report(y, y_pred, target_names=['Short', 'Neutral', 'Long'])}"
    )

    return f1_macro


# ============================================================
# 10. Schwellenwert-Analyse für Trade-Ausführung
# ============================================================


def schwellenwert_analyse(
    modell,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    modell_name: str,
    schwellenwerte: list = None,
) -> float:
    """
    Analysiert Precision/Recall für Long- und Short-Signale bei verschiedenen
    Wahrscheinlichkeits-Schwellenwerten.

    Im Live-Trading bedeutet das: Das Modell löst nur dann eine Order aus,
    wenn die Wahrscheinlichkeit für Long oder Short über dem Schwellenwert liegt.
    Höherer Schwellenwert → weniger aber zuverlässigere Trades.

    Args:
        modell: Trainiertes Modell mit predict_proba()
        X_val: Validierungs-Features
        y_val: Wahre Labels (0=Short, 1=Neutral, 2=Long)
        modell_name: Modellname (für Logging)
        schwellenwerte: Liste der zu testenden Schwellenwerte (Standard: [0.40..0.65])

    Returns:
        Empfohlener Schwellenwert (höchste Precision bei ausreichend Trades)
    """
    if schwellenwerte is None:
        schwellenwerte = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    # Wahrscheinlichkeiten für alle 3 Klassen berechnen
    # proba hat Form (n_samples, 3): [prob_short, prob_neutral, prob_long]
    proba = modell.predict_proba(X_val)
    prob_short = proba[:, 0]  # Wahrscheinlichkeit für Short (Klasse 0)
    prob_long = proba[:, 2]  # Wahrscheinlichkeit für Long (Klasse 2)

    logger.info(f"\n{'─' * 60}")
    logger.info(f"Schwellenwert-Analyse – {modell_name}")
    logger.info(f"{'─' * 60}")
    logger.info(
        f"{'Schwelle':>9} | {'Long-Prec':>10} {'Long-Rec':>9} {'Long-N':>7} "
        f"| {'Short-Prec':>11} {'Short-Rec':>9} {'Short-N':>7}"
    )
    logger.info(f"{'-' * 75}")

    # Bester Schwellenwert: höchste Precision bei mindestens 100 Trades
    bester_schwellenwert = 0.50  # Fallback
    beste_precision = 0.0

    for sw in schwellenwerte:
        # Long-Analyse: Modell sagt Long UND Wahrscheinlichkeit > sw
        long_maske = prob_long > sw
        n_long = long_maske.sum()

        if n_long > 0:
            # Wie oft liefert Long-Signal tatsächlich ein Long-Label?
            long_richtig = ((y_val == 2) & long_maske).sum()
            long_precision = long_richtig / n_long
            # Wie viele echte Long-Labels werden gefunden?
            n_echte_long = (y_val == 2).sum()
            long_recall = long_richtig / n_echte_long if n_echte_long > 0 else 0.0
        else:
            long_precision = long_recall = 0.0

        # Short-Analyse: Modell sagt Short UND Wahrscheinlichkeit > sw
        short_maske = prob_short > sw
        n_short = short_maske.sum()

        if n_short > 0:
            short_richtig = ((y_val == 0) & short_maske).sum()
            short_precision = short_richtig / n_short
            n_echte_short = (y_val == 0).sum()
            short_recall = short_richtig / n_echte_short if n_echte_short > 0 else 0.0
        else:
            short_precision = short_recall = 0.0

        logger.info(
            f"  {sw:.0%}     | {long_precision:10.1%} {long_recall:9.1%} {n_long:7d} "
            f"| {short_precision:11.1%} {short_recall:9.1%} {n_short:7d}"
        )

        # Bester Schwellenwert: Precision > 50% und mindestens 100 Trades beider Richtungen
        mittlere_precision = (long_precision + short_precision) / 2
        min_trades = min(n_long, n_short)
        if mittlere_precision > beste_precision and min_trades >= 100:
            beste_precision = mittlere_precision
            bester_schwellenwert = sw

    logger.info(f"\n  Empfehlung: Schwellenwert = {bester_schwellenwert:.0%}")
    logger.info(
        f"  (Höchste mittlere Precision bei ≥100 Trades pro Richtung: {beste_precision:.1%})"
    )
    logger.info(
        f"  Hinweis: Höherer Schwellenwert → weniger aber zuverlässigere Trades"
    )

    return bester_schwellenwert


# ============================================================
# 11. Feature Importance visualisieren
# ============================================================


def feature_importance_plotten(
    modell,
    feature_namen: list,
    symbol: str,
    modell_name: str,
    top_n: int = 25,
) -> None:
    """
    Erstellt einen Barplot der wichtigsten Features.

    Zeigt die Top-N Features nach Wichtigkeit (Gain für LightGBM,
    Feature Importance für XGBoost).

    Args:
        modell: Trainiertes Modell
        feature_namen: Liste der Feature-Namen
        symbol: Handelssymbol (für Dateiname)
        modell_name: Modellname (für Titel)
        top_n: Anzahl der angezeigten Features
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature Importance aus dem Modell lesen
    if hasattr(modell, "feature_importances_"):
        importance = modell.feature_importances_
    else:
        logger.warning("Modell hat keine feature_importances_ – überspringe Plot")
        return

    # Nach Wichtigkeit sortieren (absteigend)
    importance_df = (
        pd.DataFrame({"Feature": feature_namen, "Wichtigkeit": importance})
        .sort_values("Wichtigkeit", ascending=False)
        .head(top_n)
    )

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 8))
    farben = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df)))[::-1]

    ax.barh(
        importance_df["Feature"],
        importance_df["Wichtigkeit"],
        color=farben,
        edgecolor="white",
    )
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(
        f"{symbol} – {modell_name}\nTop {top_n} Features nach Wichtigkeit",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_yaxis()  # Wichtigstes Feature oben
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    plt.tight_layout()
    name_safe = modell_name.lower().replace(" ", "_")
    pfad = PLOTS_DIR / f"{symbol}_feature_importance_{name_safe}.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature Importance Plot: {pfad}")


# ============================================================
# 11. Confusion Matrix visualisieren
# ============================================================


def confusion_matrix_plotten(
    modell,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    symbol: str,
    modell_name: str,
) -> None:
    """
    Erstellt eine Confusion Matrix als Heatmap.

    Args:
        modell: Trainiertes Modell
        X_val, y_val: Validierungsdaten
        symbol: Handelssymbol
        modell_name: Modellname (für Dateiname)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = modell.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Short", "Neutral", "Long"],
        yticklabels=["Short", "Neutral", "Long"],
        ax=ax,
    )
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title(
        f"{symbol} – {modell_name}\nConfusion Matrix (Validierung)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    name_safe = modell_name.lower().replace(" ", "_")
    pfad = PLOTS_DIR / f"{symbol}_confusion_matrix_{name_safe}.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion Matrix Plot: {pfad}")


# ============================================================
# 12. Modell speichern
# ============================================================


def modell_speichern(
    modell,
    symbol: str,
    modell_typ: str,
    version: str = "v1",
    timeframe: str = "H1",
) -> Path:
    """
    Speichert das Modell als .pkl mit joblib.

    NIEMALS pickle verwenden – joblib ist sicherer und schneller!

    Dateiname-Schema:
        H1: lgbm_SYMBOL_v1.pkl   (Standard, rückwärtskompatibel)
        H4: lgbm_SYMBOL_H4_v1.pkl

    Args:
        modell:    Trainiertes Modell
        symbol:    Handelssymbol (z.B. "EURUSD")
        modell_typ: "lgbm" oder "xgb"
        version:   Versions-String (z.B. "v1")
        timeframe: "H1" oder "H4" (Standard: "H1")

    Returns:
        Pfad zur gespeicherten Datei
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if timeframe == "H4":
        dateiname = f"{modell_typ}_{symbol.lower()}_H4_{version}.pkl"
    else:
        dateiname = f"{modell_typ}_{symbol.lower()}_{version}.pkl"
    pfad = MODEL_DIR / dateiname

    joblib.dump(modell, pfad)
    groesse_kb = pfad.stat().st_size / 1024
    logger.info(f"Modell gespeichert: {pfad} ({groesse_kb:.0f} KB)")
    return pfad


# ============================================================
# 13. Hauptprogramm
# ============================================================


def symbol_trainieren(
    symbol: str, n_trials: int, version: str, timeframe: str = "H1"
) -> bool:
    """
    Kompletter Training-Ablauf für ein einzelnes Symbol.

    Args:
        symbol:    Handelssymbol (z.B. "EURUSD")
        n_trials:  Anzahl Optuna-Trials
        version:   Versions-String für I/O-Dateien (z.B. "v1", "v2", "v3")
        timeframe: Zeitrahmen der Daten – "H1" oder "H4" (Standard: "H1")

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    logger.info("=" * 60)
    logger.info(
        f"Phase 4 – Modelltraining – {symbol} ({version}, {timeframe})"
    )
    logger.info(f"Optuna Trials: {n_trials}")
    logger.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # ---- Schritt 1: Daten laden ----
        df = daten_laden(symbol, version, timeframe)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # ---- Schritt 2: Features und Ziel ----
    X, y = features_und_ziel(df)
    feature_namen = list(X.columns)

    # ---- Schritt 3: Zeitliche Aufteilung ----
    logger.info("\nZeitliche Datenaufteilung:")
    X_train, X_val, X_test, y_train, y_val, y_test = daten_aufteilen(X, y)

    # ---- Schritt 4: Klassen-Gewichte ----
    gewichte = gewichte_berechnen(y_train)

    # ---- Schritt 5: XGBoost Baseline ----
    logger.info("\n" + "=" * 40)
    logger.info("XGBoost Baseline")
    logger.info("=" * 40)
    xgb_basis = xgboost_baseline(X_train, y_train, X_val, y_val, gewichte)
    f1_xgb_basis = modell_evaluieren(xgb_basis, X_val, y_val, "XGBoost Baseline")

    # ---- Schritt 6: LightGBM Baseline ----
    logger.info("\n" + "=" * 40)
    logger.info("LightGBM Baseline")
    logger.info("=" * 40)
    lgbm_basis = lightgbm_baseline(X_train, y_train, X_val, y_val, gewichte)
    f1_lgbm_basis = modell_evaluieren(lgbm_basis, X_val, y_val, "LightGBM Baseline")

    # ---- Schritt 7: Optuna XGBoost ----
    logger.info("\n" + "=" * 40)
    logger.info(f"Optuna XGBoost ({n_trials} Trials)")
    logger.info("=" * 40)
    xgb_opt = optuna_xgboost(X_train, y_train, X_val, y_val, gewichte, n_trials)
    f1_xgb_opt = modell_evaluieren(xgb_opt, X_val, y_val, "XGBoost Optuna")

    # ---- Schritt 8: Optuna LightGBM ----
    logger.info("\n" + "=" * 40)
    logger.info(f"Optuna LightGBM ({n_trials} Trials)")
    logger.info("=" * 40)
    lgbm_opt = optuna_lightgbm(X_train, y_train, X_val, y_val, gewichte, n_trials)
    f1_lgbm_opt = modell_evaluieren(lgbm_opt, X_val, y_val, "LightGBM Optuna")

    # ---- Schritt 9: Bestes Modell auswählen und speichern ----
    logger.info("\n" + "=" * 50)
    logger.info("ZUSAMMENFASSUNG – F1-Macro (Validierung)")
    logger.info("=" * 50)
    ergebnisse = [
        ("XGBoost Baseline", f1_xgb_basis, xgb_basis, "xgb"),
        ("LightGBM Baseline", f1_lgbm_basis, lgbm_basis, "lgbm"),
        ("XGBoost Optuna", f1_xgb_opt, xgb_opt, "xgb"),
        ("LightGBM Optuna", f1_lgbm_opt, lgbm_opt, "lgbm"),
    ]
    for name, f1, _, _ in sorted(ergebnisse, key=lambda x: -x[1]):
        stern = " ← BESTES MODELL" if f1 == max(e[1] for e in ergebnisse) else ""
        logger.info(f"  {name:22s}: F1-Macro = {f1:.4f}{stern}")

    # Bestes Modell wählen
    bestes_name, bestes_f1, bestes_modell, bestes_typ = max(
        ergebnisse, key=lambda x: x[1]
    )
    logger.info(f"\nBestes Modell: {bestes_name} (F1={bestes_f1:.4f})")

    # Beide Optuna-Modelle mit der richtigen Version + Zeitrahmen speichern
    modell_speichern(xgb_opt, symbol, "xgb", version, timeframe)
    modell_speichern(lgbm_opt, symbol, "lgbm", version, timeframe)

    # ---- Schritt 10: Schwellenwert-Analyse (bestes Modell) ----
    logger.info("\n" + "=" * 50)
    logger.info("Schwellenwert-Analyse – Trade-Ausführung")
    logger.info("=" * 50)
    empfohlener_schwellenwert = schwellenwert_analyse(
        bestes_modell, X_val, y_val, bestes_name
    )

    # ---- Schritt 11: Visualisierungen ----
    logger.info("\nErstelle Visualisierungen ...")
    # Versions-Suffix im Modellnamen damit Plots sich nicht überschreiben
    lgbm_name = f"LightGBM Optuna {version}" if version != "v1" else "LightGBM Optuna"
    xgb_name  = f"XGBoost Optuna {version}"  if version != "v1" else "XGBoost Optuna"
    feature_importance_plotten(lgbm_opt, feature_namen, symbol, lgbm_name)
    feature_importance_plotten(xgb_opt, feature_namen, symbol, xgb_name)
    confusion_matrix_plotten(bestes_modell, X_val, y_val, symbol,
                              f"{bestes_name} {version}" if version != "v1" else bestes_name)

    # ---- Abschluss ----
    ende = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 60)
    print(f"ABGESCHLOSSEN – Training {symbol} ({version}, {timeframe})")
    print("=" * 60)
    print(f"  Bestes Modell:   {bestes_name}")
    print(f"  F1-Macro (Val):  {bestes_f1:.4f} ({bestes_f1 * 100:.2f}%)")
    print(f"\n  Gespeicherte Modelle:")
    if timeframe == "H4":
        print(f"    models/xgb_{symbol.lower()}_H4_{version}.pkl")
        print(f"    models/lgbm_{symbol.lower()}_H4_{version}.pkl")
    else:
        print(f"    models/xgb_{symbol.lower()}_{version}.pkl")
        print(f"    models/lgbm_{symbol.lower()}_{version}.pkl")
    print(f"\n  Empfohlener Schwellenwert: {empfohlener_schwellenwert:.0%}")
    print(f"  (Trades nur wenn Modell-Wahrscheinlichkeit > {empfohlener_schwellenwert:.0%})")
    print(
        f"\n  ACHTUNG: Test-Set ({X_test.index[0].date()} bis {X_test.index[-1].date()})"
    )
    print(f"  ist NICHT bewertet worden – für die finale Evaluation aufheben!")
    print(f"\n  Fertig um: {ende}")
    print("=" * 60)
    return True


def main() -> None:
    """Kompletter Training-Ablauf: Laden → Aufteilen → Trainieren → Speichern."""

    # Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="MT5 ML-Trading – Modelltraining")
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help=(
            "Handelssymbol (Standard: EURUSD) oder 'alle' für alle 7 Forex-Paare. "
            "Mögliche Werte: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD"
        ),
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Optuna Trials (Standard: 50)"
    )
    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Versions-Suffix für Ein- und Ausgabe-Dateien (Standard: v1). "
            "v1 → SYMBOL_H1_labeled.csv + lgbm_SYMBOL_v1.pkl | "
            "v2 → SYMBOL_H1_labeled_v2.csv + lgbm_SYMBOL_v2.pkl | "
            "v3 → SYMBOL_H1_labeled_v3.csv + lgbm_SYMBOL_v3.pkl"
        ),
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "H4"],
        help=(
            "Zeitrahmen der Eingabedaten (Standard: H1). "
            "H1 → SYMBOL_H1_labeled.csv → lgbm_SYMBOL_v1.pkl | "
            "H4 → SYMBOL_H4_labeled.csv → lgbm_SYMBOL_H4_v1.pkl. "
            "Vor H4 zuerst features/h4_pipeline.py ausführen!"
        ),
    )
    args = parser.parse_args()

    # Symbole bestimmen
    SYMBOLE_LISTE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
    if args.symbol.lower() == "alle":
        ziel_symbole = SYMBOLE_LISTE
    elif args.symbol.upper() in SYMBOLE_LISTE:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: {args.symbol}")
        print(f"Verfügbar: {', '.join(SYMBOLE_LISTE)} oder 'alle'")
        return

    gesamt_start = datetime.now()
    gesamt_ergebnisse = []

    for symbol in ziel_symbole:
        print(f"\n{'═' * 60}")
        print(f"  Training: {symbol} ({args.version}, {args.timeframe})")
        print(f"{'═' * 60}")
        erfolg = symbol_trainieren(symbol, args.trials, args.version, args.timeframe)
        gesamt_ergebnisse.append((symbol, "OK" if erfolg else "FEHLER"))

    # Gesamtzusammenfassung (nur bei mehreren Symbolen)
    if len(ziel_symbole) > 1:
        dauer = int((datetime.now() - gesamt_start).total_seconds())
        print("\n" + "=" * 60)
        print(f"ALLE SYMBOLE – Training {args.version} abgeschlossen")
        print("=" * 60)
        for symbol, status in gesamt_ergebnisse:
            zeichen = "✓" if status == "OK" else "✗"
            print(f"  {zeichen} {symbol}: {status}")
        erfolge = sum(1 for _, s in gesamt_ergebnisse if s == "OK")
        print(f"\n{erfolge}/{len(ziel_symbole)} Symbole erfolgreich trainiert")
        print(f"Laufzeit: {dauer // 60}m {dauer % 60}s")
        print(f"\nNächster Schritt: walk_forward.py --symbol alle --version {args.version}")
        print("=" * 60)
    else:
        print(f"\nNächster Schritt: walk_forward.py --version {args.version}")


if __name__ == "__main__":
    main()
