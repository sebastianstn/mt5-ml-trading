"""
walk_forward.py – Walk-Forward-Analyse für das ML-Trading-System

Überprüft ob das Modell zeitlich STABIL ist: Lernt es wirklich Muster,
oder ist der Trainingserfolg nur auf einen bestimmten Zeitraum beschränkt?

Methode: Expanding Window (wachsendes Trainingsfenster)
    - Training beginnt immer 2018-04
    - Test-Block schiebt sich alle 6 Monate vorwärts
    - 5 Fenster insgesamt (2019-Q4 bis 2022-Q1)
    - Heiliges Test-Set (2023+) wird NICHT berührt!

Fenster:
    1. Train: 2018-04 – 2019-09  |  Test: 2019-10 – 2020-03  (~18 Monate Train)
    2. Train: 2018-04 – 2020-03  |  Test: 2020-04 – 2020-09  (~24 Monate Train)
    3. Train: 2018-04 – 2020-09  |  Test: 2020-10 – 2021-03  (~30 Monate Train)
    4. Train: 2018-04 – 2021-03  |  Test: 2021-04 – 2021-09  (~36 Monate Train)
    5. Train: 2018-04 – 2021-09  |  Test: 2021-10 – 2022-03  (~42 Monate Train)

Bewertung: LightGBM Baseline (kein Optuna – für 5 Fenster zu aufwendig)
Metrik: F1-Macro (gewichtet alle Klassen gleich, robust bei Ungleichgewicht)

Stabilitätskriterium:
    ✅ Stabil:   Kein Fenster > 0.10 unter dem Durchschnitt
    ❌ Instabil: Ein Fenster deutlich schwächer → Overfitting oder Regime-Shift

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python walk_forward.py [--symbol EURUSD]
    python walk_forward.py --symbol alle

Eingabe:  data/SYMBOL_H1_labeled.csv
Ausgabe:  plots/SYMBOL_walk_forward.png
          walk_forward.log
"""

# Standard-Bibliotheken
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Datenverarbeitung
import numpy as np
import pandas as pd

# ML
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Visualisierung
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("walk_forward.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# Gleiche Ausschluss-Spalten wie in train_model.py (scale-abhängige Features)
AUSSCHLUSS_SPALTEN = {
    "open",
    "high",
    "low",
    "close",  # Rohe Preise – skalenabhängig
    "volume",
    "spread",  # Rohes Volumen und Spread
    "sma_20",
    "sma_50",
    "sma_200",  # Absolute SMA-Level
    "ema_12",
    "ema_26",  # Absolute EMA-Level
    "atr_14",  # ATR in Preis-Einheiten (nicht normiert)
    "bb_upper",
    "bb_mid",
    "bb_lower",  # Absolute Bollinger-Bänder
    "obv",  # Kumulativer OBV (nicht normiert)
    "label",  # Zielvariable – darf kein Feature sein!
}

# ============================================================
# Walk-Forward-Fenster (Expanding Window)
# ============================================================

# WICHTIG: Alle Test-Blöcke enden VOR 2023 → heiliges Test-Set bleibt unberührt!
FENSTER = [
    {
        "name": "Fenster 1",
        "train_von": "2018-04-01",
        "train_bis": "2019-09-30",
        "test_von": "2019-10-01",
        "test_bis": "2020-03-31",
    },
    {
        "name": "Fenster 2",
        "train_von": "2018-04-01",
        "train_bis": "2020-03-31",
        "test_von": "2020-04-01",
        "test_bis": "2020-09-30",
    },
    {
        "name": "Fenster 3",
        "train_von": "2018-04-01",
        "train_bis": "2020-09-30",
        "test_von": "2020-10-01",
        "test_bis": "2021-03-31",
    },
    {
        "name": "Fenster 4",
        "train_von": "2018-04-01",
        "train_bis": "2021-03-31",
        "test_von": "2021-04-01",
        "test_bis": "2021-09-30",
    },
    {
        "name": "Fenster 5",
        "train_von": "2018-04-01",
        "train_bis": "2021-09-30",
        "test_von": "2021-10-01",
        "test_bis": "2022-03-31",
    },
]

# LightGBM-Parameter (aus Baseline, bewährt für alle Symbole)
# Kein Optuna-Tuning pro Fenster – das wäre 50 Trials × 5 Fenster = zu langsam!
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}

# Stabilitätsschwelle: F1 darf maximal um diesen Wert unter dem Durchschnitt liegen
STABILITAETS_SCHWELLE = 0.10


# ============================================================
# 1. Daten laden
# ============================================================


def labeled_pfad(symbol: str, version: str = "v1") -> Path:
    """
    Gibt den Pfad zum gelabelten CSV zurück (konsistent mit labeling.py + train_model.py).

    v1 → data/SYMBOL_H1_labeled.csv        (Original, rückwärtskompatibel)
    v2 → data/SYMBOL_H1_labeled_v2.csv
    v3 → data/SYMBOL_H1_labeled_v3.csv

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versions-String (Standard: "v1")

    Returns:
        Path zum gelabelten CSV
    """
    if version == "v1":
        return DATA_DIR / f"{symbol}_H1_labeled.csv"
    return DATA_DIR / f"{symbol}_H1_labeled_{version}.csv"


def daten_laden(symbol: str, version: str = "v1") -> pd.DataFrame:
    """
    Lädt den gelabelten Feature-DataFrame für ein Symbol.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versions-String für den Datei-Pfad (Standard: "v1")

    Returns:
        DataFrame mit Features und 'label'-Spalte.

    Raises:
        FileNotFoundError: Wenn die gelabelte CSV nicht existiert.
    """
    pfad = labeled_pfad(symbol, version)
    if not pfad.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {pfad}\n"
            f"Zuerst labeling.py --version {version} ausführen!"
        )

    logger.info(f"[{symbol}] Lade {pfad.name} ...")
    df = pd.read_csv(pfad, index_col="time", parse_dates=True)
    logger.info(
        f"[{symbol}] {len(df):,} Kerzen | "
        f"{df.index[0].date()} bis {df.index[-1].date()}"
    )
    return df


# ============================================================
# 2. Features aufbereiten
# ============================================================


def features_aufbereiten(df: pd.DataFrame) -> tuple:
    """
    Trennt Features (X) von Zielvariable (y) und kodiert Labels.

    Label-Kodierung: -1 → 0 (Short), 0 → 1 (Neutral), 1 → 2 (Long)
    XGBoost und LightGBM benötigen Klassen als 0, 1, 2.

    Args:
        df: Gelabelter DataFrame mit 'label'-Spalte

    Returns:
        Tuple (X, y) – Features als DataFrame, Labels als Series (0/1/2)
    """
    # Features: alle Spalten außer den Ausschluss-Spalten
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    X = df[feature_spalten].copy()

    # NaN-Werte auffüllen (sollten nach feature_engineering nicht vorhanden sein)
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"{nan_count} NaN-Werte gefunden – werden mit Median gefüllt")
        X = X.fillna(X.median())

    # Label umkodieren: {-1, 0, 1} → {0, 1, 2}
    y = df["label"].map({-1: 0, 0: 1, 1: 2})

    return X, y


# ============================================================
# 3. Ein Fenster trainieren und evaluieren
# ============================================================


def fenster_trainieren(
    X: pd.DataFrame,
    y: pd.Series,
    fenster: Dict,
) -> Dict:
    """
    Trainiert LightGBM auf einem Walk-Forward-Fenster und evaluiert den Test-Block.

    Early Stopping nutzt die letzten 20% des Trainings-Fensters als internen
    Validierungssplit – der Test-Block bleibt vollständig unsichtbar.

    Args:
        X: Alle Features mit DatetimeIndex
        y: Alle Labels mit DatetimeIndex (kodiert als 0/1/2)
        fenster: Dict mit train_von, train_bis, test_von, test_bis

    Returns:
        Dict mit Ergebnis: name, f1_macro, accuracy, n_train, n_test, bericht
    """
    name = fenster["name"]
    logger.info(
        f"\n  {name}: Train {fenster['train_von'][:7]}–{fenster['train_bis'][:7]}"
        f"  |  Test {fenster['test_von'][:7]}–{fenster['test_bis'][:7]}"
    )

    # Zeitliche Aufteilung (kein shuffle! – Zeitreihen-Regel)
    train_maske = (X.index >= fenster["train_von"]) & (X.index <= fenster["train_bis"])
    test_maske = (X.index >= fenster["test_von"]) & (X.index <= fenster["test_bis"])

    X_train, y_train = X[train_maske], y[train_maske]
    X_test, y_test = X[test_maske], y[test_maske]

    logger.info(
        f"    Trainingsdaten: {len(X_train):,} Kerzen | "
        f"Testdaten: {len(X_test):,} Kerzen"
    )

    # Sicherheitscheck: Genug Daten für Training und Test?
    if len(X_train) < 500 or len(X_test) < 100:
        logger.warning("    Zu wenige Daten – Fenster wird übersprungen!")
        return {
            "name": name,
            "f1_macro": np.nan,
            "accuracy": np.nan,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "bericht": "",
        }

    # Klassen-Gewichte (gleiche Methode wie train_model.py)
    gewichte = compute_sample_weight(class_weight="balanced", y=y_train)

    # Interner Val-Split für Early Stopping: letzte 20% des Trainingsfensters
    # WICHTIG: Zeitlich trennen, kein shuffle!
    val_groesse = max(int(len(X_train) * 0.2), 200)  # Mindestens 200 Kerzen
    X_fit = X_train.iloc[:-val_groesse]
    y_fit = y_train.iloc[:-val_groesse]
    gewichte_fit = gewichte[:-val_groesse]
    X_val_fit = X_train.iloc[-val_groesse:]
    y_val_fit = y_train.iloc[-val_groesse:]

    # LightGBM trainieren
    modell = lgb.LGBMClassifier(**LGBM_PARAMS)
    modell.fit(
        X_fit,
        y_fit,
        sample_weight=gewichte_fit,
        eval_set=[(X_val_fit, y_val_fit)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    # Auf dem Test-Block evaluieren (der NICHT im Training war)
    y_pred = modell.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = float((y_test == y_pred).mean())

    bericht = classification_report(
        y_test,
        y_pred,
        target_names=["Short", "Neutral", "Long"],
        zero_division=0,
    )

    logger.info(
        f"    F1-Macro: {f1_macro:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Bäume: {modell.best_iteration_}"
    )

    return {
        "name": name,
        "train_von": fenster["train_von"][:7],
        "train_bis": fenster["train_bis"][:7],
        "test_von": fenster["test_von"][:7],
        "test_bis": fenster["test_bis"][:7],
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "bericht": bericht,
    }


# ============================================================
# 4. Stabilität bewerten
# ============================================================


def stabilitaet_bewerten(ergebnisse: List[Dict]) -> bool:
    """
    Bewertet ob das Modell über alle Zeitfenster stabil ist.

    Ein Modell gilt als stabil wenn kein Fenster mehr als STABILITAETS_SCHWELLE
    unterhalb des Durchschnitts liegt.

    Args:
        ergebnisse: Liste der Fenster-Ergebnisse

    Returns:
        True wenn stabil, False wenn instabil
    """
    f1_werte = [e["f1_macro"] for e in ergebnisse if not np.isnan(e["f1_macro"])]
    if not f1_werte:
        logger.error("Keine gültigen F1-Werte – kann Stabilität nicht bewerten!")
        return False

    durchschnitt = float(np.mean(f1_werte))
    minimum = float(np.min(f1_werte))
    maximum = float(np.max(f1_werte))
    std = float(np.std(f1_werte))

    logger.info("\n  F1-Macro Statistiken über alle Fenster:")
    logger.info(f"    Durchschnitt: {durchschnitt:.4f}")
    logger.info(f"    Minimum:      {minimum:.4f}")
    logger.info(f"    Maximum:      {maximum:.4f}")
    logger.info(f"    Std-Abweichung: {std:.4f}")
    logger.info(f"    Schwankungsbreite: {maximum - minimum:.4f}")
    schwellenwert = durchschnitt - STABILITAETS_SCHWELLE
    logger.info(
        f"\n  Stabilitätsschwelle: Ø − {STABILITAETS_SCHWELLE} = {schwellenwert:.4f}"
    )

    # Fenster analysieren
    for e in ergebnisse:
        if not np.isnan(e["f1_macro"]):
            abstand = durchschnitt - e["f1_macro"]
            status = "❌ ZU SCHWACH" if abstand > STABILITAETS_SCHWELLE else "✅ OK"
            logger.info(
                f"    {e['name']}: F1={e['f1_macro']:.4f} | Abstand: {abstand:+.4f} {status}"
            )

    # Stabilität: kein Fenster darf zu weit unter dem Durchschnitt liegen
    ist_stabil = (durchschnitt - minimum) <= STABILITAETS_SCHWELLE
    return ist_stabil


# ============================================================
# 5. Ergebnisse visualisieren
# ============================================================


def ergebnisse_visualisieren(ergebnisse: List[Dict], symbol: str) -> None:
    """
    Erstellt ein Balkendiagramm der F1-Scores pro Walk-Forward-Fenster.

    Grüne Balken = innerhalb der Stabilitätsschwelle (OK)
    Rote Balken  = zu weit unter dem Durchschnitt (instabil)

    Args:
        ergebnisse: Liste der Fenster-Ergebnisse
        symbol: Handelssymbol (für Dateiname und Titel)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Daten vorbereiten
    namen = [e["name"] for e in ergebnisse]
    f1_werte = [e["f1_macro"] for e in ergebnisse]
    test_perioden = [
        f"{e.get('test_von', '?')}–{e.get('test_bis', '?')}" for e in ergebnisse
    ]

    # Durchschnitt und Schwellenwert berechnen
    gueltige = [f for f in f1_werte if not np.isnan(f)]
    mittelwert = float(np.mean(gueltige)) if gueltige else 0.0
    schwelle = mittelwert - STABILITAETS_SCHWELLE

    # Farben: Grün = stabil (OK), Rot = instabil (zu schwach)
    farben = [
        "#27AE60" if (not np.isnan(f) and f >= schwelle) else "#E74C3C"
        for f in f1_werte
    ]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(11, 6))
    balken = ax.bar(
        namen, f1_werte, color=farben, edgecolor="white", linewidth=1.5, width=0.6
    )

    # F1-Werte über den Balken anzeigen
    for balken_obj, wert in zip(balken, f1_werte):
        if not np.isnan(wert):
            ax.text(
                balken_obj.get_x() + balken_obj.get_width() / 2,
                wert + 0.003,
                f"{wert:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#2C3E50",
            )

    # Referenzlinien: Durchschnitt und Stabilitätsschwelle
    ax.axhline(
        mittelwert,
        color="#2C3E50",
        linestyle="--",
        linewidth=2,
        label=f"Durchschnitt: {mittelwert:.4f}",
    )
    ax.axhline(
        schwelle,
        color="#E74C3C",
        linestyle=":",
        linewidth=2,
        label=f"Stabilitätsschwelle: {schwelle:.4f} (Ø − {STABILITAETS_SCHWELLE})",
    )

    # X-Achsen-Beschriftungen: Fenster-Name + Test-Periode
    ax.set_xticks(range(len(namen)))
    ax.set_xticklabels(
        [f"{n}\n({p})" for n, p in zip(namen, test_perioden)],
        fontsize=9,
    )

    # Achsenbeschriftungen und Titel
    ax.set_ylabel("F1-Macro", fontsize=12)
    y_min = (
        max(0, min(f for f in f1_werte if not np.isnan(f)) - 0.05) if gueltige else 0
    )
    y_max = max(f for f in f1_werte if not np.isnan(f)) * 1.12 if gueltige else 1.0
    ax.set_ylim(y_min, y_max)
    ax.set_title(
        f"{symbol} – Walk-Forward-Analyse\n"
        f"LightGBM Baseline | 5 Expanding Windows | Test-Set 2023+ unberührt",
        fontsize=13,
        fontweight="bold",
    )

    # Legende
    legende_patches = [
        mpatches.Patch(color="#27AE60", label="Stabil (OK)"),
        mpatches.Patch(
            color="#E74C3C", label=f"Instabil (> {STABILITAETS_SCHWELLE} unter Ø)"
        ),
        mpatches.Patch(color="#2C3E50", label=f"Durchschnitt: {mittelwert:.4f}"),
        mpatches.Patch(color="#E74C3C", alpha=0.4, label=f"Schwelle: {schwelle:.4f}"),
    ]
    ax.legend(handles=legende_patches, fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_walk_forward.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Walk-Forward-Plot gespeichert: {pfad}")


# ============================================================
# 6. Walk-Forward für ein Symbol
# ============================================================


def walk_forward_analyse(symbol: str, version: str = "v1") -> bool:
    """
    Führt die vollständige Walk-Forward-Analyse für ein Symbol durch.

    Schritte:
        1. Daten laden (versioniertes labeled CSV)
        2. Features aufbereiten
        3. 5 Fenster trainieren und evaluieren
        4. Stabilität bewerten
        5. Plot speichern
        6. Detailberichte ausgeben

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versions-String für den Datei-Pfad (Standard: "v1")

    Returns:
        True wenn das Modell über alle Fenster stabil ist, sonst False.
    """
    logger.info("=" * 60)
    logger.info(f"Walk-Forward-Analyse – {symbol} ({version})")
    logger.info(f"Fenster: {len(FENSTER)} | Methode: Expanding Window")
    logger.info(f"Stabilität OK wenn: kein Fenster > {STABILITAETS_SCHWELLE} unter Ø")
    logger.info("=" * 60)

    # Daten laden und aufbereiten (versionierter Pfad)
    df = daten_laden(symbol, version)
    X, y = features_aufbereiten(df)
    logger.info(f"[{symbol}] Features: {len(X.columns)} Spalten")

    # Alle 5 Fenster trainieren und evaluieren
    ergebnisse = []
    for i, fenster in enumerate(FENSTER, 1):
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Starte Fenster {i}/{len(FENSTER)}")
        ergebnis = fenster_trainieren(X, y, fenster)
        ergebnisse.append(ergebnis)

    # Stabilität über alle Fenster bewerten
    logger.info(f"\n{'=' * 40}")
    logger.info("STABILITÄTS-BEWERTUNG")
    logger.info(f"{'=' * 40}")
    ist_stabil = stabilitaet_bewerten(ergebnisse)

    if ist_stabil:
        logger.info(f"\n✅ [{symbol}] Modell ist STABIL über alle Fenster!")
        logger.info(
            f"   Kein Fenster liegt mehr als {STABILITAETS_SCHWELLE} unter dem Durchschnitt."
        )
        logger.info(
            "   → Modell kann für Backtesting und Live-Trading eingesetzt werden."
        )
    else:
        logger.warning(f"\n⚠️  [{symbol}] Modell zeigt INSTABILITÄT über die Zeit!")
        logger.warning(
            f"   Mindestens ein Fenster liegt > {STABILITAETS_SCHWELLE} unter dem Durchschnitt."
        )
        logger.warning("   Mögliche Ursachen:")
        logger.warning("     1. Overfitting auf bestimmte Marktphasen")
        logger.warning("     2. Regime-Shift (z.B. COVID-Crash 2020)")
        logger.warning("     3. Hyperparameter müssen pro Zeitraum angepasst werden")
        logger.warning("   Empfehlung: TP/SL-Parameter oder Features anpassen")

    # Plot speichern
    ergebnisse_visualisieren(ergebnisse, symbol)

    # Detailberichte pro Fenster ausgeben (für Debugging)
    logger.info(f"\n{'─' * 40}")
    logger.info("DETAILBERICHTE PRO FENSTER:")
    for e in ergebnisse:
        if e.get("bericht"):
            logger.info(
                f"\n{e['name']} "
                f"(Test: {e.get('test_von', '?')}–{e.get('test_bis', '?')}):"
            )
            logger.info(e["bericht"])

    return ist_stabil


# ============================================================
# 7. Hauptprogramm
# ============================================================


def main() -> None:
    """Walk-Forward-Analyse für ein oder alle Symbole."""
    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – Walk-Forward-Analyse"
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help=(
            "Handelssymbol (Standard: EURUSD) oder 'alle' für alle 7 Forex-Paare. "
            "Mögliche Werte: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD"
        ),
    )
    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Versions-Suffix für den Eingabe-Dateinamen (Standard: v1). "
            "Muss mit --version in labeling.py und train_model.py übereinstimmen."
        ),
    )
    args = parser.parse_args()

    SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

    # Symbole bestimmen
    if args.symbol.lower() == "alle":
        ziel_symbole = SYMBOLE
    elif args.symbol.upper() in SYMBOLE:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: {args.symbol}")
        print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
        return

    start_zeit = datetime.now()
    logger.info(f"Start: {start_zeit.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Symbole: {', '.join(ziel_symbole)} | Version: {args.version}")

    gesamt_ergebnisse = []

    for symbol in ziel_symbole:
        try:
            ist_stabil = walk_forward_analyse(symbol, args.version)
            gesamt_ergebnisse.append((symbol, ist_stabil))
        except FileNotFoundError as e:
            logger.error(str(e))
            gesamt_ergebnisse.append((symbol, None))

    # Abschlusszusammenfassung
    ende_zeit = datetime.now()
    dauer_sek = int((ende_zeit - start_zeit).total_seconds())

    print("\n" + "=" * 60)
    print(f"ABGESCHLOSSEN – Walk-Forward-Analyse ({args.version})")
    print("=" * 60)
    for symbol, stabil in gesamt_ergebnisse:
        if stabil is True:
            zeichen = "✅"
            status = "STABIL"
        elif stabil is False:
            zeichen = "⚠️ "
            status = "INSTABIL – bitte Parameter prüfen"
        else:
            zeichen = "✗ "
            status = "FEHLER – labeled CSV fehlt"
        print(f"  {zeichen} {symbol}: {status}")

    stabil_anzahl = sum(1 for _, s in gesamt_ergebnisse if s is True)
    print(f"\n{stabil_anzahl}/{len(gesamt_ergebnisse)} Modelle stabil")
    print(f"Laufzeit: {dauer_sek // 60}m {dauer_sek % 60}s")
    print("\nPlots gespeichert: plots/SYMBOL_walk_forward.png")
    print(f"\nNächster Schritt: backtest.py --symbol alle --version {args.version} ausführen")
    print("=" * 60)


if __name__ == "__main__":
    main()
