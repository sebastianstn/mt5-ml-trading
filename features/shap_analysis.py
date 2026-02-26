"""
shap_analysis.py – SHAP-basierte Modell-Erklärbarkeit

Erklärt das trainierte LightGBM-Modell mit SHAP (SHapley Additive exPlanations):
    - Welche Features sind global am wichtigsten?
    - Wie wirkt sich jedes Feature positiv/negativ auf die Vorhersage aus?

SHAP erklärt WARUM das Modell eine Entscheidung trifft – nicht nur WAS es entscheidet.
Das ist wichtig, um Overfitting zu erkennen und das Modell zu verstehen.

Läuft auf: Linux-Server (nach train_model.py)

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/shap_analysis.py [--symbol EURUSD]
    python features/shap_analysis.py --symbol alle

Eingabe:  models/lgbm_SYMBOL_v1.pkl
          data/SYMBOL_H1_labeled.csv
Ausgabe:  plots/SYMBOL_shap_summary.png    ← globale Feature-Wichtigkeit
          plots/SYMBOL_shap_beeswarm.png   ← Einfluss pro Feature auf Long-Klasse
"""

# Standard-Bibliotheken
import argparse
import logging
from pathlib import Path

# Datenverarbeitung
import numpy as np
import pandas as pd

# Modell laden
import joblib

# SHAP-Bibliothek für Modell-Erklärbarkeit
import shap

# Visualisierung
import matplotlib

matplotlib.use("Agg")  # Kein Fenster öffnen – wir speichern nur PNG
import matplotlib.pyplot as plt  # noqa: E402

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("shap_analysis.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade (absolut und plattformunabhängig)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

# Gleiche Ausschluss-Spalten wie in train_model.py (skalenabhängige Features)
AUSSCHLUSS_SPALTEN = {
    "open",
    "high",
    "low",
    "close",  # Rohe Preise – skalenabhängig
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
    "bb_lower",  # Absolute Bollinger-Bänder
    "obv",  # Kumulativer OBV (nicht normiert)
    "label",  # Zielvariable – kein Feature!
}

# Symbole (alle 7 Forex-Hauptpaare)
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]


# ============================================================
# 1. Daten laden und aufbereiten
# ============================================================


def daten_laden(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Lädt den Validierungszeitraum (2022) für SHAP-Analyse.

    Wir nutzen das Validierungsset (2022), weil:
    - Es während des Trainings NICHT gesehen wurde (kein Look-Ahead-Bias)
    - Es repräsentativ für neue Marktbedingungen ist
    - Es groß genug für aussagekräftige SHAP-Werte ist (~6.250 Kerzen)

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        Tuple (X_val, y_val) – Validierungs-Features und Labels

    Raises:
        FileNotFoundError: Wenn die gelabelte CSV nicht existiert.
    """
    pfad = DATA_DIR / f"{symbol}_H1_labeled.csv"
    if not pfad.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {pfad}\n" f"Zuerst labeling.py ausführen!"
        )

    logger.info(f"[{symbol}] Lade {pfad.name} ...")
    df = pd.read_csv(pfad, index_col="time", parse_dates=True)
    logger.info(f"[{symbol}] {len(df):,} Kerzen | {len(df.columns)} Spalten")

    # Features: alle Spalten außer den Ausschluss-Spalten
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    X = df[feature_spalten].copy()

    # NaN-Werte mit Median auffüllen (Sicherheitsnetz)
    nan_anzahl = X.isna().sum().sum()
    if nan_anzahl > 0:
        logger.warning(f"[{symbol}] {nan_anzahl} NaN-Werte – werden mit Median gefüllt")
        X = X.fillna(X.median())

    # Labels umkodieren: {-1, 0, 1} → {0, 1, 2}
    y = df["label"].map({-1: 0, 0: 1, 1: 2})

    # Validierungszeitraum: 2022 (nach dem Training, vor dem Test-Set)
    val_maske = (X.index > "2021-12-31") & (X.index <= "2022-12-31")
    X_val = X[val_maske]
    y_val = y[val_maske]

    logger.info(
        f"[{symbol}] Validierungsset: {len(X_val):,} Kerzen | "
        f"{X_val.index[0].date()} bis {X_val.index[-1].date()}"
    )
    logger.info(f"[{symbol}] Features: {len(feature_spalten)}")

    return X_val, y_val


# ============================================================
# 2. SHAP-Werte berechnen
# ============================================================


def shap_werte_berechnen(
    modell,
    X_val: pd.DataFrame,
    n_stichproben: int = 2000,
) -> tuple:
    """
    Berechnet SHAP-Werte mit dem TreeExplainer.

    shap.TreeExplainer ist optimiert für Baum-Modelle (LightGBM, XGBoost).
    Er ist exakt (keine Approximation) und deutlich schneller als KernelSHAP.

    Bei Multiclass-Modellen (3 Klassen) gibt shap_values eine Liste zurück:
        shap_werte[0] = SHAP-Werte für Klasse 0 (Short)
        shap_werte[1] = SHAP-Werte für Klasse 1 (Neutral)
        shap_werte[2] = SHAP-Werte für Klasse 2 (Long)

    Args:
        modell: Trainiertes LightGBM-Modell
        X_val: Validierungs-Features
        n_stichproben: Anzahl der Stichproben für SHAP (mehr = langsamer aber genauer)

    Returns:
        Tuple (shap_werte, X_shap, erklaerer) – SHAP-Werte, Stichproben, Explainer
    """
    # Zufällige Stichprobe für SHAP (verhindert zu lange Laufzeit)
    n_shap = min(n_stichproben, len(X_val))
    X_shap = X_val.sample(n=n_shap, random_state=42)
    logger.info(f"Berechne SHAP-Werte auf {n_shap} Stichproben ...")

    # TreeExplainer erstellen (optimiert für Gradient-Boosting-Modelle)
    erklaerer = shap.TreeExplainer(modell)

    # SHAP-Werte berechnen
    # Ergebnis: Liste mit 3 Arrays (je eine pro Klasse), je Form (n_shap, n_features)
    shap_werte = erklaerer.shap_values(X_shap)

    if isinstance(shap_werte, list):
        logger.info(
            f"SHAP-Werte berechnet: {len(shap_werte)} Klassen × "
            f"{shap_werte[0].shape[0]} Stichproben × "
            f"{shap_werte[0].shape[1]} Features"
        )
    else:
        logger.info(f"SHAP-Werte berechnet: Form = {shap_werte.shape}")

    return shap_werte, X_shap, erklaerer


# ============================================================
# 3. Plot 1: SHAP Summary Bar (globale Feature-Wichtigkeit)
# ============================================================


def summary_bar_plotten(
    shap_werte,
    feature_namen: list,
    symbol: str,
    top_n: int = 20,
) -> None:
    """
    Erstellt einen Balkenplot der globalen SHAP-Feature-Wichtigkeit.

    Zeigt die mittlere absolute SHAP-Wichtigkeit pro Feature,
    gemittelt über alle Klassen (Short + Neutral + Long).

    Args:
        shap_werte: SHAP-Werte (Liste bei Multiclass)
        feature_namen: Liste der Feature-Namen
        symbol: Handelssymbol (für Dateiname und Titel)
        top_n: Anzahl der angezeigten Features
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mittlere absolute SHAP-Wichtigkeit über alle Klassen berechnen
    # Für jede Klasse: mean(|SHAP|) pro Feature → dann über Klassen mitteln
    if isinstance(shap_werte, list):
        # Multiclass: mittlerer Absolutwert über alle 3 Klassen
        wichtigkeit = np.mean([np.abs(sv).mean(axis=0) for sv in shap_werte], axis=0)
    else:
        # Binary oder Regression: direkt mitteln
        wichtigkeit = np.abs(shap_werte).mean(axis=0)

    # Top-N Features nach Wichtigkeit sortieren (absteigend)
    sortiert = np.argsort(wichtigkeit)[::-1][:top_n]
    top_namen = [feature_namen[i] for i in sortiert]
    top_werte = wichtigkeit[sortiert]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 8))
    farben = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_namen)))[::-1]

    ax.barh(top_namen, top_werte, color=farben, edgecolor="white", linewidth=0.8)
    ax.invert_yaxis()  # Wichtigstes Feature oben
    ax.set_xlabel("Mittlere |SHAP|-Wichtigkeit (über alle Klassen)", fontsize=11)
    ax.set_title(
        f"{symbol} – LightGBM Modell-Erklärbarkeit\n"
        f"Top {top_n} Features nach SHAP-Wichtigkeit",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # Top-3 Features hervorheben
    for i, (balken, wert) in enumerate(zip(ax.patches, top_werte)):
        if i < 3:
            ax.text(
                wert + max(top_werte) * 0.01,
                balken.get_y() + balken.get_height() / 2,
                f"#{i + 1}",
                va="center",
                fontsize=9,
                color="#2C3E50",
                fontweight="bold",
            )

    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_shap_summary.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] SHAP Summary-Plot gespeichert: {pfad}")

    # Top-10 im Log ausgeben
    logger.info(f"\n[{symbol}] Top 10 wichtigste Features (SHAP):")
    for i, (name, wert) in enumerate(zip(top_namen[:10], top_werte[:10]), 1):
        logger.info(f"  {i:2d}. {name:30s}: {wert:.6f}")


# ============================================================
# 4. Plot 2: SHAP Beeswarm (Einfluss auf Long-Klasse)
# ============================================================


def beeswarm_plotten(
    shap_werte,
    erklaerer,
    X_shap: pd.DataFrame,
    symbol: str,
) -> None:
    """
    Erstellt einen SHAP Beeswarm-Plot für die Long-Klasse (Klasse 2).

    Der Beeswarm-Plot zeigt für jedes Feature:
    - X-Achse: SHAP-Wert (positiv = erhöht Long-Wahrscheinlichkeit)
    - Farbe: Feature-Wert (rot = hoch, blau = niedrig)
    - Jeder Punkt = eine Datenzeile

    So sehen wir z.B.: "Hoher RSI (rot) → positiver SHAP → erhöht Long-Signal"

    Args:
        shap_werte: SHAP-Werte (Liste bei Multiclass)
        erklaerer: SHAP TreeExplainer (für expected_value)
        X_shap: Stichproben-Features (für Farbkodierung)
        symbol: Handelssymbol (für Dateiname)
    """
    if not isinstance(shap_werte, list) or len(shap_werte) < 3:
        logger.warning("[{symbol}] Kein Multiclass-Modell – Beeswarm übersprungen")
        return

    # SHAP-Werte für Klasse 2 (Long) auswählen
    shap_long = shap_werte[2]

    # Expected Value (Basiswert) für Long-Klasse
    expected_value = erklaerer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        basiswert = float(expected_value[2])
    else:
        basiswert = float(expected_value)

    # shap.Explanation-Objekt erstellen (benötigt für neuere SHAP-API)
    erklaerung = shap.Explanation(
        values=shap_long,
        base_values=np.full(len(shap_long), basiswert),
        data=X_shap.values,
        feature_names=list(X_shap.columns),
    )

    # Beeswarm-Plot erstellen
    plt.figure(figsize=(10, 9))
    shap.plots.beeswarm(erklaerung, max_display=20, show=False)

    plt.title(
        f"{symbol} – SHAP Beeswarm (Long-Klasse)\n"
        f"Rot = hoher Feature-Wert | Blau = niedriger Feature-Wert\n"
        f"SHAP > 0 → erhöht Long-Wahrscheinlichkeit | SHAP < 0 → verringert sie",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_shap_beeswarm.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] SHAP Beeswarm-Plot gespeichert: {pfad}")


# ============================================================
# 5. Vollständige SHAP-Analyse für ein Symbol
# ============================================================


def shap_analyse(symbol: str) -> None:
    """
    Führt die vollständige SHAP-Analyse für ein Symbol durch.

    Schritte:
        1. LightGBM-Modell laden
        2. Validierungsdaten laden (2022)
        3. SHAP-Werte berechnen
        4. Summary-Bar-Plot erstellen (globale Wichtigkeit)
        5. Beeswarm-Plot erstellen (Einfluss auf Long-Klasse)
        6. Top-Features im Log ausgeben

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Raises:
        FileNotFoundError: Wenn Modell oder Daten fehlen.
    """
    logger.info(f"\n{'─' * 50}")
    logger.info(f"[{symbol}] Starte SHAP-Analyse")
    logger.info(f"{'─' * 50}")

    # Modell laden
    modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_v1.pkl"
    if not modell_pfad.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Zuerst train_model.py --symbol {symbol} ausführen!"
        )
    logger.info(f"[{symbol}] Lade Modell: {modell_pfad.name}")
    modell = joblib.load(modell_pfad)

    # Daten laden (Validierungsset 2022)
    X_val, y_val = daten_laden(symbol)
    feature_namen = list(X_val.columns)

    # SHAP-Werte berechnen
    shap_werte, X_shap, erklaerer = shap_werte_berechnen(modell, X_val)

    # Plot 1: Summary Bar (globale Feature-Wichtigkeit über alle Klassen)
    summary_bar_plotten(shap_werte, feature_namen, symbol)

    # Plot 2: Beeswarm für Long-Klasse
    beeswarm_plotten(shap_werte, erklaerer, X_shap, symbol)

    logger.info(f"[{symbol}] SHAP-Analyse abgeschlossen ✅")


# ============================================================
# 6. Hauptprogramm
# ============================================================


def main() -> None:
    """SHAP-Analyse für ein oder alle Symbole."""

    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – SHAP Modell-Erklärbarkeit"
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help=(
            "Handelssymbol (Standard: EURUSD) oder 'alle' für alle 7 Forex-Paare. "
            "Mögliche Werte: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD"
        ),
    )
    args = parser.parse_args()

    # Symbole bestimmen
    if args.symbol.lower() == "alle":
        ziel_symbole = SYMBOLE
    elif args.symbol.upper() in SYMBOLE:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: {args.symbol}")
        print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
        return

    logger.info("=" * 60)
    logger.info("Phase 4 – SHAP Modell-Erklärbarkeit")
    logger.info(f"Symbole: {', '.join(ziel_symbole)}")
    logger.info("=" * 60)

    ergebnisse = []
    for symbol in ziel_symbole:
        try:
            shap_analyse(symbol)
            ergebnisse.append((symbol, "OK"))
        except FileNotFoundError as e:
            logger.error(str(e))
            ergebnisse.append((symbol, "FEHLER – Modell oder Daten fehlen"))
        except Exception as e:
            logger.error(f"[{symbol}] Unerwarteter Fehler: {e}")
            ergebnisse.append((symbol, f"FEHLER – {e}"))

    # Abschlusszusammenfassung
    print("\n" + "=" * 60)
    print("ABGESCHLOSSEN – SHAP-Analyse")
    print("=" * 60)
    for symbol, status in ergebnisse:
        zeichen = "✓" if status == "OK" else "✗"
        print(f"  {zeichen} {symbol}: {status}")

    erfolge = sum(1 for _, s in ergebnisse if s == "OK")
    print(f"\n{erfolge}/{len(ergebnisse)} Symbole erfolgreich analysiert")
    print(f"\nPlots gespeichert in: plots/")
    print("  SYMBOL_shap_summary.png   ← globale Feature-Wichtigkeit")
    print("  SYMBOL_shap_beeswarm.png  ← Einfluss auf Long-Klasse")
    print("\nNächster Schritt: backtest.py ausführen")
    print("=" * 60)


if __name__ == "__main__":
    main()
