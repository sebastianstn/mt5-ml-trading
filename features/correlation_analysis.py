"""
correlation_analysis.py – Feature-Korrelationsmatrix
und Normalisierungsanalyse.

Läuft auf: Linux-Server

Was dieses Skript macht:
    1. Lädt die EURUSD_H1_labeled.csv (repräsentativ für alle Symbole)
    2. Berechnet die Pearson-Korrelationsmatrix aller Modell-Features
    3. Identifiziert hoch korrelierte Feature-Paare (|r| > 0.85)
    4. Erstellt eine visuelle Heatmap
    5. Bewertet den Normalisierungsbedarf
    6. Speichert einen Textbericht mit Empfehlungen

HINWEIS zu Normalisierung:
    XGBoost und LightGBM sind baumbasierte Modelle. Sie sind NICHT
    von der Skalierung der Features abhängig. Normalisierung ist für
    diese Modelle technisch nicht nötig, schadet aber auch nicht.

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/correlation_analysis.py

Output:
    plots/correlation_matrix.png     – Heatmap der Korrelationsmatrix
    plots/high_correlation_pairs.png – Balkendiagramm stärkster Korrelationen
    reports/feature_analysis.txt     – Textbericht mit Empfehlungen
"""

# pylint: disable=duplicate-code

# Standard-Bibliotheken
import logging
from pathlib import Path
from typing import List, Tuple

# Datenverarbeitung
import numpy as np
import pandas as pd

# Visualisierung
import matplotlib.pyplot as plt

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade (absolut, plattformunabhängig)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

# Schwellenwert für "hoch korreliert" (kann angepasst werden)
KORRELATIONS_SCHWELLE = 0.85

# Diese Spalten werden vom Modell NICHT verwendet → aus Analyse ausschließen
# (entspricht AUSSCHLUSS_SPALTEN in train_model.py)
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


# ============================================================
# 1. Daten laden und Modell-Features isolieren
# ============================================================


def modell_features_laden(symbol: str = "EURUSD") -> pd.DataFrame:
    """
    Lädt die labeled CSV und isoliert nur die Modell-Features.

    Args:
        symbol: Währungspaar (Standard: EURUSD)

    Returns:
        DataFrame mit ausschließlich Modell-Features (keine OHLCV, kein Label).

    Raises:
        FileNotFoundError: Wenn die CSV nicht gefunden wird.
    """
    pfad = DATA_DIR / f"{symbol}_H1_labeled.csv"
    if not pfad.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {pfad}")

    df = pd.read_csv(pfad, index_col="time", parse_dates=True)
    logger.info(
        "Geladen: %s – %s Kerzen, %s Spalten",
        symbol,
        f"{len(df):,}",
        len(df.columns),
    )

    # Nur Modell-Features behalten (alles außer AUSSCHLUSS_SPALTEN)
    modell_cols = [
        col
        for col in df.columns
        if col not in AUSSCHLUSS_SPALTEN
        and df[col].dtype
        in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    df_features = df[modell_cols].copy()
    logger.info(
        "Modell-Features: %s Spalten für die Analyse", len(modell_cols)
    )
    return df_features


# ============================================================
# 2. Korrelationsmatrix berechnen
# ============================================================


def korrelationsmatrix_berechnen(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Berechnet die Pearson-Korrelationsmatrix und findet hoch korrelierte Paare.

    Args:
        df: DataFrame mit Modell-Features

    Returns:
        Tuple aus:
            - Korrelationsmatrix (DataFrame)
            - Liste von (Feature1, Feature2, Korrelation) für |r| > SCHWELLE
    """
    # NaN-Werte entfernen (Rolling-Features haben am Anfang NaN)
    df_clean = df.dropna()
    logger.info(
        "Korrelationsmatrix: %s Zeilen (nach NaN-Bereinigung), %s Features",
        f"{len(df_clean):,}",
        len(df_clean.columns),
    )

    # Pearson-Korrelation berechnen
    korr_matrix = df_clean.corr(method="pearson")

    # Hoch korrelierte Paare finden
    # (nur obere Dreiecksmatrix → keine Duplikate)
    hohe_paare = []
    cols = korr_matrix.columns.tolist()
    for i, col_i in enumerate(cols):
        for j in range(i + 1, len(cols)):
            wert = korr_matrix.loc[col_i, cols[j]]
            if abs(wert) >= KORRELATIONS_SCHWELLE:
                hohe_paare.append((col_i, cols[j], round(wert, 4)))

    # Nach absolutem Korrelationswert sortieren (stärkste zuerst)
    hohe_paare.sort(key=lambda x: abs(x[2]), reverse=True)

    logger.info(
        "Gefunden: %s Paare mit |r| >= %s",
        len(hohe_paare),
        KORRELATIONS_SCHWELLE,
    )
    return korr_matrix, hohe_paare


# ============================================================
# 3. Visualisierung: Korrelations-Heatmap
# ============================================================


def heatmap_erstellen(korr_matrix: pd.DataFrame, ausgabe_pfad: Path) -> None:
    """
    Erstellt eine Korrelations-Heatmap und speichert sie als PNG.

    Args:
        korr_matrix: Pearson-Korrelationsmatrix
        ausgabe_pfad: Pfad für die PNG-Datei
    """
    n = len(korr_matrix)
    # Größe der Abbildung dynamisch an Feature-Anzahl anpassen
    groesse = max(12, n * 0.35)

    fig, ax = plt.subplots(figsize=(groesse, groesse * 0.85))

    # Korrelationsmatrix als Heatmap mit Diverging-Colormap
    # Rot = positive Korrelation, Blau = negative Korrelation
    cmap = plt.colormaps["RdBu_r"]  # plt.cm.RdBu_r ersetzt durch colormaps
    im = ax.imshow(
        korr_matrix.values,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    # Achsenbeschriftungen
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(korr_matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(korr_matrix.columns, fontsize=7)

    # Farbbalken (Legende)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson Korrelation")

    # Hochkorrelierte Zellen hervorheben (schwarzer Rahmen)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and abs(korr_matrix.values[i, j]) >= KORRELATIONS_SCHWELLE
            ):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    ax.set_title(
        f"Feature-Korrelationsmatrix ({n} Modell-Features)\n"
        f"Schwarz umrandet: |r| >= {KORRELATIONS_SCHWELLE}",
        fontsize=12,
        pad=15,
    )

    plt.tight_layout()
    fig.savefig(ausgabe_pfad, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap gespeichert: %s", ausgabe_pfad.name)


# ============================================================
# 4. Visualisierung: Hoch korrelierte Paare (Balkendiagramm)
# ============================================================


def paare_diagramm_erstellen(
    hohe_paare: List[Tuple[str, str, float]], ausgabe_pfad: Path
) -> None:
    """
    Erstellt ein Balkendiagramm der stärksten Korrelationen.

    Args:
        hohe_paare: Liste von (Feature1, Feature2, Korrelation)
        ausgabe_pfad: Pfad für die PNG-Datei
    """
    if not hohe_paare:
        logger.info(
            "Keine hoch korrelierten Paare gefunden – kein Diagramm erstellt."
        )
        return

    # Maximal 30 Paare anzeigen
    top_paare = hohe_paare[:30]
    labels = [f"{a}\n<-> {b}" for a, b, _ in top_paare]
    werte = [abs(r) for _, _, r in top_paare]
    farben = ["#d32f2f" if r > 0 else "#1565c0" for _, _, r in top_paare]

    fig, ax = plt.subplots(figsize=(14, max(6, len(top_paare) * 0.4)))

    balken_liste = ax.barh(
        range(len(top_paare)),
        werte,
        color=farben,
        edgecolor="white",
        height=0.7,
    )

    # Werte in die Balken schreiben
    for i, (balken, (_, _, r)) in enumerate(zip(balken_liste, top_paare)):
        ax.text(
            balken.get_width() + 0.005,
            i,
            f"r = {r:+.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(range(len(top_paare)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.axvline(
        KORRELATIONS_SCHWELLE,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Schwelle: {KORRELATIONS_SCHWELLE}",
    )
    ax.set_xlabel("Absolute Pearson-Korrelation |r|")
    ax.set_title(
        f"Top {len(top_paare)} hoch korrelierte Feature-Paare\n"
        "Rot = positive Korrelation, Blau = negative Korrelation",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()  # Stärkstes Paar oben

    plt.tight_layout()
    fig.savefig(ausgabe_pfad, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Paare-Diagramm gespeichert: %s", ausgabe_pfad.name)


# ============================================================
# 5. Normalisierungsanalyse – Hilfsfunktionen
# ============================================================


def _features_kategorisieren(
    df_clean: pd.DataFrame,
) -> Tuple[list, list, list]:
    """
    Kategorisiert Features nach ihrem Wertebereich.

    Args:
        df_clean: DataFrame ohne NaN-Werte

    Returns:
        Tuple (grosser_bereich, mittlerer_bereich, bereits_normalisiert)
    """
    grosser_bereich = []
    mittlerer_bereich = []
    bereits_normalisiert = []

    for col in df_clean.columns:
        serie = df_clean[col]
        min_val = serie.min()
        max_val = serie.max()
        mean_val = serie.mean()
        std_val = serie.std()

        if abs(min_val) > 100 or abs(max_val) > 100:
            grosser_bereich.append((col, min_val, max_val, mean_val, std_val))
        elif abs(min_val) > 10 or abs(max_val) > 10:
            mittlerer_bereich.append(
                (col, min_val, max_val, mean_val, std_val)
            )
        else:
            bereits_normalisiert.append(
                (col, min_val, max_val, mean_val, std_val)
            )

    return grosser_bereich, mittlerer_bereich, bereits_normalisiert


def _bericht_abschnitte_erstellen(
    grosser_bereich: list,
    mittlerer_bereich: list,
    bereits_normalisiert: list,
) -> List[str]:
    """
    Erstellt die Textabschnitte für den Normalisierungsbericht.

    Args:
        grosser_bereich: Features mit |Wert| > 100
        mittlerer_bereich: Features mit |Wert| in (10, 100]
        bereits_normalisiert: Features mit |Wert| <= 10

    Returns:
        Liste der Berichtszeilen.
    """
    zeilen: List[str] = []

    zeilen.append(
        "FEATURES MIT GROSSEM WERTEBEREICH (Priorität für Skalierung):"
    )
    zeilen.append("-" * 70)
    if grosser_bereich:
        for col, mn, mx, me, sd in grosser_bereich:
            zeilen.append(
                f"  {col:<25} Min={mn:>10.2f}  Max={mx:>10.2f}  "
                f"Mean={me:>8.2f}  Std={sd:>8.2f}"
            )
    else:
        zeilen.append("  - Keine Features in dieser Kategorie -")

    zeilen.append("")
    zeilen.append("FEATURES MIT MITTLEREM WERTEBEREICH:")
    zeilen.append("-" * 70)
    if mittlerer_bereich:
        for col, mn, mx, me, sd in mittlerer_bereich:
            zeilen.append(
                f"  {col:<25} Min={mn:>10.2f}  Max={mx:>10.2f}  "
                f"Mean={me:>8.2f}  Std={sd:>8.2f}"
            )
    else:
        zeilen.append("  - Keine Features in dieser Kategorie -")

    zeilen.append("")
    zeilen.append(
        f"BEREITS SKALIERTE FEATURES ({len(bereits_normalisiert)} Stueck):"
    )
    zeilen.append("-" * 70)
    for col, mn, mx, me, sd in bereits_normalisiert:
        zeilen.append(
            f"  {col:<25} Min={mn:>8.4f}  Max={mx:>8.4f}  "
            f"Mean={me:>7.4f}  Std={sd:>7.4f}"
        )

    return zeilen


# ============================================================
# 6. Normalisierungsanalyse – Hauptfunktion
# ============================================================


def normalisierung_analysieren(df: pd.DataFrame) -> str:
    """
    Bewertet den Normalisierungsbedarf jedes Features.

    Für XGBoost und LightGBM ist Normalisierung TECHNISCH NICHT NÖTIG.
    Baumbasierte Modelle teilen Features an Schwellenwerten – die absolute
    Größe spielt keine Rolle. Nur für neuronale Netze oder lineare Modelle
    ist Normalisierung wichtig.

    Args:
        df: DataFrame mit Modell-Features

    Returns:
        Textbericht als String.
    """
    df_clean = df.dropna()
    bericht_zeilen: List[str] = []

    bericht_zeilen.append("=" * 70)
    bericht_zeilen.append("NORMALISIERUNGSANALYSE")
    bericht_zeilen.append("=" * 70)
    bericht_zeilen.append("")
    bericht_zeilen.append(
        "WICHTIG: XGBoost und LightGBM sind baumbasierte Modelle."
    )
    bericht_zeilen.append(
        "Normalisierung ist für diese Modelle NICHT erforderlich."
    )
    bericht_zeilen.append(
        "Baeume entscheiden über Schwellenwerte, nicht über Abstände."
    )
    bericht_zeilen.append(
        "Falls in Zukunft neuronale Netze oder SVM eingesetzt werden,"
    )
    bericht_zeilen.append("sollten die folgenden Features skaliert werden:")
    bericht_zeilen.append("")

    # Features kategorisieren und Abschnitte erstellen (Hilfsfunktionen)
    grosser_bereich, mittlerer_bereich, bereits_normalisiert = (
        _features_kategorisieren(df_clean)
    )
    abschnitte = _bericht_abschnitte_erstellen(
        grosser_bereich, mittlerer_bereich, bereits_normalisiert
    )
    bericht_zeilen.extend(abschnitte)

    bericht_zeilen.append("")
    bericht_zeilen.append("EMPFEHLUNG:")
    bericht_zeilen.append("  Für XGBoost/LightGBM: Kein Handlungsbedarf.")
    bericht_zeilen.append(
        "  Falls SVM/Neural Network: "
        "StandardScaler auf alle Features anwenden."
    )
    bericht_zeilen.append("  Code-Beispiel:")
    bericht_zeilen.append(
        "    from sklearn.preprocessing import StandardScaler"
    )
    bericht_zeilen.append("    scaler = StandardScaler()")
    bericht_zeilen.append("    X_train_scaled = scaler.fit_transform(X_train)")
    bericht_zeilen.append(
        "    X_val_scaled = scaler.transform(X_val)  # NUR transform()!"
    )

    return "\n".join(bericht_zeilen)


# ============================================================
# 7. Textbericht: Korrelations-Zusammenfassung
# ============================================================


def korrelations_bericht(
    hohe_paare: List[Tuple[str, str, float]],
    df: pd.DataFrame,
) -> str:
    """
    Erstellt einen Textbericht mit Korrelationsanalyse und Empfehlungen.

    Args:
        hohe_paare: Liste hochkorrelierter Paare
        df: Feature DataFrame

    Returns:
        Bericht als String.
    """
    zeilen: List[str] = []
    zeilen.append("=" * 70)
    zeilen.append("KORRELATIONSANALYSE – ZUSAMMENFASSUNG")
    zeilen.append("=" * 70)
    zeilen.append(f"Analysiert: {len(df.columns)} Modell-Features")
    zeilen.append(f"Schwellenwert: |r| >= {KORRELATIONS_SCHWELLE}")
    zeilen.append(f"Hochkorrelierte Paare: {len(hohe_paare)}")
    zeilen.append("")

    if not hohe_paare:
        zeilen.append(
            "Keine problematisch hoch korrelierten Features gefunden!"
        )
        zeilen.append(
            "  Alle Feature-Paare haben |r| < " + str(KORRELATIONS_SCHWELLE)
        )
        return "\n".join(zeilen)

    zeilen.append(
        "HOCH KORRELIERTE PAARE (|r| >= " + str(KORRELATIONS_SCHWELLE) + "):"
    )
    zeilen.append("-" * 70)

    # Features kategorisieren (potenziell entfernbar)
    zu_entfernen = set()

    for f1, f2, r in hohe_paare:
        zeilen.append(f"  r={r:+.4f}  {f1}  <->  {f2}")

        # Empfehlung: Das weniger interpretierbare Feature merken
        # rsi_14 und rsi_centered sind identisch (rsi_centered = rsi_14 - 50)
        if "centered" in f2 or "zscore" in f2:
            zu_entfernen.add(f1)
        elif "centered" in f1 or "zscore" in f1:
            zu_entfernen.add(f2)

    # Spezifische bekannte Redundanzen erklären
    zeilen.append("")
    zeilen.append("BEKANNTE REDUNDANZEN (erwartetes Verhalten):")
    zeilen.append("-" * 70)
    bekannte = [
        (
            "rsi_14 <-> rsi_centered",
            "rsi_centered = rsi_14 - 50. Perfekte Korrelation. Entfernbar.",
        ),
        (
            "stoch_k <-> stoch_d",
            "stoch_d = geglaettetes stoch_k. Hohe Korrelation erwartet.",
        ),
        (
            "session_london <-> session_overlap",
            "session_overlap ist Teilmenge von session_london.",
        ),
        (
            "trend_h4 <-> trend_d1",
            "Beide messen Trend auf höheren Zeitrahmen. Oft gleichgerichtet.",
        ),
        (
            "price_sma20_ratio <-> price_sma50_ratio",
            "Beide messen Abstand zum gleitenden Durchschnitt.",
        ),
    ]
    for paar, erklaerung in bekannte:
        zeilen.append(f"  {paar}:")
        zeilen.append(f"    -> {erklaerung}")

    zeilen.append("")
    zeilen.append("EMPFEHLUNG FUER TREE-BASED MODELLE (XGBoost / LightGBM):")
    zeilen.append("-" * 70)
    zeilen.append(
        "  Hohe Korrelation ist für Tree-basierte Modelle KEIN krit. Problem."
    )
    zeilen.append(
        "  Trees können redundante Features ignorieren (Importance nahe 0)."
    )
    zeilen.append(
        "  Optional: rsi_centered entfernen (identisch zu rsi_14 - 50)."
    )
    zeilen.append(
        "  Optional: Ein Feature aus sehr hoch korrelierten Paaren entfernen."
    )
    zeilen.append(
        "  Pruefe Feature Importance: Features < 0.001 sind Kandidaten."
    )
    zeilen.append(
        "  ! Nicht zu viele Features entfernen – Trees nutzen Redundanz."
    )

    return "\n".join(zeilen)


# ============================================================
# 8. Hauptfunktion
# ============================================================


def main() -> None:
    """Korrelationsanalyse für EURUSD durchführen und Bericht erstellen."""
    # Ausgabe-Ordner erstellen falls nicht vorhanden
    PLOTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("KORRELATIONSANALYSE – START")
    logger.info("Gerät: Linux-Server")
    logger.info("=" * 60)

    # ── Schritt 1: Daten laden ─────────────────────────────────
    logger.info("\n[1/4] Modell-Features laden...")
    try:
        df = modell_features_laden(symbol="EURUSD")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # ── Schritt 2: Korrelationsmatrix berechnen ────────────────
    logger.info("\n[2/4] Korrelationsmatrix berechnen...")
    korr_matrix, hohe_paare = korrelationsmatrix_berechnen(df)

    # ── Schritt 3: Visualisierungen erstellen ─────────────────
    logger.info("\n[3/4] Visualisierungen erstellen...")
    heatmap_erstellen(korr_matrix, PLOTS_DIR / "correlation_matrix.png")
    paare_diagramm_erstellen(
        hohe_paare, PLOTS_DIR / "high_correlation_pairs.png"
    )

    # ── Schritt 4: Textbericht speichern ──────────────────────
    logger.info("\n[4/4] Textbericht erstellen...")

    korr_text = korrelations_bericht(hohe_paare, df)
    norm_text = normalisierung_analysieren(df)

    bericht = "\n\n".join(
        [
            "PHASE 2 – FEATURE-QUALITAETSANALYSE",
            "Erstellt von: features/correlation_analysis.py",
            f"Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"Datenquelle: EURUSD_H1_labeled.csv ({len(df):,} Kerzen)",
            "",
            korr_text,
            "",
            norm_text,
        ]
    )

    bericht_pfad = REPORTS_DIR / "feature_analysis.txt"
    bericht_pfad.write_text(bericht, encoding="utf-8")
    logger.info("Bericht gespeichert: %s", bericht_pfad)

    # Terminal-Ausgabe
    print("\n" + korr_text)
    print("\n" + norm_text)

    print("\n" + "=" * 60)
    print("KORRELATIONSANALYSE – ABGESCHLOSSEN")
    print("=" * 60)
    print("  * plots/correlation_matrix.png")
    print("  * plots/high_correlation_pairs.png")
    print("  * reports/feature_analysis.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
