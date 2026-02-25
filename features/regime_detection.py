"""
regime_detection.py – Automatische Marktphasen-Erkennung (Regime Detection)

Erkennt für jede H1-Kerze die aktuelle Marktphase:
    0 = Seitwärts     (geringe Volatilität, kein klarer Trend)
    1 = Aufwärtstrend (starker Trend, Preis über SMA 50)
    2 = Abwärtstrend  (starker Trend, Preis unter SMA 50)
    3 = Hohe Volatilität (Marktschock, News-Events)

Methodik:
    - Trendstärke:   ADX(14) – Average Directional Index
    - Richtung:      Close vs. SMA 50
    - Volatilität:   ATR% vs. rollender Median ATR% (50 Perioden)
    - Priorität:     Hohe Vola > Trend > Seitwärts

Läuft auf: Linux-Server (/mnt/1T-Data/XGBoost-LightGBM/)

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/regime_detection.py

Eingabe:  data/SYMBOL_H1_features.csv  (aus feature_engineering.py)
Ausgabe:  data/SYMBOL_H1_features.csv  (überschrieben, +2 Spalten: market_regime, adx_14)
          plots/SYMBOL_regime.png      (Visualisierung)
"""

# Standard-Bibliotheken
import logging
from pathlib import Path
from typing import Tuple

# Datenverarbeitung
import pandas as pd
import numpy as np

# Technische Indikatoren (ADX)
import pandas_ta as ta

# Visualisierung (Agg-Backend = kein Display-Fenster nötig auf dem Server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Logging konfigurieren (Terminal + Datei)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("regime_detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade (absolut, relativ zum Skript-Verzeichnis)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# Alle 7 Forex-Hauptpaare
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# Regime-Beschriftungen für Logging und Charts
REGIME_NAMEN = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

# Farben für die Visualisierung
REGIME_FARBEN = {
    0: "#888888",  # Grau
    1: "#00AA00",  # Grün
    2: "#CC0000",  # Rot
    3: "#FF8C00",  # Orange
}


# ============================================================
# 1. ADX berechnen (Trendstärke)
# ============================================================


def adx_berechnen(df: pd.DataFrame, laenge: int = 14) -> pd.Series:
    """
    Berechnet den Average Directional Index (ADX).

    ADX misst die STÄRKE eines Trends, NICHT die Richtung.
    Interpretation:
        ADX < 20 → kein klarer Trend (Seitwärtsmarkt)
        ADX > 25 → klarer Trend (egal ob hoch oder runter)
        ADX > 40 → sehr starker Trend

    Args:
        df: DataFrame mit Spalten 'high', 'low', 'close'
        laenge: Berechnungsperiode (Standard: 14)

    Returns:
        ADX-Serie (Werte 0–100, NaN für die ersten `laenge` Kerzen)
    """
    # pandas_ta berechnet ADX, +DI und -DI zusammen
    adx_df = df.ta.adx(length=laenge)

    # Spaltenname dynamisch finden (z.B. "ADX_14")
    adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]

    return adx_df[adx_col]


# ============================================================
# 2. Volatilitätsschwelle berechnen
# ============================================================


def volatilitaet_berechnen(
    df: pd.DataFrame, fenster: int = 50
) -> Tuple[pd.Series, pd.Series]:
    """
    Berechnet aktuelle Volatilität und rollende Referenzschwelle.

    Statt absoluter ATR-Schwellen verwenden wir einen relativen
    Ansatz: Wenn die aktuelle ATR% mehr als 1.5× über dem rollenden
    Median liegt, gilt die Volatilität als "erhöht".

    Args:
        df: DataFrame mit Spalte 'atr_pct' (ATR in Prozent vom Close,
            bereits in feature_engineering.py berechnet)
        fenster: Lookback-Fenster für rollenden Median (Standard: 50)

    Returns:
        Tuple (atr_pct, rollender_median_atr)
    """
    atr_pct = df["atr_pct"]

    # Rollender Median der letzten 50 Kerzen = "normales" Vola-Niveau
    # min_periods=fenster: Erst ab 50 gültigen Werten berechnen
    median_atr = atr_pct.rolling(window=fenster, min_periods=fenster).median()

    return atr_pct, median_atr


# ============================================================
# 3. Regime klassifizieren
# ============================================================


def regime_klassifizieren(
    df: pd.DataFrame,
    adx: pd.Series,
    atr_pct: pd.Series,
    median_atr: pd.Series,
    adx_schwelle: float = 25.0,
    vol_faktor: float = 1.5,
) -> pd.Series:
    """
    Klassifiziert jede Kerze in eine der 4 Marktphasen.

    Entscheidungsbaum (Priorität: Volatilität > Trend > Seitwärts):
        1. Hohe Vola:      atr_pct > vol_faktor × median_atr    → Regime 3
        2. Aufwärtstrend:  ADX > adx_schwelle AND Close > SMA50  → Regime 1
        3. Abwärtstrend:   ADX > adx_schwelle AND Close < SMA50  → Regime 2
        4. Seitwärts:      alles andere                          → Regime 0

    WICHTIG – kein Look-Ahead-Bias:
        Alle Berechnungen verwenden nur Daten bis Zeitpunkt T.
        ADX und rollender Median sind rein rückwärtsschauende Indikatoren.

    Args:
        df: Feature DataFrame mit Spalten 'close' und 'sma_50'
        adx: ADX-Serie (0–100)
        atr_pct: Aktuelle ATR in Prozent
        median_atr: Rollender Median der ATR% (Referenz)
        adx_schwelle: Mindestwert für Trend-Erkennung (Standard: 25.0)
        vol_faktor: Multiplikator für Volatilitätsschwelle (Standard: 1.5)

    Returns:
        Regime-Serie mit Werten 0, 1, 2, 3 (oder -1 für NaN-Anfangsperiode)
    """
    # Start: alle Kerzen als Seitwärts (0)
    regime = pd.Series(0, index=df.index, dtype=int)

    # Bedingungen vektorisiert berechnen (kein Python-Loop!)
    hoch_vol = atr_pct > (vol_faktor * median_atr)
    aufwaerts = (adx > adx_schwelle) & (df["close"] > df["sma_50"]) & ~hoch_vol
    abwaerts = (adx > adx_schwelle) & (df["close"] < df["sma_50"]) & ~hoch_vol

    # Regime zuweisen (Reihenfolge = Priorität)
    regime[aufwaerts] = 1
    regime[abwaerts] = 2
    regime[hoch_vol] = 3

    # Anfangsperiode ohne gültige Indikatoren als -1 markieren
    nan_maske = adx.isna() | median_atr.isna()
    regime[nan_maske] = -1

    return regime


# ============================================================
# 4. Regime-Verteilung validieren
# ============================================================


def regime_validieren(regime: pd.Series, symbol: str) -> bool:
    """
    Prüft ob die Regime-Verteilung realistisch ist.

    Eine gute Regime-Erkennung sollte alle 4 Klassen abdecken.
    Warnung wenn:
        - Eine Klasse > 60% aller Kerzen dominiert
        - Eine Klasse < 5% vorkommt (praktisch nie erkannt)

    Args:
        regime: Regime-Serie (0, 1, 2, 3; -1 = NaN)
        symbol: Symbol-Name (für Logging)

    Returns:
        True wenn Verteilung sinnvoll, False wenn Probleme gefunden.
    """
    # NaN-Anfangsperiode ausschließen
    gueltig = regime[regime >= 0]
    verteilung = gueltig.value_counts(normalize=True).sort_index()

    # Verteilung als Balkendiagramm im Log ausgeben
    logger.info(f"\n[{symbol}] Regime-Verteilung ({len(gueltig):,} gültige Kerzen):")
    for regime_nr in [0, 1, 2, 3]:
        anteil = verteilung.get(regime_nr, 0.0)
        name = REGIME_NAMEN[regime_nr]
        balken = "█" * int(anteil * 40)
        leer = "░" * (40 - int(anteil * 40))
        logger.info(f"  {regime_nr} {name:18s}: {anteil * 100:5.1f}% [{balken}{leer}]")

    probleme = False

    # Warnung: Dominanz einer Klasse
    if verteilung.max() > 0.60:
        dom = verteilung.idxmax()
        logger.warning(
            f"[{symbol}] Regime {dom} ({REGIME_NAMEN[dom]}) dominiert mit "
            f"{verteilung.max() * 100:.1f}%! Schwellen prüfen."
        )
        probleme = True

    # Warnung: Klasse fast nicht vorhanden
    for r in [0, 1, 2, 3]:
        anteil = verteilung.get(r, 0.0)
        if anteil < 0.05:
            logger.warning(
                f"[{symbol}] Regime {r} ({REGIME_NAMEN[r]}) nur {anteil * 100:.1f}% – "
                f"Schwellen anpassen?"
            )
            probleme = True

    return not probleme


# ============================================================
# 5. Visualisierung
# ============================================================


def regime_visualisieren(df: pd.DataFrame, symbol: str) -> None:
    """
    Erstellt einen 2-Panel-Chart: Kursverlauf + Marktphasen.

    Oberer Panel: Kursverlauf mit SMA50 und farbiger Hintergrund je Regime
    Unterer Panel: Regime-Zeitreihe als farbiger Stufenchart

    Speichert als: plots/SYMBOL_regime.png

    Args:
        df: DataFrame mit Spalten 'close', 'sma_50', 'market_regime'
        symbol: Handelssymbol (für Titel und Dateiname)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor("#F8F9FA")

    # --- Oberer Panel: Kursverlauf ---
    ax1.plot(
        df.index, df["close"],
        color="#1E90FF", linewidth=0.6, label="Close", zorder=3
    )
    ax1.plot(
        df.index, df["sma_50"],
        color="#FF8C00", linewidth=1.2, label="SMA 50", alpha=0.9, zorder=2
    )

    # Hintergrundfarbe für jede Marktphase
    preis_min = df["close"].min() * 0.998
    preis_max = df["close"].max() * 1.002
    for regime_nr, farbe in REGIME_FARBEN.items():
        maske = df["market_regime"] == regime_nr
        if maske.any():
            ax1.fill_between(
                df.index, preis_min, preis_max,
                where=maske,
                alpha=0.12, color=farbe,
                label=f"Regime {regime_nr}: {REGIME_NAMEN[regime_nr]}",
            )

    ax1.set_ylim(preis_min, preis_max)
    ax1.set_title(
        f"{symbol} H1 – Regime Detection | "
        f"ADX>25=Trend, ATR>1.5×Median=Vola",
        fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Kurs")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.25)
    ax1.set_facecolor("#F8F9FA")

    # --- Unterer Panel: Regime als Stufenchart ---
    regime_werte = df["market_regime"].values
    farben_pro_kerze = [REGIME_FARBEN.get(r, "#888888") for r in regime_werte]

    # Effizienter als bar() für 8760 Kerzen: scatter mit kleinen Marken
    ax2.scatter(
        df.index, regime_werte,
        c=farben_pro_kerze, s=2, marker="|", zorder=2
    )
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(
        ["0 Seitwärts", "1 Aufwärts", "2 Abwärts", "3 Hoch Vola"],
        fontsize=8
    )
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Datum")
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.set_facecolor("#F8F9FA")

    plt.tight_layout()

    pfad = PLOTS_DIR / f"{symbol}_regime.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()

    logger.info(f"[{symbol}] Chart gespeichert: {pfad}")


# ============================================================
# 6. Hauptfunktion pro Symbol
# ============================================================


def regime_berechnen(symbol: str) -> bool:
    """
    Vollständiger Regime-Detection-Ablauf für ein Symbol.

    Schritte:
        1. Feature-CSV laden
        2. ADX(14) berechnen
        3. Volatilitätsschwelle berechnen (rollender Median)
        4. Regime klassifizieren
        5. Verteilung validieren
        6. Visualisierung erstellen
        7. CSV mit neuen Spalten speichern

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    eingabe_pfad = DATA_DIR / f"{symbol}_H1_features.csv"

    # Datei prüfen
    if not eingabe_pfad.exists():
        logger.error(f"[{symbol}] Datei nicht gefunden: {eingabe_pfad}")
        logger.error(f"[{symbol}] Zuerst feature_engineering.py ausführen!")
        return False

    # Feature-CSV laden
    logger.info(f"[{symbol}] Lade {eingabe_pfad.name} ...")
    df = pd.read_csv(eingabe_pfad, index_col="time", parse_dates=True)
    logger.info(f"[{symbol}] {len(df):,} Kerzen, {len(df.columns)} Features")

    # Pflicht-Spalten prüfen
    erforderliche_spalten = ["high", "low", "close", "sma_50", "atr_pct"]
    fehlende = [c for c in erforderliche_spalten if c not in df.columns]
    if fehlende:
        logger.error(f"[{symbol}] Fehlende Spalten: {fehlende}")
        return False

    # Schritt 1: ADX(14) berechnen
    logger.info(f"[{symbol}] Berechne ADX(14) ...")
    adx = adx_berechnen(df, laenge=14)

    # Schritt 2: Volatilitätsschwelle
    logger.info(f"[{symbol}] Berechne Volatilitätsschwelle (Median-50) ...")
    atr_pct, median_atr = volatilitaet_berechnen(df, fenster=50)

    # Schritt 3: Regime klassifizieren
    logger.info(f"[{symbol}] Klassifiziere Marktphasen ...")
    regime = regime_klassifizieren(
        df, adx, atr_pct, median_atr,
        adx_schwelle=25.0,
        vol_faktor=1.5,
    )

    # Neue Spalten hinzufügen
    df["adx_14"] = adx          # ADX als Feature für späteres ML-Modell
    df["market_regime"] = regime  # Regime-Label

    # Anfangszeilen ohne gültigen Indikator entfernen
    n_ungueltig = (df["market_regime"] == -1).sum()
    df = df[df["market_regime"] >= 0].copy()
    if n_ungueltig > 0:
        logger.info(
            f"[{symbol}] {n_ungueltig} Anfangs-Kerzen ohne Indikator entfernt "
            f"(Initialisierungsperiode ADX+Median)"
        )

    # Schritt 4: Verteilung validieren
    regime_validieren(df["market_regime"], symbol)

    # Schritt 5: Visualisierung (letztes Jahr = ~8.760 H1-Kerzen)
    logger.info(f"[{symbol}] Erstelle Chart ...")
    df_plot = df.iloc[-8760:].copy()
    regime_visualisieren(df_plot, symbol)

    # Schritt 6: Aktualisierte CSV speichern
    df.to_csv(eingabe_pfad)
    groesse_mb = eingabe_pfad.stat().st_size / 1024 / 1024
    logger.info(
        f"[{symbol}] Gespeichert: {eingabe_pfad.name} "
        f"({len(df):,} Kerzen, {len(df.columns)} Features, {groesse_mb:.1f} MB)"
    )

    return True


# ============================================================
# 7. Hauptprogramm
# ============================================================


def main() -> None:
    """Regime Detection für alle 7 Forex-Hauptpaare."""
    logger.info("=" * 60)
    logger.info("Phase 3 – Regime Detection – gestartet")
    logger.info(f"Symbole: {', '.join(SYMBOLE)}")
    logger.info("Methode: ADX(14) + ATR%/Median-50 + SMA50-Richtung")
    logger.info("=" * 60)

    ergebnisse = []

    for symbol in SYMBOLE:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Verarbeite: {symbol}")
        logger.info(f"{'─' * 40}")

        erfolg = regime_berechnen(symbol)
        ergebnisse.append((symbol, "OK" if erfolg else "FEHLER"))

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ABGESCHLOSSEN – Phase 3 Regime Detection")
    print("=" * 60)
    for symbol, status in ergebnisse:
        zeichen = "✓" if status == "OK" else "✗"
        print(f"  {zeichen} {symbol}: {status}")

    erfolge = [r for r in ergebnisse if r[1] == "OK"]
    print(f"\n{len(erfolge)}/{len(SYMBOLE)} Symbole erfolgreich verarbeitet.")
    print(f"\nCharts gespeichert in:   {PLOTS_DIR}/")
    print(f"Features + Regime in:    {DATA_DIR}/SYMBOL_H1_features.csv")
    print("\nNächster Schritt: Phase 4 – Labeling & Modelltraining")
    print("=" * 60)


if __name__ == "__main__":
    main()
