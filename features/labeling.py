"""
labeling.py – Double-Barrier Labeling für das ML-Trading-System

Erstellt für jede H1-Kerze ein Label basierend auf der
zukünftigen Kursbewegung:
    1  = Long-Signal   (Kurs steigt um ≥ tp_pct innerhalb von horizon Kerzen)
   -1  = Short-Signal  (Kurs fällt um ≥ sl_pct innerhalb von horizon Kerzen)
    0  = Kein Signal   (weder TP noch SL erreicht → Seitwärts)

Methode: Symmetrische Double-Barrier (vereinfacht nach Lopez de Prado)
    - Obere Schranke: close[T] × (1 + tp_pct) → Label 1
    - Untere Schranke: close[T] × (1 − sl_pct) → Label -1
    - Zeitschranke:    horizon Kerzen → Label 0

WICHTIG – kein Look-Ahead-Bias:
    Das Label für Kerze T basiert auf Daten T+1 bis T+horizon.
    Die Features für Kerze T verwenden nur Daten bis T.
    Beim Training muss darauf geachtet werden, dass Labels als Zielvariable
    (nicht als Feature) genutzt werden.

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    # Standard (v1, TP=SL=0.3%, Horizon=5):
    python features/labeling.py

    # Option A – Horizon=10 (v2):
    python features/labeling.py --horizon 10 --version v2

    # Option B – TP/SL=0.15% (v3):
    python features/labeling.py --tp_pct 0.0015 --sl_pct 0.0015 --version v3

    # Einzelnes Symbol:
    python features/labeling.py --symbol EURUSD --version v2 --horizon 10

Eingabe:  data/SYMBOL_H1_features.csv
Ausgabe:  data/SYMBOL_H1_labeled.csv       (v1 – Original)
          data/SYMBOL_H1_labeled_v2.csv    (v2 – Option A: Horizon=10)
          data/SYMBOL_H1_labeled_v3.csv    (v3 – Option B: TP/SL=0.15%)
"""

# pylint: disable=duplicate-code

# Standard-Bibliotheken
import argparse
import logging
from pathlib import Path

# Datenverarbeitung
import numpy as np
import pandas as pd

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("labeling.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Symbole
SYMBOLE = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
]

# ============================================================
# Labeling-Parameter (hier anpassen wenn gewünscht)
# ============================================================
TP_PCT = 0.003  # Take-Profit: 0.3% des Close-Preises (≈30 Pips für EURUSD)
SL_PCT = 0.003  # Stop-Loss:   0.3% symmetrisch (gleiche Distanz nach unten)
HORIZON = 5  # Zeitschranke: 5 H1-Kerzen = 5 Stunden voraus schauen


# ============================================================
# 1. Double-Barrier Labeling
# ============================================================


# pylint: disable=too-many-arguments,too-many-positional-arguments
def double_barrier_label(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    horizon: int = HORIZON,
) -> np.ndarray:
    """
    Berechnet Double-Barrier Labels für jede Kerze.

    Für jede Kerze T: Schaut in die nächsten `horizon` Kerzen.
        - Wenn High[T+j] >= close[T] * (1 + tp_pct) zuerst: Label 1  (Long)
        - Wenn Low[T+j]  <= close[T] * (1 - sl_pct) zuerst: Label -1 (Short)
        - Wenn weder noch bis Horizon:                        Label 0  (Nichts)

    Symmetrische Barrieren (tp_pct == sl_pct) sorgen für ausgewogene Klassen.

    Args:
        close: Close-Preise als numpy-Array
        high:  High-Preise als numpy-Array
        low:   Low-Preise als numpy-Array
        tp_pct: Take-Profit-Schwelle als Anteil (z.B. 0.003 = 0.3%)
        sl_pct: Stop-Loss-Schwelle als Anteil (z.B. 0.003 = 0.3%)
        horizon: Maximale Anzahl Kerzen voraus (z.B. 5)

    Returns:
        numpy-Array mit Labels (-1, 0, 1), letzte `horizon` Werte = NaN
    """
    n = len(close)
    # NaN als Platzhalter – letzte 'horizon' Kerzen werden NaN bleiben
    labels = np.full(n, np.nan)

    for i in range(n - horizon):
        # Barrieren basierend auf Close-Preis zum Zeitpunkt T
        tp_level = close[i] * (1.0 + tp_pct)  # Obere Schranke (Long-TP)
        sl_level = close[i] * (1.0 - sl_pct)  # Untere Schranke (Short-TP)

        label = 0  # Standard: keine Barriere getroffen → kein Signal

        # Vorwärts schauen: welche Schranke wird ZUERST getroffen?
        for j in range(1, horizon + 1):
            if high[i + j] >= tp_level:
                label = 1  # Kurs erreicht obere Schranke zuerst → Long
                break
            if low[i + j] <= sl_level:
                label = -1  # Kurs erreicht untere Schranke zuerst → Short
                break
            # Sonst: nächste Kerze prüfen

        labels[i] = label

    # Letzte 'horizon' Einträge bleiben NaN (kein vollständiger Vorausblick)
    return labels


# pylint: disable=too-many-arguments,too-many-positional-arguments
def double_barrier_label_rrr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    horizon: int,
) -> np.ndarray:
    """
    Korrekte Double-Barrier Labels für asymmetrisches RRR (z.B. 2:1).

    Im Gegensatz zur Standard-Funktion werden BEIDE Richtungen mit derselben
    tp_pct-Hürde geprüft:
        - Label  1 (Long):  Kurs steigt tp_pct VOR dem Fallen um sl_pct
        - Label -1 (Short): Kurs fällt  tp_pct VOR dem Steigen um sl_pct
        - Label  0 (Neutral): Weder Long noch Short wäre profitabel gewesen

    Warum das wichtig ist: Mit tp=0.6%, sl=0.3% (2:1) würde die Standard-Funktion
    Long selten labeln (+0.6% nötig) und Short oft labeln (-0.3% reicht) →
    starkes Klassenungleichgewicht. Diese Funktion ist symmetrisch.

    Args:
        close:   Close-Preise als numpy-Array
        high:    High-Preise als numpy-Array
        low:     Low-Preise als numpy-Array
        tp_pct:  Zielbewegung in BEIDE Richtungen (z.B. 0.006 = 0.6%)
        sl_pct:  Adverse Bewegung bis Stop-Loss (z.B. 0.003 = 0.3%)
        horizon: Maximale Anzahl Kerzen voraus

    Returns:
        numpy-Array mit Labels (-1, 0, 1), letzte `horizon` Werte = NaN
    """
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - horizon):
        # Barrieren für 2:1 RRR (BEIDE Richtungen mit demselben tp_pct)
        long_tp = close[i] * (1.0 + tp_pct)  # Long:  TP bei +tp_pct (z.B. +0.6%)
        long_sl = close[i] * (1.0 - sl_pct)  # Long:  SL bei -sl_pct (z.B. -0.3%)
        short_tp = close[i] * (1.0 - tp_pct)  # Short: TP bei -tp_pct (z.B. -0.6%)
        short_sl = close[i] * (1.0 + sl_pct)  # Short: SL bei +sl_pct (z.B. +0.3%)

        label = 0  # Standard: kein klares Signal

        for j in range(1, horizon + 1):
            h = high[i + j]
            l = low[i + j]

            # Long-TP zuerst prüfen (Kurs stieg tp_pct)
            if h >= long_tp:
                label = 1  # Long würde TP treffen → Long Signal
                break

            # Short-TP prüfen (Kurs fiel tp_pct)
            if l <= short_tp:
                label = -1  # Short würde TP treffen → Short Signal
                break

            # Adverse Bewegungen (SL-Level) → kein profitabler Trade möglich
            if l <= long_sl or h >= short_sl:
                label = 0  # Long-SL oder Short-SL getroffen → Kein Signal
                break

        labels[i] = label

    return labels


# ============================================================
# 2. Label-Verteilung analysieren
# ============================================================


def label_verteilung_pruefen(df: pd.DataFrame, symbol: str) -> None:
    """
    Zeigt die Verteilung der Labels im Log an.

    Warnt wenn eine Klasse < 10% vorkommt (starkes Ungleichgewicht).

    Args:
        df: DataFrame mit 'label'-Spalte
        symbol: Symbol-Name (für Logging)
    """
    gueltig = df["label"].dropna()
    anz = len(gueltig)
    verteilung = gueltig.value_counts().sort_index()

    logger.info("\n[%s] Label-Verteilung (%s gültige Kerzen):", symbol, f"{anz:,}")
    namen = {-1: "Short (-1)", 0: "Kein Signal (0)", 1: "Long  (+1)"}
    for label_nr in [-1, 0, 1]:
        anzahl = verteilung.get(label_nr, 0)
        anteil = anzahl / anz
        balken = "█" * int(anteil * 40)
        leer = "░" * (40 - int(anteil * 40))
        logger.info(
            "  %s: %s (%s%%) [%s%s]",
            f"{namen[label_nr]:15s}",
            f"{anzahl:6,}",
            f"{anteil * 100:5.1f}",
            balken,
            leer,
        )

    # Warnung bei starkem Ungleichgewicht
    for label_nr in [-1, 0, 1]:
        anteil = verteilung.get(label_nr, 0) / anz
        if anteil < 0.10:
            logger.warning(
                "[%s] Klasse %s nur %s%%! " "TP/SL-Schwellen oder Horizon anpassen?",
                symbol,
                label_nr,
                f"{anteil * 100:.1f}",
            )


# ============================================================
# 3. Labeling für ein Symbol
# ============================================================


def labeled_pfad(symbol: str, version: str = "v1") -> Path:
    """
    Gibt den Ausgabepfad für das gelabelte CSV zurück.

    v1 → data/SYMBOL_H1_labeled.csv        (Original, rückwärtskompatibel)
    v2 → data/SYMBOL_H1_labeled_v2.csv     (Option A: Horizon=10)
    v3 → data/SYMBOL_H1_labeled_v3.csv     (Option B: TP/SL=0.15%)

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")
        version: Versions-String (Standard: "v1")

    Returns:
        Path zum gelabelten CSV
    """
    if version == "v1":
        # Rückwärtskompatibel: kein Versions-Suffix für v1
        return DATA_DIR / f"{symbol}_H1_labeled.csv"
    return DATA_DIR / f"{symbol}_H1_labeled_{version}.csv"


def symbol_labeln(
    symbol: str,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    horizon: int = HORIZON,
    version: str = "v1",
    modus: str = "standard",
) -> bool:
    """
    Vollständiger Labeling-Ablauf für ein Symbol.

    Schritte:
        1. Feature-CSV laden
        2. Double-Barrier Labels berechnen
        3. Label-Verteilung analysieren
        4. Als *_labeled{_vN}.csv speichern

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        tp_pct:  Take-Profit-Schwelle (Standard: TP_PCT = 0.3%)
        sl_pct:  Stop-Loss-Schwelle   (Standard: SL_PCT = 0.3%)
        horizon: Zeitschranke in Kerzen (Standard: HORIZON = 5)
        version: Versions-String für den Ausgabe-Dateinamen (Standard: "v1")
        modus:   "standard" = original symmetrisch | "rrr" = korrekte RRR-Logik

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    eingabe_pfad = DATA_DIR / f"{symbol}_H1_features.csv"
    ausgabe_pfad = labeled_pfad(symbol, version)

    # Datei prüfen
    if not eingabe_pfad.exists():
        logger.error("[%s] Nicht gefunden: %s", symbol, eingabe_pfad)
        logger.error(
            "[%s] Zuerst feature_engineering.py und " "regime_detection.py ausführen!",
            symbol,
        )
        return False

    # Feature-CSV laden
    logger.info("[%s] Lade %s ...", symbol, eingabe_pfad.name)
    df = pd.read_csv(eingabe_pfad, index_col="time", parse_dates=True)
    logger.info("[%s] %s Kerzen, %s Features", symbol, f"{len(df):,}", len(df.columns))

    # Pflicht-Spalten prüfen
    for col in ["close", "high", "low"]:
        if col not in df.columns:
            logger.error("[%s] Fehlende Spalte: '%s'", symbol, col)
            return False

    # Double-Barrier Labels berechnen (Funktion abhängig vom Modus)
    if modus == "rrr":
        # Korrekte RRR-Logik: BEIDE Richtungen verwenden tp_pct als Ziel
        logger.info(
            "[%s] Berechne Labels (Modus=RRR, TP=%s, SL=%s, Horizon=%s Kerzen) ...",
            symbol,
            f"{tp_pct:.2%}",
            f"{sl_pct:.2%}",
            horizon,
        )
        logger.info(
            "[%s] RRR-Logik: Long=+%.2f%% vor -%.2f%% | Short=-%.2f%% vor +%.2f%%",
            symbol,
            tp_pct * 100,
            sl_pct * 100,
            tp_pct * 100,
            sl_pct * 100,
        )
        labels = double_barrier_label_rrr(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            horizon=horizon,
        )
    else:
        # Standard-Modus: symmetrische Barrieren (v1-kompatibel)
        logger.info(
            "[%s] Berechne Labels (Modus=Standard, TP=%s, SL=%s, Horizon=%s Kerzen) ...",
            symbol,
            f"{tp_pct:.2%}",
            f"{sl_pct:.2%}",
            horizon,
        )
        labels = double_barrier_label(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            horizon=horizon,
        )
    df["label"] = labels

    # Letzte horizon Zeilen ohne gültiges Label entfernen
    n_vorher = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)  # float → int (-1, 0, 1)
    logger.info("[%s] %s Zeilen ohne Label entfernt", symbol, n_vorher - len(df))

    # Label-Verteilung analysieren und prüfen
    label_verteilung_pruefen(df, symbol)

    # Als CSV speichern
    df.to_csv(ausgabe_pfad)
    groesse_mb = ausgabe_pfad.stat().st_size / 1024 / 1024
    logger.info(
        "[%s] Gespeichert: %s (%s Kerzen, %s Spalten, %s MB)",
        symbol,
        ausgabe_pfad.name,
        f"{len(df):,}",
        len(df.columns),
        f"{groesse_mb:.1f}",
    )

    return True


# ============================================================
# 4. Hauptprogramm
# ============================================================


def main() -> None:
    """Labeling für alle oder ausgewählte Forex-Paare.

    Unterstützt konfigurierbare Parameter via CLI-Argumente.
    """

    # ---- CLI-Argumente ----
    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – Double-Barrier Labeling"
    )
    parser.add_argument(
        "--symbol",
        default="alle",
        help=(
            "Handelssymbol oder 'alle' (Standard: alle). "
            "Mögliche Werte: "
            "EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD"
        ),
    )
    parser.add_argument(
        "--tp_pct",
        type=float,
        default=TP_PCT,
        help=(
            f"Take-Profit als Anteil (Standard: {TP_PCT} = "
            f"{TP_PCT:.2%}). Option B: 0.0015"
        ),
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=SL_PCT,
        help=(
            f"Stop-Loss als Anteil (Standard: {SL_PCT} = "
            f"{SL_PCT:.2%}). Option B: 0.0015"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help=f"Zeitschranke in H1-Kerzen (Standard: {HORIZON}). Option A: 10",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Versions-Suffix für den Ausgabe-Dateinamen (Standard: v1). "
            "v1 → SYMBOL_H1_labeled.csv (rückwärtskompatibel) | "
            "v2 → SYMBOL_H1_labeled_v2.csv | "
            "v3 → SYMBOL_H1_labeled_v3.csv"
        ),
    )
    parser.add_argument(
        "--modus",
        default="standard",
        choices=["standard", "rrr"],
        help=(
            "Labeling-Logik: "
            "'standard' = symmetrische Barrieren (v1-kompatibel, Standard) | "
            "'rrr' = korrekte RRR-Logik (beide Richtungen mit tp_pct als Ziel, "
            "für 2:1 oder 3:1 RRR empfohlen)"
        ),
    )
    args = parser.parse_args()

    # ---- Symbole bestimmen ----
    if args.symbol.lower() == "alle":
        ziel_symbole = SYMBOLE
    elif args.symbol.upper() in SYMBOLE:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: {args.symbol}")
        print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
        return

    logger.info("=" * 60)
    logger.info("Phase 4 – Labeling – gestartet")
    logger.info(
        "Parameter: TP=%s, SL=%s, Horizon=%s H1-Barren | Version: %s | Modus: %s",
        f"{args.tp_pct:.2%}",
        f"{args.sl_pct:.2%}",
        args.horizon,
        args.version,
        args.modus.upper(),
    )
    logger.info("Symbole: %s", ", ".join(ziel_symbole))
    logger.info("=" * 60)

    ergebnisse = []

    for symbol in ziel_symbole:
        logger.info("\n%s", "─" * 40)
        logger.info("Verarbeite: %s", symbol)
        logger.info("%s", "─" * 40)

        erfolg = symbol_labeln(
            symbol,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            horizon=args.horizon,
            version=args.version,
            modus=args.modus,
        )
        ergebnisse.append((symbol, "OK" if erfolg else "FEHLER"))

    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"ABGESCHLOSSEN – Labeling ({args.version}, Modus={args.modus.upper()})")
    print(
        f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | "
        f"Horizon={args.horizon} H1-Barren | RRR={args.tp_pct/args.sl_pct:.1f}:1"
    )
    print("=" * 60)
    for symbol, status in ergebnisse:
        zeichen = "✓" if status == "OK" else "✗"
        print(f"  {zeichen} {symbol}: {status}")

    erfolge = [r for r in ergebnisse if r[1] == "OK"]
    print(f"\n{len(erfolge)}/{len(ziel_symbole)} Symbole erfolgreich gelabelt.")

    # Ausgabe-Dateiname anzeigen
    beispiel_pfad = labeled_pfad("EURUSD", args.version)
    symbol_name = beispiel_pfad.name.replace("EURUSD", "SYMBOL")
    print(f"\nGelabelte Daten in: {DATA_DIR}/{symbol_name}")
    print(
        f"\nNächster Schritt: train_model.py --symbol alle "
        f"--version {args.version} ausführen"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
