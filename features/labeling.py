"""
labeling.py – Double-Barrier Labeling für das ML-Trading-System

Erstellt für jede H1-Kerze ein Label basierend auf der zukünftigen Kursbewegung:
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
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# ============================================================
# Labeling-Parameter (hier anpassen wenn gewünscht)
# ============================================================
TP_PCT = 0.003  # Take-Profit: 0.3% des Close-Preises (≈30 Pips für EURUSD)
SL_PCT = 0.003  # Stop-Loss:   0.3% symmetrisch (gleiche Distanz nach unten)
HORIZON = 5  # Zeitschranke: 5 H1-Kerzen = 5 Stunden voraus schauen


# ============================================================
# 1. Double-Barrier Labeling
# ============================================================


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
            elif low[i + j] <= sl_level:
                label = -1  # Kurs erreicht untere Schranke zuerst → Short
                break
            # Sonst: nächste Kerze prüfen

        labels[i] = label

    # Letzte 'horizon' Einträge bleiben NaN (kein vollständiger Vorausblick)
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

    logger.info(f"\n[{symbol}] Label-Verteilung ({anz:,} gültige Kerzen):")
    namen = {-1: "Short (-1)", 0: "Kein Signal (0)", 1: "Long  (+1)"}
    for label_nr in [-1, 0, 1]:
        anzahl = verteilung.get(label_nr, 0)
        anteil = anzahl / anz
        balken = "█" * int(anteil * 40)
        leer = "░" * (40 - int(anteil * 40))
        logger.info(
            f"  {namen[label_nr]:15s}: {anzahl:6,} ({anteil * 100:5.1f}%) [{balken}{leer}]"
        )

    # Warnung bei starkem Ungleichgewicht
    for label_nr in [-1, 0, 1]:
        anteil = verteilung.get(label_nr, 0) / anz
        if anteil < 0.10:
            logger.warning(
                f"[{symbol}] Klasse {label_nr} nur {anteil * 100:.1f}%! "
                f"TP/SL-Schwellen oder Horizon anpassen?"
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

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    eingabe_pfad = DATA_DIR / f"{symbol}_H1_features.csv"
    ausgabe_pfad = labeled_pfad(symbol, version)

    # Datei prüfen
    if not eingabe_pfad.exists():
        logger.error(f"[{symbol}] Nicht gefunden: {eingabe_pfad}")
        logger.error(
            f"[{symbol}] Zuerst feature_engineering.py und regime_detection.py ausführen!"
        )
        return False

    # Feature-CSV laden
    logger.info(f"[{symbol}] Lade {eingabe_pfad.name} ...")
    df = pd.read_csv(eingabe_pfad, index_col="time", parse_dates=True)
    logger.info(f"[{symbol}] {len(df):,} Kerzen, {len(df.columns)} Features")

    # Pflicht-Spalten prüfen
    for col in ["close", "high", "low"]:
        if col not in df.columns:
            logger.error(f"[{symbol}] Fehlende Spalte: '{col}'")
            return False

    # Double-Barrier Labels berechnen (mit den übergebenen Parametern)
    logger.info(
        f"[{symbol}] Berechne Labels (TP={tp_pct:.2%}, SL={sl_pct:.2%}, "
        f"Horizon={horizon} Kerzen) ..."
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
    logger.info(f"[{symbol}] {n_vorher - len(df)} Zeilen ohne Label entfernt")

    # Label-Verteilung analysieren und prüfen
    label_verteilung_pruefen(df, symbol)

    # Als CSV speichern
    df.to_csv(ausgabe_pfad)
    groesse_mb = ausgabe_pfad.stat().st_size / 1024 / 1024
    logger.info(
        f"[{symbol}] Gespeichert: {ausgabe_pfad.name} "
        f"({len(df):,} Kerzen, {len(df.columns)} Spalten, {groesse_mb:.1f} MB)"
    )

    return True


# ============================================================
# 4. Hauptprogramm
# ============================================================


def main() -> None:
    """Labeling für alle oder ausgewählte Forex-Paare (mit konfigurierbaren Parametern)."""

    # ---- CLI-Argumente ----
    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – Double-Barrier Labeling"
    )
    parser.add_argument(
        "--symbol",
        default="alle",
        help=(
            "Handelssymbol oder 'alle' (Standard: alle). "
            "Mögliche Werte: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD"
        ),
    )
    parser.add_argument(
        "--tp_pct",
        type=float,
        default=TP_PCT,
        help=f"Take-Profit als Anteil (Standard: {TP_PCT} = {TP_PCT:.2%}). Option B: 0.0015",
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=SL_PCT,
        help=f"Stop-Loss als Anteil (Standard: {SL_PCT} = {SL_PCT:.2%}). Option B: 0.0015",
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
        f"Parameter: TP={args.tp_pct:.2%}, SL={args.sl_pct:.2%}, "
        f"Horizon={args.horizon} H1-Barren | Version: {args.version}"
    )
    logger.info(f"Symbole: {', '.join(ziel_symbole)}")
    logger.info("=" * 60)

    ergebnisse = []

    for symbol in ziel_symbole:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Verarbeite: {symbol}")
        logger.info(f"{'─' * 40}")

        erfolg = symbol_labeln(
            symbol,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            horizon=args.horizon,
            version=args.version,
        )
        ergebnisse.append((symbol, "OK" if erfolg else "FEHLER"))

    # Zusammenfassung
    print("\n" + "=" * 60)
    print(f"ABGESCHLOSSEN – Labeling ({args.version})")
    print(f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | Horizon={args.horizon} H1-Barren")
    print("=" * 60)
    for symbol, status in ergebnisse:
        zeichen = "✓" if status == "OK" else "✗"
        print(f"  {zeichen} {symbol}: {status}")

    erfolge = [r for r in ergebnisse if r[1] == "OK"]
    print(f"\n{len(erfolge)}/{len(ziel_symbole)} Symbole erfolgreich gelabelt.")

    # Ausgabe-Dateiname anzeigen
    beispiel_pfad = labeled_pfad("EURUSD", args.version)
    print(f"\nGelabelte Daten in: {DATA_DIR}/{beispiel_pfad.name.replace('EURUSD', 'SYMBOL')}")
    print(f"\nNächster Schritt: train_model.py --symbol alle --version {args.version} ausführen")
    print("=" * 60)


if __name__ == "__main__":
    main()
