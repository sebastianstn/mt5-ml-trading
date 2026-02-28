"""
backtest.py – Realistische Backtesting-Simulation des ML-Trading-Systems

Simuliert das System auf dem heiligen Test-Set (2023–2026) und bewertet ob
das Modell in echten, ungesehenen Marktbedingungen funktioniert.

Ablauf pro Kerze:
    1. Features aus dem gelabelten CSV laden
    2. Modell-Vorhersage + Wahrscheinlichkeit berechnen
    3. Nur traden wenn Wahrscheinlichkeit > Schwellenwert
    4. Trade simulieren: TP/SL/Horizon wie beim Labeling
    5. Spread einrechnen (realistischer Kostenfaktor)
    6. P&L pro Trade berechnen

Kennzahlen:
    - Gesamtrendite (%), Sharpe Ratio, Max. Drawdown, Gewinnfaktor, Win-Rate
    - Performance aufgeschlüsselt nach Market-Regime
    - Monatliche Returns-Heatmap

WARNUNG: Das Test-Set wird hier zum ersten Mal verwendet!
    Das ist die FINALE Bewertung des Systems.
    Wenn die Ergebnisse enttäuschen, das Modell NICHT mehr anpassen!
    (→ das wäre Look-Ahead-Bias auf Modellebene)

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python backtest/backtest.py [--symbol EURUSD] [--schwelle 0.55]
    python backtest/backtest.py --symbol alle
    python backtest/backtest.py --symbol alle --tp_pct 0.006 --sl_pct 0.003
    python backtest/backtest.py --symbol alle --regime_filter 1,2

    # Dynamische Positionsgrößenberechnung (1%-Risiko-Regel):
    python backtest/backtest.py --symbol EURUSD --kapital 10000 --risiko_pct 0.01

    # ATR-basiertes dynamisches Stop-Loss (1.5× ATR_14):
    python backtest/backtest.py --symbol EURUSD --atr_sl --atr_faktor 1.5

    # Beides kombiniert + Zeitraum-Filter (nur 2024):
    python backtest/backtest.py --symbol alle --kapital 10000 --atr_sl \\
        --zeitraum_von 2024-01-01 --zeitraum_bis 2024-12-31

Eingabe:  models/lgbm_SYMBOL_v1.pkl
          data/SYMBOL_H1_labeled.csv
Ausgabe:  plots/SYMBOL_backtest_equity.png      ← Equity-Kurve
          plots/SYMBOL_backtest_regime.png      ← Performance nach Regime
          plots/SYMBOL_backtest_monatlich.png   ← Monatliche Returns-Heatmap
          plots/SYMBOL_backtest_perioden.png    ← Jahresvergleich
          plots/regime_performance_matrix.png  ← NEU: Regime-Vergleich alle Symbole
          backtest/SYMBOL_trades.csv            ← Alle Trades als CSV
          backtest/backtest_zusammenfassung.csv ← Kennzahlen aller Symbole
          backtest/regime_performance_matrix.csv ← NEU: Regime-Performance-Daten
"""

# pylint: disable=too-many-lines,logging-fstring-interpolation

# Standard-Bibliotheken
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

# Datenverarbeitung
import numpy as np
import pandas as pd

# Modell laden
import joblib

# Visualisierung
import matplotlib

matplotlib.use("Agg")  # Kein Fenster öffnen – nur PNG speichern
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position
import seaborn as sns  # noqa: E402  # pylint: disable=wrong-import-position

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest/backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
BACKTEST_DIR = BASE_DIR / "backtest"

# Symbole (alle 7 Forex-Hauptpaare)
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# ============================================================
# Simulation-Parameter
# ============================================================

# Labeling-Parameter (MÜSSEN mit features/labeling.py übereinstimmen!)
TP_PCT = 0.003  # Take-Profit: 0.3% (identisch mit labeling.py)
SL_PCT = 0.003  # Stop-Loss:   0.3% (identisch mit labeling.py)
HORIZON = 5  # Zeitschranke: 5 H1-Kerzen

# Spread-Kosten pro Trade (als Anteil des Eintrittspreises)
# Realistisch: ~1–2 Pips je nach Symbol und Broker
# 1 Pip EURUSD ≈ 0.0001 → 0.01% des Preises → 1 Pip / 1.1000 ≈ 0.009%
SPREAD_KOSTEN = {
    "EURUSD": 0.000100,  # ~1 Pip
    "GBPUSD": 0.000120,  # ~1.2 Pips (etwas breiter)
    "USDJPY": 0.012000,  # ~1.2 Pips (JPY-Paare in anderen Einheiten)
    "AUDUSD": 0.000130,  # ~1.3 Pips
    "USDCAD": 0.000140,  # ~1.4 Pips
    "USDCHF": 0.000130,  # ~1.3 Pips
    "NZDUSD": 0.000160,  # ~1.6 Pips (Minor, etwas teurer)
}

# Swap-Kosten (Overnight-Gebühr) als Anteil des Positionswerts pro Overnight-Übertragung
# Typische IC Markets-Raten: Long = du hältst Basis-Währung über Nacht
# Positive Werte = du erhältst Swap (selten), negative = du zahlst Gebühr (häufig)
# Quelle: typische Retail-Broker-Werte für Standard-Lot (1 Pip ≈ 0.0001 EURUSD)
SWAP_KOSTEN_LONG = {
    "EURUSD": -0.000053,  # ~-0.53 Pips/Tag (Long EUR = Short USD-Zins)
    "GBPUSD": -0.000042,  # Long GBP, Short USD
    "USDJPY": +0.000012,  # Long USD-Zins > JPY-Zins → leicht positiv
    "AUDUSD": -0.000038,  # Long AUD
    "USDCAD": -0.000021,  # Long USD
    "USDCHF": -0.000031,  # Long USD, Short CHF
    "NZDUSD": -0.000045,  # Long NZD
}
SWAP_KOSTEN_SHORT = {
    "EURUSD": -0.000010,  # Short EURUSD (Short EUR, Long USD)
    "GBPUSD": -0.000015,  # Short GBP
    "USDJPY": -0.000065,  # Short USD → verlierst USD-Zins → teuer
    "AUDUSD": -0.000018,  # Short AUD
    "USDCAD": +0.000005,  # Short USD, Long CAD-Zins → leicht positiv
    "USDCHF": -0.000008,  # Short USD
    "NZDUSD": -0.000022,  # Short NZD
}

# Test-Zeitraum (heiliges Test-Set – nur einmal verwenden!)
TEST_VON = "2023-01-01"

# Klassen-Namen (für Plots und Berichte)
KLASSEN_NAMEN = {0: "Short", 1: "Neutral", 2: "Long"}

# Regime-Namen (aus regime_detection.py)
REGIME_NAMEN = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

# Gleiche Ausschluss-Spalten wie in train_model.py
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
# Risikomanagement-Konfiguration
# ============================================================


@dataclass
class RisikoConfig:
    """
    Risikomanagement-Parameter für die Trade-Simulation.

    Kapselt zwei erweiterte Risiko-Features:
    1. Dynamische Positionsgrößenberechnung (1%-Risiko-Regel):
       Lot-Größe = (Kapital × Risiko%) / (SL% × Kontrakt-Größe)
       → Bei 10.000€ Kapital, 1% Risiko, 0.3% SL:
         Lots = 100 / (0.003 × 100.000) ≈ 0.33 Lots

    2. ATR-basiertes dynamisches Stop-Loss:
       SL = ATR_14 × atr_faktor  (statt fixer 0.3%)
       → Passt sich der aktuellen Marktvolatilität an
       → In ruhigen Märkten: kleiner SL (weniger Risiko)
       → In volatilen Märkten: größerer SL (bleibt im Trade)
    """

    # --- Positionsgrößenberechnung ---
    kapital: float = 10_000.0
    """Startkapital in EUR/USD (Standard: 10.000)."""
    risiko_pct: float = 0.01
    """Max. Risiko pro Trade als Anteil (Standard: 1% = 0.01)."""
    kontrakt_groesse: float = 100_000.0
    """Kontraktgröße 1 Lot (Standard: 100.000 Einheiten bei Forex)."""

    # --- ATR-basiertes Stop-Loss ---
    atr_sl: bool = False
    """ATR-basiertes SL aktivieren (Standard: deaktiviert → fixer SL%)."""
    atr_faktor: float = 1.5
    """ATR-Multiplikator für SL (Standard: 1.5 × ATR_14)."""


# ============================================================
# 1. Daten laden
# ============================================================


def labeled_pfad(
    symbol: str, version: str = "v1", timeframe: str = "H1"
) -> Path:
    """
    Gibt den Pfad zum gelabelten CSV zurück (konsistent mit anderen Skripten).

    H1 (Standard):
        v1 → data/SYMBOL_H1_labeled.csv        (Original, rückwärtskompatibel)
        v2 → data/SYMBOL_H1_labeled_v2.csv
    H4:
        v1 → data/SYMBOL_H4_labeled.csv
        v2 → data/SYMBOL_H4_labeled_v2.csv

    Args:
        symbol:    Handelssymbol (z.B. "EURUSD")
        version:   Versions-String (Standard: "v1")
        timeframe: Zeitrahmen – "H1" oder "H4" (Standard: "H1")

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
    symbol: str,
    version: str = "v1",
    zeitraum_von: Optional[str] = None,
    zeitraum_bis: Optional[str] = None,
    timeframe: str = "H1",
) -> pd.DataFrame:
    """
    Lädt das gelabelte Feature-CSV und isoliert das Test-Set (2023+).

    Das Test-Set enthält sowohl die Features für das Modell als auch
    die Rohdaten (OHLC) für die Trade-Simulation.

    Optional kann ein Teilzeitraum des Test-Sets gefiltert werden.
    So lassen sich verschiedene Marktphasen separat analysieren.

    Args:
        symbol:       Handelssymbol (z.B. "EURUSD")
        version:      Versions-String für den Datei-Pfad (Standard: "v1")
        zeitraum_von: Optional – Startdatum für den Teilzeitraum (z.B. "2023-01-01")
        zeitraum_bis: Optional – Enddatum für den Teilzeitraum (z.B. "2023-12-31")
        timeframe:    Zeitrahmen der Daten – "H1" oder "H4" (Standard: "H1")

    Returns:
        DataFrame mit Features, OHLC-Preisen, label und market_regime.

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

    logger.info(f"[{symbol}] Lade {pfad.name} ...")
    df = pd.read_csv(pfad, index_col="time", parse_dates=True)

    # Test-Set isolieren (ab TEST_VON = 2023-01-01)
    df_test = df[df.index >= TEST_VON].copy()

    if len(df_test) == 0:
        raise ValueError(
            f"[{symbol}] Keine Daten ab {TEST_VON}! "
            f"Datei endet am {df.index[-1].date()}"
        )

    # Optionaler Teilzeitraum (für Perioden-Vergleich)
    if zeitraum_von:
        df_test = df_test[df_test.index >= zeitraum_von]
    if zeitraum_bis:
        df_test = df_test[df_test.index <= zeitraum_bis]

    if len(df_test) == 0:
        raise ValueError(
            f"[{symbol}] Keine Daten im Zeitraum "
            f"{zeitraum_von or TEST_VON} – {zeitraum_bis or 'heute'}!"
        )

    logger.info(
        f"[{symbol}] Test-Set: {len(df_test):,} Kerzen | "
        f"{df_test.index[0].date()} bis {df_test.index[-1].date()}"
    )
    return df_test


# ============================================================
# 2. Modell laden und Signale generieren
# ============================================================


def signale_generieren(  # pylint: disable=too-many-locals
    df: pd.DataFrame,
    symbol: str,
    schwelle: float = 0.55,
    version: str = "v1",
    timeframe: str = "H1",
) -> pd.DataFrame:
    """
    Lädt das LightGBM-Modell und generiert Trade-Signale mit Wahrscheinlichkeit.

    Nur Signale über dem Schwellenwert werden als Trade gewertet:
    - Long-Signal:  Modell sagt Klasse 2 UND prob_long  > schwelle
    - Short-Signal: Modell sagt Klasse 0 UND prob_short > schwelle
    - Kein Signal:  Klasse 1 (Neutral) ODER Wahrscheinlichkeit zu niedrig

    Modell-Dateiname:
        H1: lgbm_SYMBOL_v1.pkl      (Standard)
        H4: lgbm_SYMBOL_H4_v1.pkl  (H4-Experiment)

    Args:
        df:        Test-Set DataFrame mit Features und OHLC
        symbol:    Handelssymbol
        schwelle:  Mindest-Wahrscheinlichkeit für einen Trade (Standard: 0.55)
        version:   Versions-String für das Modell-File (Standard: "v1")
        timeframe: Zeitrahmen für den Modell-Dateinamen – "H1" oder "H4" (Standard: "H1")

    Returns:
        DataFrame mit zusätzlichen Spalten: signal, prob_signal
    """
    # Modell laden (versioniertes Modell-File, abhängig vom Zeitrahmen)
    if timeframe == "H4":
        modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_H4_{version}.pkl"
    else:
        modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{version}.pkl"

    if not modell_pfad.exists():
        hilfe = (
            f"train_model.py --symbol {symbol} --timeframe {timeframe}"
        )
        raise FileNotFoundError(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Zuerst {hilfe} ausführen!"
        )
    logger.info(f"[{symbol}] Lade Modell: {modell_pfad.name}")
    modell = joblib.load(modell_pfad)

    # Features aufbereiten (gleiche Spalten wie beim Training)
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    x_features = df[feature_spalten].copy()

    # NaN-Werte mit Median auffüllen (Sicherheitsnetz)
    nan_anzahl = x_features.isna().sum().sum()
    if nan_anzahl > 0:
        logger.warning(f"[{symbol}] {nan_anzahl} NaN-Werte – werden mit Median gefüllt")
        x_features = x_features.fillna(x_features.median())

    logger.info(f"[{symbol}] Berechne Vorhersagen für {len(x_features):,} Kerzen ...")

    # Wahrscheinlichkeiten für alle 3 Klassen
    # proba[:,0] = Short-Wahrscheinlichkeit
    # proba[:,1] = Neutral-Wahrscheinlichkeit
    # proba[:,2] = Long-Wahrscheinlichkeit
    proba = modell.predict_proba(x_features)

    # Rohe Vorhersage (Klasse mit höchster Wahrscheinlichkeit)
    raw_pred = np.argmax(proba, axis=1)

    # Signal mit Schwellenwert-Filter:
    # Nur handeln wenn die Wahrscheinlichkeit der vorhergesagten Richtung > schwelle
    signal = np.zeros(len(df), dtype=int)  # 0 = kein Trade
    prob_signal = np.zeros(len(df))  # Wahrscheinlichkeit des Signals

    for i in range(len(df)):
        if raw_pred[i] == 2 and proba[i, 2] > schwelle:
            signal[i] = 2  # Long-Signal
            prob_signal[i] = proba[i, 2]
        elif raw_pred[i] == 0 and proba[i, 0] > schwelle:
            signal[i] = 0  # Short-Signal (!)
            # Hinweis: 0 = kein Trade und 0 = Short sind mehrdeutig
            # Lösung: Long=2, Short=-1, Kein Trade=1
            signal[i] = -1  # Short-Signal (eindeutige Kodierung!)
            prob_signal[i] = proba[i, 0]

    # Umkodierung für Klarheit: 2=Long, -1=Short, 0=Kein Trade
    # signal ist schon korrekt kodiert (Long=2, Short=-1, KeinTrade=0)

    df = df.copy()
    df["signal"] = signal
    df["prob_signal"] = prob_signal

    # Signal-Statistik
    n_long = (signal == 2).sum()
    n_short = (signal == -1).sum()
    n_gesamt = len(signal)
    logger.info(
        f"[{symbol}] Signale (Schwelle={schwelle:.0%}): "
        f"Long={n_long} ({n_long / n_gesamt:.1%}) | "
        f"Short={n_short} ({n_short / n_gesamt:.1%}) | "
        f"Kein Trade={n_gesamt - n_long - n_short} ({(n_gesamt - n_long - n_short) / n_gesamt:.1%})"
    )

    return df


# ============================================================
# 3. Einzelnen Trade simulieren
# ============================================================


def trade_simulieren(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    df: pd.DataFrame,
    eintritts_index: int,
    richtung: int,
    spread_kosten: float,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    horizon: int = HORIZON,
    risiko_config: Optional[RisikoConfig] = None,
    atr_wert: float = 0.0,
    swap_aktiv: bool = False,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    entry_time: Optional[pd.Timestamp] = None,
    stunden_pro_bar: int = 1,
) -> dict:
    """
    Simuliert einen einzelnen Trade mit Double-Barrier Exit.

    Schritte:
        1. Eintrittspreis = close[T] (Market Order auf Kerzenschluss)
        2. TP/SL-Level berechnen (fix oder ATR-basiert via risiko_config)
        3. Nächste horizon Kerzen prüfen: welche Schranke wird ZUERST getroffen?
        4. P&L berechnen (inkl. Spread-Kosten)
        5. Lot-Größe und absoluten P&L berechnen (wenn kapital > 0)

    Asymmetrisches TP/SL: z.B. tp_pct=0.006, sl_pct=0.003 → RRR=2:1
    → Bei 40% Win-Rate schon profitabel: 0.4×0.6% − 0.6×0.3% = +0.06% p.Trade

    ATR-basiertes SL (wenn risiko_config.atr_sl=True und atr_wert>0):
        SL = ATR_14 × atr_faktor / eintrittspreis
        TP = SL × (tp_pct / sl_pct)  → gleiche RRR wie bei festem SL

    Positionsgrößenberechnung (wenn risiko_config.kapital > 0):
        Lots = (kapital × risiko_pct) / (sl_pct_aktiv × kontrakt_groesse)
        pnl_absolut = pnl_pct × lots × kontrakt_groesse

    Args:
        df:              Test-Set DataFrame (mit OHLC und index)
        eintritts_index: Integer-Position des Eintrittsbalkens
        richtung:        2=Long, -1=Short
        spread_kosten:   Spread als Anteil des Preises (z.B. 0.0001)
        tp_pct:          Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:          Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        horizon:         Zeitschranke in Kerzen (Standard: HORIZON = 5)
        risiko_config:   Risikomanagement-Konfiguration (Standard: None = einfacher Modus)
        atr_wert:        ATR_14-Wert dieser Kerze in Preiseinheiten (Standard: 0.0)
        swap_aktiv:      Swap-Kosten aktivieren (Standard: False)
        swap_long:       Swap-Satz für Long-Positionen (als Anteil, z.B. -0.000053)
        swap_short:      Swap-Satz für Short-Positionen (als Anteil, z.B. -0.000010)
        entry_time:      Eintrittszeitpunkt (pd.Timestamp) für Mitternacht-Prüfung

    Returns:
        Dict mit Trade-Ergebnis: pnl_pct, exit_grund, n_bars, eintrittspreis,
        sl_pct_verwendet, lot_groesse, pnl_absolut, hat_swap
    """
    n = len(df)

    # Eintrittspreis (Close der Signal-Kerze)
    eintrittspreis = df["close"].iloc[eintritts_index]

    # ────────────────────────────────────────────────────────
    # Feature 2: ATR-basiertes dynamisches Stop-Loss
    # ────────────────────────────────────────────────────────
    cfg = risiko_config
    if cfg and cfg.atr_sl and atr_wert > 0 and eintrittspreis > 0:
        # SL dynamisch: ATR × Faktor → in Preiseinheiten → umgerechnet in %
        sl_abs = atr_wert * cfg.atr_faktor
        sl_pct_aktiv = sl_abs / eintrittspreis
        # TP hält die gleiche RRR wie beim festen TP/SL-Verhältnis
        rrr = tp_pct / sl_pct if sl_pct > 0 else 1.0
        tp_pct_aktiv = sl_pct_aktiv * rrr
    else:
        # Festes TP/SL (Standard-Modus)
        sl_pct_aktiv = sl_pct
        tp_pct_aktiv = tp_pct

    # TP/SL-Level berechnen (mit aktivem sl_pct_aktiv)
    if richtung == 2:  # Long
        tp_level = eintrittspreis * (1.0 + tp_pct_aktiv)  # Obere Schranke (Ziel)
        sl_level = eintrittspreis * (1.0 - sl_pct_aktiv)  # Untere Schranke (Stop)
    else:  # Short
        tp_level = eintrittspreis * (1.0 - tp_pct_aktiv)  # Untere Schranke (Ziel)
        sl_level = eintrittspreis * (1.0 + sl_pct_aktiv)  # Obere Schranke (Stop)

    # Vorwärts schauen: nächste horizon Kerzen
    austritt_pnl = None
    exit_grund = "horizon"  # Standard: Zeitschranke erreicht
    n_bars = horizon

    for j in range(1, horizon + 1):
        idx = eintritts_index + j
        if idx >= n:
            # Ende des Datensatzes erreicht
            n_bars = j - 1
            break

        high_j = df["high"].iloc[idx]
        low_j = df["low"].iloc[idx]

        if richtung == 2:  # Long
            if high_j >= tp_level:
                # Take-Profit getroffen → Gewinn = TP-Abstand
                austritt_pnl = tp_pct_aktiv
                exit_grund = "tp"
                n_bars = j
                break
            if low_j <= sl_level:
                # Stop-Loss getroffen → Verlust = SL-Abstand
                austritt_pnl = -sl_pct_aktiv
                exit_grund = "sl"
                n_bars = j
                break
        else:  # Short
            if low_j <= tp_level:
                # Take-Profit getroffen (Short) → Gewinn
                austritt_pnl = tp_pct_aktiv
                exit_grund = "tp"
                n_bars = j
                break
            if high_j >= sl_level:
                # Stop-Loss getroffen (Short) → Verlust
                austritt_pnl = -sl_pct_aktiv
                exit_grund = "sl"
                n_bars = j
                break

    # Wenn kein TP/SL getroffen: Austritt zum Close der letzten horizon-Kerze
    if austritt_pnl is None:
        idx_exit = min(eintritts_index + horizon, n - 1)
        austrittspreis = df["close"].iloc[idx_exit]
        if richtung == 2:  # Long
            austritt_pnl = (austrittspreis - eintrittspreis) / eintrittspreis
        else:  # Short
            austritt_pnl = (eintrittspreis - austrittspreis) / eintrittspreis
        exit_grund = "horizon"

    # Spread-Kosten abziehen (Eintritt + Austritt = 2× Spread)
    spread_als_pct = (
        (spread_kosten / eintrittspreis) if eintrittspreis > 10 else spread_kosten
    )
    netto_pnl = austritt_pnl - 2 * spread_als_pct

    # ────────────────────────────────────────────────────────
    # Swap-Kosten (Review-Punkt 5): Overnight-Gebühr bei Mitternacht-Überschreitung
    # ────────────────────────────────────────────────────────
    # H1: Ein Trade mit Eintrittsstunde H läuft n_bars Stunden.
    #     Wenn H + n_bars >= 24, überschreitet er Mitternacht.
    # H4: Jede Kerze = 4 Stunden. Laufzeit = n_bars × stunden_pro_bar.
    #     Wenn H + n_bars*4 >= 24, überschreitet der Trade Mitternacht.
    # Der Broker berechnet den Swap bei jeder Mitternacht-Überschreitung.
    hat_swap = False
    if swap_aktiv and entry_time is not None:
        entry_hour = pd.Timestamp(entry_time).hour
        # Gesamte Trade-Dauer in Stunden (H1: 1h/Bar, H4: 4h/Bar)
        trade_stunden = n_bars * stunden_pro_bar
        # Prüfen ob der Trade über Mitternacht läuft
        kreuzt_mitternacht = (entry_hour + trade_stunden) >= 24
        if kreuzt_mitternacht:
            # Long-Swap für Long-Positionen, Short-Swap für Short-Positionen
            swap_satz = swap_long if richtung == 2 else swap_short
            # abs(): beide Swap-Sätze werden als Kosten abgezogen (meist negativ)
            netto_pnl -= abs(swap_satz)
            hat_swap = True

    # ────────────────────────────────────────────────────────
    # Feature 1: Dynamische Positionsgrößenberechnung
    # ────────────────────────────────────────────────────────
    lot_groesse = 0.0
    pnl_absolut = 0.0
    if cfg and cfg.kapital > 0 and sl_pct_aktiv > 0:
        # Risikobetrag in EUR/USD (z.B. 10.000 × 1% = 100)
        risikobetrag = cfg.kapital * cfg.risiko_pct
        # SL-Betrag pro Lot = sl_pct × Eintrittspreis × Kontraktgröße
        sl_pro_lot = sl_pct_aktiv * cfg.kontrakt_groesse
        # Lot-Größe: wie viele Lots können wir handeln ohne mehr als risikobetrag zu riskieren?
        lot_groesse = risikobetrag / sl_pro_lot if sl_pro_lot > 0 else 0.01
        # Absoluter P&L = prozentualer P&L × Lots × Kontraktgröße
        pnl_absolut = netto_pnl * lot_groesse * cfg.kontrakt_groesse

    return {
        "pnl_pct": netto_pnl,
        "exit_grund": exit_grund,
        "n_bars": n_bars,
        "eintrittspreis": eintrittspreis,
        "sl_pct_verwendet": round(sl_pct_aktiv * 100, 4),  # in % für Trade-Log
        "lot_groesse": round(lot_groesse, 4),
        "pnl_absolut": round(pnl_absolut, 4),
        "hat_swap": hat_swap,  # True wenn Overnight-Swap-Kosten angefallen
    }


# ============================================================
# 4. Alle Trades für ein Symbol simulieren
# ============================================================


def trades_simulieren(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    df: pd.DataFrame,
    symbol: str,
    schwelle: float = 0.55,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    regime_erlaubt: Optional[list] = None,
    horizon: int = HORIZON,
    risiko_config: Optional[RisikoConfig] = None,
    spread_faktor: float = 1.0,
    swap_aktiv: bool = False,
    timeframe: str = "H1",
) -> pd.DataFrame:
    """
    Simuliert alle Trades auf dem Test-Set und gibt eine Trade-Liste zurück.

    Überlappende Trades werden vermieden: Nach einem Trade wird die nächste
    Signal-Kerze erst nach horizon Kerzen gesucht (no re-entry während Trade).

    Verbesserungs-Parameter:
        tp_pct / sl_pct:  Asymmetrisches TP/SL für besseres Chance-Risiko-Verhältnis.
                          Beispiel: tp=0.006, sl=0.003 → RRR=2:1 (nur 40% Win-Rate nötig)
        regime_erlaubt:   Nur in bestimmten Regimes handeln.
                          [1,2] = nur Aufwärts-/Abwärtstrend
        risiko_config:    Dynamisches SL und Positionsgrößenberechnung (optional).
        spread_faktor:    Multiplikator für Spread-Kosten (Standard: 1.0 = real).
                          2.0 = Transaction Cost Sensitivity Test (verdoppelter Spread).

    Args:
        df:             Test-Set DataFrame mit Signalen (aus signale_generieren())
        symbol:         Handelssymbol (für Spread-Kosten und Logging)
        schwelle:       Wahrscheinlichkeits-Schwellenwert (für Logging)
        tp_pct:         Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:         Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        regime_erlaubt: Liste erlaubter Regime-Nummern (None = alle Regimes handeln)
        horizon:        Zeitschranke in Kerzen (Standard: HORIZON = 5)
        risiko_config:  Risikomanagement-Konfiguration (Standard: None)
        spread_faktor:  Spread-Multiplikator für Kosten-Stress-Test (Standard: 1.0)
        swap_aktiv:     Swap-Kosten (Overnight-Gebühr) aktivieren (Standard: False)

    Returns:
        DataFrame mit allen simulierten Trades.
    """
    # Spread-Kosten mit optionalem Multiplikator für Sensitivity Test
    spread_kosten = SPREAD_KOSTEN.get(symbol, 0.000150) * spread_faktor  # Fallback: 1.5 Pips
    if spread_faktor != 1.0:
        logger.info(
            f"[{symbol}] ⚡ Spread-Stress-Test: Faktor={spread_faktor}× "
            f"→ Spread={spread_kosten:.6f} "
            f"(statt {SPREAD_KOSTEN.get(symbol, 0.000150):.6f})"
        )

    # Regime-Spalte und ATR-Spalte vorbereiten (falls vorhanden)
    hat_regime = "market_regime" in df.columns
    # atr_14 ist in AUSSCHLUSS_SPALTEN (nicht für Modell), aber im CSV vorhanden
    hat_atr = "atr_14" in df.columns

    # ATR-SL aktiv? → Nur wenn Config gesetzt und Spalte vorhanden
    cfg = risiko_config
    atr_sl_aktiv = cfg is not None and cfg.atr_sl and hat_atr

    if atr_sl_aktiv:
        logger.info(
            f"[{symbol}] ATR-SL aktiv: SL = ATR_14 × "  # type: ignore[union-attr]
            f"{cfg.atr_faktor:.1f} (statt festem SL={sl_pct:.2%})"
        )
    if cfg and cfg.kapital > 0:
        logger.info(
            f"[{symbol}] Positionsgrößenberechnung: Kapital={cfg.kapital:,.0f} | "
            f"Risiko={cfg.risiko_pct:.0%} pro Trade | "
            f"Kontrakt={cfg.kontrakt_groesse:,.0f}"
        )

    # Swap-Sätze aus den globalen Dicts für dieses Symbol holen
    swap_long_satz = SWAP_KOSTEN_LONG.get(symbol, -0.000030)   # Fallback: -3 Pips
    swap_short_satz = SWAP_KOSTEN_SHORT.get(symbol, -0.000015)  # Fallback: -1.5 Pips
    if swap_aktiv:
        logger.info(
            f"[{symbol}] Swap-Kosten aktiv: "
            f"Long={swap_long_satz:.6f} | Short={swap_short_satz:.6f} "
            f"(Overnight-Gebühr bei Mitternacht-Überschreitung)"
        )

    trades = []  # Ergebnis-Liste
    n_gefiltert_regime = 0  # Zähler: übersprungene Trades wegen Regime-Filter
    n_swap_trades = 0       # Zähler: Trades mit Swap-Kosten
    i = 0  # Aktueller Balken-Index

    while i < len(df) - horizon:
        signal = df["signal"].iloc[i]

        # Nur Long (2) oder Short (-1) Signale handeln
        if signal in (2, -1):

            # Regime-Filter: Signal überspringen wenn aktuelles Regime nicht erlaubt
            if regime_erlaubt is not None and hat_regime:
                aktuelles_regime = df["market_regime"].iloc[i]
                if (
                    not np.isnan(aktuelles_regime)
                    and int(aktuelles_regime) not in regime_erlaubt
                ):
                    # Regime nicht erlaubt → Signal ignorieren
                    n_gefiltert_regime += 1
                    i += 1
                    continue

            # ATR-Wert dieser Kerze (für dynamisches SL)
            atr_wert = float(df["atr_14"].iloc[i]) if atr_sl_aktiv else 0.0

            # Trade simulieren (mit konfigurierbarem TP/SL, ATR, Positionsgröße und Swap)
            # stunden_pro_bar: 1 für H1 (Standard), 4 für H4 (für Swap-Kostenberechnung)
            stunden_pro_bar = 4 if timeframe == "H4" else 1
            ergebnis = trade_simulieren(
                df, i, signal, spread_kosten, tp_pct, sl_pct, horizon,
                risiko_config=cfg, atr_wert=atr_wert,
                swap_aktiv=swap_aktiv,
                swap_long=swap_long_satz,
                swap_short=swap_short_satz,
                entry_time=df.index[i],
                stunden_pro_bar=stunden_pro_bar,
            )

            # Swap-Zähler erhöhen wenn Overnight-Kosten angefallen
            if ergebnis.get("hat_swap", False):
                n_swap_trades += 1

            # Trade-Details speichern (inkl. neuer Risiko-Felder + Swap-Flag)
            regime_wert = df["market_regime"].iloc[i] if hat_regime else np.nan
            trades.append(
                {
                    "time": df.index[i],
                    "symbol": symbol,
                    "richtung": "Long" if signal == 2 else "Short",
                    "signal_klasse": signal,
                    "prob": df["prob_signal"].iloc[i],
                    "eintrittspreis": ergebnis["eintrittspreis"],
                    "sl_pct_verwendet": ergebnis["sl_pct_verwendet"],
                    "lot_groesse": ergebnis["lot_groesse"],
                    "exit_grund": ergebnis["exit_grund"],
                    "n_bars": ergebnis["n_bars"],
                    "pnl_pct": ergebnis["pnl_pct"],
                    "pnl_absolut": ergebnis["pnl_absolut"],
                    "gewinn": ergebnis["pnl_pct"] > 0,
                    "market_regime": regime_wert,
                    "hat_swap": ergebnis.get("hat_swap", False),
                }
            )

            # Nächste Signal-Suche: nach dem Trade (keine überlappenden Trades)
            i += ergebnis["n_bars"] + 1
        else:
            i += 1

    if regime_erlaubt is not None and n_gefiltert_regime > 0:
        regime_namen_str = [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        logger.info(
            f"[{symbol}] Regime-Filter aktiv: {n_gefiltert_regime} Signale "
            f"außerhalb [{', '.join(regime_namen_str)}] übersprungen"
        )

    if not trades:
        logger.warning(
            f"[{symbol}] Keine Trades gefunden! "
            f"Schwelle zu hoch oder Regime-Filter zu streng?"
        )
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df.set_index("time", inplace=True)

    logger.info(
        f"[{symbol}] {len(trades_df)} Trades simuliert | Schwelle={schwelle:.0%} | "
        f"Long: {(trades_df['richtung'] == 'Long').sum()} | "
        f"Short: {(trades_df['richtung'] == 'Short').sum()} | "
        f"TP={tp_pct:.1%} / SL={sl_pct:.1%} (RRR={tp_pct/sl_pct:.1f}:1)"
    )
    # Swap-Kosten-Statistik ausgeben (nur wenn Swap aktiviert)
    if swap_aktiv and n_swap_trades > 0:
        logger.info(
            f"[{symbol}] Swap-Kosten: {n_swap_trades}/{len(trades_df)} Trades "
            f"({n_swap_trades/len(trades_df):.0%}) hatten Overnight-Gebühren"
        )

    return trades_df


# ============================================================
# 5. Kennzahlen berechnen
# ============================================================


def kennzahlen_berechnen(  # pylint: disable=too-many-locals
    trades_df: pd.DataFrame,
    symbol: str,
) -> dict:
    """
    Berechnet alle Trading-Kennzahlen aus der Trade-Liste.

    Kennzahlen:
        - Gesamtrendite (%): Summe aller Trade-P&Ls
        - Win-Rate (%): Anteil der Gewinn-Trades
        - Gewinnfaktor: Summe Gewinne / Summe Verluste (Ziel: >1.3)
        - Sharpe Ratio: Rendite / Risiko (Ziel: >1.0)
        - Max. Drawdown (%): Größter Kursrückgang von Hoch zu Tief
        - Anzahl Trades

    Args:
        trades_df: Trade-Liste aus trades_simulieren()
        symbol: Handelssymbol (für Logging)

    Returns:
        Dict mit allen Kennzahlen.
    """
    if trades_df.empty:
        return {
            "symbol": symbol,
            "n_trades": 0,
            "gesamtrendite_pct": 0.0,
            "win_rate_pct": 0.0,
            "gewinnfaktor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl = trades_df["pnl_pct"].values
    n_trades = len(pnl)

    # Gesamtrendite: Summe aller P&Ls (kumulativ)
    gesamtrendite = float(pnl.sum()) * 100  # in Prozent

    # Win-Rate: Anteil Gewinn-Trades
    n_gewinner = (pnl > 0).sum()
    win_rate = n_gewinner / n_trades * 100

    # Gewinnfaktor: Summe aller Gewinne / Summe aller Verluste
    gewinne = pnl[pnl > 0].sum()
    verluste = abs(pnl[pnl < 0].sum())
    gewinnfaktor = gewinne / verluste if verluste > 0 else float("inf")

    # Equity-Kurve (kumulativer P&L)
    equity = np.cumsum(pnl)

    # Max. Drawdown: größter Rückgang von Hoch zu nachfolgendem Tief
    laufendes_maximum = np.maximum.accumulate(equity)
    drawdowns = equity - laufendes_maximum
    max_drawdown = float(drawdowns.min()) * 100  # in Prozent (negativ!)

    # Sharpe Ratio: annualisiert
    # H1-Kerzen: ~6 Stunden Forex-Handel × 5 Tage × 52 Wochen ≈ 1560 Trades/Jahr
    # Aber wir messen pro Trade, nicht pro Stunde → Annualisierung approximiert
    if len(pnl) > 1:
        sharpe = float(pnl.mean() / pnl.std()) * np.sqrt(252)  # √252 für Tagesdaten
    else:
        sharpe = 0.0

    # ────────────────────────────────────────────────────────
    # Feature 1: Absoluter P&L (nur wenn Positionsgrößen berechnet wurden)
    # ────────────────────────────────────────────────────────
    hat_absolut = (
        "pnl_absolut" in trades_df.columns
        and trades_df["pnl_absolut"].abs().sum() > 0
    )
    if hat_absolut:
        pnl_abs = trades_df["pnl_absolut"].values
        gesamtrendite_absolut = float(pnl_abs.sum())
        hat_lot = "lot_groesse" in trades_df.columns
        avg_lot = float(trades_df["lot_groesse"].mean()) if hat_lot else 0.0
    else:
        gesamtrendite_absolut = 0.0
        avg_lot = 0.0

    # Ergebnisse zusammenstellen
    kennzahlen = {
        "symbol": symbol,
        "n_trades": n_trades,
        "n_long": int((trades_df["richtung"] == "Long").sum()),
        "n_short": int((trades_df["richtung"] == "Short").sum()),
        "gesamtrendite_pct": round(gesamtrendite, 2),
        "win_rate_pct": round(win_rate, 1),
        "gewinnfaktor": round(gewinnfaktor, 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "tp_hits": int((trades_df["exit_grund"] == "tp").sum()),
        "sl_hits": int((trades_df["exit_grund"] == "sl").sum()),
        "horizon_exits": int((trades_df["exit_grund"] == "horizon").sum()),
        "zeitraum_von": str(trades_df.index[0].date()),
        "zeitraum_bis": str(trades_df.index[-1].date()),
        "gesamtrendite_absolut": round(gesamtrendite_absolut, 2),
        "avg_lot_groesse": round(avg_lot, 4),
    }

    # Log-Ausgabe
    logger.info(f"\n{'─' * 55}")
    logger.info(
        f"[{symbol}] KENNZAHLEN "
        f"(Test-Set {kennzahlen['zeitraum_von']} bis {kennzahlen['zeitraum_bis']})"
    )
    logger.info(f"{'─' * 55}")
    logger.info(
        f"  Anzahl Trades:    {n_trades:6d} "
        f"(Long: {kennzahlen['n_long']}, Short: {kennzahlen['n_short']})"
    )
    logger.info(f"  Gesamtrendite:    {gesamtrendite:+7.2f}%")
    if hat_absolut:
        logger.info(
            f"  Gesamtrendite:  {gesamtrendite_absolut:+10.2f} EUR/USD "
            f"(Ø Lot: {avg_lot:.4f})"
        )
    logger.info(f"  Win-Rate:         {win_rate:7.1f}%")
    logger.info(f"  Gewinnfaktor:     {gewinnfaktor:7.3f}  (Ziel: >1.3)")
    logger.info(f"  Sharpe Ratio:     {sharpe:7.3f}  (Ziel: >1.0)")
    logger.info(f"  Max. Drawdown:    {max_drawdown:+7.2f}%  (Ziel: >-20%)")
    logger.info(
        f"  Exits: TP={kennzahlen['tp_hits']} | "
        f"SL={kennzahlen['sl_hits']} | "
        f"Horizon={kennzahlen['horizon_exits']}"
    )

    # Zielampel
    ziele = {
        "Gewinnfaktor > 1.3": gewinnfaktor > 1.3,
        "Sharpe > 1.0": sharpe > 1.0,
        "Drawdown > -20%": max_drawdown > -20,
        "Win-Rate > 45%": win_rate > 45,
    }
    logger.info("\n  Ziel-Check:")
    for ziel, erreicht in ziele.items():
        zeichen = "✅" if erreicht else "❌"
        logger.info(f"    {zeichen} {ziel}")

    return kennzahlen


# ============================================================
# 6. Performance nach Market-Regime analysieren
# ============================================================


def regime_analyse(trades_df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """
    Analysiert die Performance aufgeschlüsselt nach Market-Regime.

    Zeigt z.B.: "Im Aufwärtstrend hat das System 65% Win-Rate,
                 in Seitwärtsphasen nur 48%"

    Args:
        trades_df: Trade-Liste mit market_regime-Spalte
        symbol: Handelssymbol

    Returns:
        DataFrame mit Kennzahlen pro Regime oder None wenn keine Regime-Daten.
    """
    if trades_df.empty or "market_regime" not in trades_df.columns:
        logger.warning(f"[{symbol}] Keine Regime-Daten – Regime-Analyse übersprungen")
        return None

    # Regime-Daten bereinigen
    trades_mit_regime = trades_df.dropna(subset=["market_regime"]).copy()
    if len(trades_mit_regime) == 0:
        return None

    trades_mit_regime["market_regime"] = trades_mit_regime["market_regime"].astype(int)

    logger.info(f"\n[{symbol}] Performance nach Regime:")
    ergebnisse = []

    for regime_nr in sorted(trades_mit_regime["market_regime"].unique()):
        regime_trades = trades_mit_regime[
            trades_mit_regime["market_regime"] == regime_nr
        ]
        regime_name = REGIME_NAMEN.get(int(regime_nr), f"Regime {regime_nr}")

        pnl = regime_trades["pnl_pct"].values
        n = len(pnl)
        win_rate = (pnl > 0).mean() * 100
        rendite = pnl.sum() * 100
        verluste = abs(pnl[pnl < 0].sum())
        gewinnfaktor = pnl[pnl > 0].sum() / verluste if verluste > 0 else float("inf")

        ergebnisse.append(
            {
                "regime": regime_name,
                "n_trades": n,
                "win_rate_pct": round(win_rate, 1),
                "gesamtrendite_pct": round(rendite, 2),
                "gewinnfaktor": round(gewinnfaktor, 3),
            }
        )
        logger.info(
            f"  {regime_name:20s}: N={n:4d} | Win-Rate={win_rate:5.1f}% | "
            f"Rendite={rendite:+6.2f}% | GF={gewinnfaktor:.2f}"
        )

    return pd.DataFrame(ergebnisse)


# ============================================================
# 6b. Regime-Performance-Matrix (alle Symbole kombiniert)
# ============================================================


def regime_matrix_erstellen(ziel_symbole: list) -> Optional[pd.DataFrame]:
    """
    Liest alle gespeicherten Trade-CSVs und erstellt eine Regime-Performance-Matrix.

    Ergebnis: Für jede Symbol/Regime-Kombination werden N, Win-Rate,
    Gesamtrendite und Gewinnfaktor berechnet. Zeigt klar welche Regimes
    profitabel sind und welche gemieden werden sollen.

    Args:
        ziel_symbole: Liste der Symbole die analysiert werden sollen

    Returns:
        DataFrame mit Spalten [symbol, regime, regime_name, n_trades,
        win_rate_pct, gesamtrendite_pct, gewinnfaktor] oder None
    """
    alle_daten = []

    for symbol in ziel_symbole:
        # Trade-CSV laden (wird am Ende von symbol_backtest() gespeichert)
        trade_pfad = BACKTEST_DIR / f"{symbol}_trades.csv"
        if not trade_pfad.exists():
            logger.warning(f"[{symbol}] Trade-CSV nicht gefunden: {trade_pfad}")
            continue

        trades = pd.read_csv(trade_pfad, index_col=0, parse_dates=True)

        if "market_regime" not in trades.columns or trades.empty:
            logger.warning(f"[{symbol}] Keine market_regime-Spalte in Trade-CSV")
            continue

        # Gültige Regime-Daten extrahieren
        trades = trades.dropna(subset=["market_regime"]).copy()
        trades["market_regime"] = trades["market_regime"].astype(int)

        # Für jedes vorhandene Regime die Kennzahlen berechnen
        for regime_nr in sorted(trades["market_regime"].unique()):
            regime_trades = trades[trades["market_regime"] == regime_nr]
            pnl = regime_trades["pnl_pct"].values
            n = len(pnl)
            if n == 0:
                continue

            win_rate = (pnl > 0).mean() * 100
            rendite = pnl.sum() * 100
            verluste = abs(pnl[pnl < 0].sum())
            gf = pnl[pnl > 0].sum() / verluste if verluste > 0 else float("inf")

            alle_daten.append(
                {
                    "symbol": symbol,
                    "regime": regime_nr,
                    "regime_name": REGIME_NAMEN.get(int(regime_nr), f"Regime {regime_nr}"),
                    "n_trades": n,
                    "win_rate_pct": round(win_rate, 1),
                    "gesamtrendite_pct": round(rendite, 2),
                    "gewinnfaktor": round(gf, 3),
                }
            )

    if not alle_daten:
        logger.warning("Keine Regime-Daten vorhanden – Matrix kann nicht erstellt werden")
        return None

    return pd.DataFrame(alle_daten)


def regime_matrix_drucken(matrix_df: pd.DataFrame, regime_info: str) -> None:
    """
    Gibt die Regime-Performance-Matrix als formatierte Tabelle aus.

    Empfiehlt für jedes Symbol den besten Regime-Filter basierend auf
    Rendite und Win-Rate.

    Args:
        matrix_df: Ergebnis aus regime_matrix_erstellen()
        regime_info: Beschreibung des aktiven Regime-Filters (für Header)
    """
    # Alle vorhandenen Regime-Nummern ermitteln
    regime_nummern = sorted(matrix_df["regime"].unique())

    # Header ausgeben
    print(f"\n{'═' * 80}")
    print(f"REGIME-PERFORMANCE-MATRIX – Welches Regime ist profitabel?")
    print(f"Regime-Filter beim Backtest: {regime_info}")
    print(f"{'═' * 80}")

    # Spalten-Header
    header = f"{'Symbol':8}"
    for reg_nr in regime_nummern:
        reg_kurz = REGIME_NAMEN.get(reg_nr, f"R{reg_nr}")[:12]
        header += f"  │ {reg_kurz:12} (N   Win%  Rend%)"
    print(header)
    print(f"{'─' * 80}")

    # Für jedes Symbol eine Zeile ausgeben
    symbole = matrix_df["symbol"].unique()
    empfehlungen = {}  # {symbol: [profitable Regime-Nummern]}

    for symbol in symbole:
        zeile = f"{symbol:8}"
        profitable_regimes = []

        for reg_nr in regime_nummern:
            subset = matrix_df[
                (matrix_df["symbol"] == symbol) & (matrix_df["regime"] == reg_nr)
            ]
            if subset.empty:
                zeile += f"  │ {'—':>35}"
            else:
                row = subset.iloc[0]
                n = int(row["n_trades"])
                win = row["win_rate_pct"]
                rend = row["gesamtrendite_pct"]

                # :+.2f fügt automatisch + oder – hinzu
                zeile += f"  │ {n:4d}  {win:5.1f}%  {rend:+.2f}%"

                # Als profitabel markieren wenn Rendite > 0 UND Win-Rate > 48%
                if rend > 0 and win > 48:
                    profitable_regimes.append(reg_nr)

        print(zeile)
        empfehlungen[symbol] = profitable_regimes

    print(f"{'─' * 80}")

    # Empfehlungen ausgeben
    print("\nREGIME-FILTER-EMPFEHLUNG (Rendite > 0% UND Win-Rate > 48%):")
    for symbol in symbole:
        regs = empfehlungen[symbol]
        if regs:
            reg_str = ",".join(str(r) for r in regs)
            namen = " + ".join(REGIME_NAMEN.get(r, str(r)) for r in regs)
            print(f"  {symbol}: --regime_filter {reg_str}  ({namen})")
        else:
            print(f"  {symbol}: Kein Regime profitabel – Symbol nicht handeln!")

    print(f"{'═' * 80}")


def regime_matrix_plotten(matrix_df: pd.DataFrame) -> None:
    """
    Erstellt eine Heatmap der Regime-Performance für alle Symbole.

    Zeigt gesamtrendite_pct als Farbskala: rot = negativ, grün = positiv.
    Speichert als: plots/regime_performance_matrix.png

    Args:
        matrix_df: Ergebnis aus regime_matrix_erstellen()
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pivot-Tabelle: Symbole als Zeilen, Regimes als Spalten
    pivot = matrix_df.pivot_table(
        index="symbol",
        columns="regime_name",
        values="gesamtrendite_pct",
        aggfunc="sum",
    )

    # Spalten-Reihenfolge nach Regime-Nummer
    regime_reihenfolge = [
        REGIME_NAMEN[r]
        for r in sorted(REGIME_NAMEN.keys())
        if REGIME_NAMEN[r] in pivot.columns
    ]
    pivot = pivot.reindex(columns=regime_reihenfolge)

    # Pivot für Win-Rate (als Annotation)
    pivot_win = matrix_df.pivot_table(
        index="symbol",
        columns="regime_name",
        values="win_rate_pct",
        aggfunc="mean",
    ).reindex(columns=regime_reihenfolge)

    # Beschriftungen: "Rend%\nN=x\nWin%"
    pivot_n = matrix_df.pivot_table(
        index="symbol",
        columns="regime_name",
        values="n_trades",
        aggfunc="sum",
    ).reindex(columns=regime_reihenfolge)

    # Annotation-Text erstellen
    annot = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=str)
    for sym in pivot.index:
        for col in pivot.columns:
            rend = pivot.loc[sym, col]
            n = pivot_n.loc[sym, col]
            win = pivot_win.loc[sym, col]
            if pd.isna(rend):
                annot.loc[sym, col] = "–"
            else:
                prefix = "+" if rend >= 0 else ""
                annot.loc[sym, col] = f"{prefix}{rend:.2f}%\nN={int(n)}\n{win:.0f}%W"

    # Heatmap zeichnen
    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.9 + 2)))
    fig.patch.set_facecolor("#F8F9FA")

    sns.heatmap(
        pivot.astype(float),
        annot=annot,
        fmt="",
        cmap="RdYlGn",  # Rot (negativ) → Gelb (0) → Grün (positiv)
        center=0,
        linewidths=0.5,
        linecolor="#CCCCCC",
        ax=ax,
        cbar_kws={"label": "Gesamtrendite (%)"},
        annot_kws={"size": 9},
    )

    ax.set_title(
        "Regime-Performance-Matrix – Gesamtrendite % pro Symbol & Regime\n"
        "(gruen = profitabel, rot = Verlust | N = Anzahl Trades | W = Win-Rate)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Marktregime", fontsize=11)
    ax.set_ylabel("Symbol", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    pfad = PLOTS_DIR / "regime_performance_matrix.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Regime-Matrix-Plot gespeichert: {pfad}")


# ============================================================
# 7. Equity-Kurve plotten
# ============================================================


def equity_kurve_plotten(
    trades_df: pd.DataFrame, symbol: str, kennzahlen: dict
) -> None:
    """
    Erstellt die Equity-Kurve des Backtests.

    Zeigt den kumulativen P&L (%) über Zeit und den Max. Drawdown.

    Args:
        trades_df: Trade-Liste mit pnl_pct-Spalte
        symbol: Handelssymbol
        kennzahlen: Berechnete Kennzahlen (für Titel)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pnl = trades_df["pnl_pct"].values
    equity = np.cumsum(pnl) * 100  # in Prozent
    daten = trades_df.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1])
    fig.patch.set_facecolor("#F8F9FA")

    # ---- Oben: Equity-Kurve ----
    ax1.plot(daten, equity, color="#2ECC71", linewidth=1.5, label="Kumulativer P&L")
    ax1.axhline(0, color="#95A5A6", linewidth=1, linestyle="--", alpha=0.7)
    ax1.fill_between(daten, equity, 0, where=(equity > 0), alpha=0.15, color="#2ECC71")
    ax1.fill_between(daten, equity, 0, where=(equity < 0), alpha=0.15, color="#E74C3C")

    # Max-Drawdown-Bereich markieren
    laufendes_max = np.maximum.accumulate(equity)
    drawdown = equity - laufendes_max
    max_dd_idx = np.argmin(drawdown)
    if max_dd_idx > 0:
        ax1.axvspan(
            daten[0],
            daten[min(max_dd_idx, len(daten) - 1)],
            alpha=0.03,
            color="#E74C3C",
        )

    ax1.set_ylabel("Kumulativer P&L (%)", fontsize=11)
    ax1.set_title(
        f"{symbol} – Backtest Equity-Kurve (Test-Set 2023+)\n"
        f"Trades: {kennzahlen['n_trades']} | "
        f"Rendite: {kennzahlen['gesamtrendite_pct']:+.2f}% | "
        f"Sharpe: {kennzahlen['sharpe_ratio']:.2f} | "
        f"Max. DD: {kennzahlen['max_drawdown_pct']:.2f}% | "
        f"GF: {kennzahlen['gewinnfaktor']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#F8F9FA")

    # ---- Unten: Drawdown ----
    laufendes_max_norm = np.maximum.accumulate(equity)
    drawdown_kurve = equity - laufendes_max_norm
    ax2.fill_between(daten, drawdown_kurve, 0, alpha=0.5, color="#E74C3C")
    ax2.plot(daten, drawdown_kurve, color="#E74C3C", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_facecolor("#F8F9FA")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        kennzahlen["max_drawdown_pct"],
        color="#C0392B",
        linestyle=":",
        linewidth=1.5,
        label=f"Max. DD: {kennzahlen['max_drawdown_pct']:.2f}%",
    )
    ax2.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_backtest_equity.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Equity-Kurve gespeichert: {pfad}")


# ============================================================
# 8. Performance nach Regime plotten
# ============================================================


def regime_plotten(regime_df: pd.DataFrame, symbol: str) -> None:  # pylint: disable=too-many-locals
    """
    Erstellt einen Balkenplot der Performance nach Market-Regime.

    Args:
        regime_df: Ergebnis aus regime_analyse()
        symbol: Handelssymbol
    """
    if regime_df is None or regime_df.empty:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F8F9FA")

    regimes = regime_df["regime"]
    farben_rendite = [
        "#2ECC71" if r >= 0 else "#E74C3C" for r in regime_df["gesamtrendite_pct"]
    ]

    # Plot 1: Rendite nach Regime
    balken = ax1.bar(
        regimes,
        regime_df["gesamtrendite_pct"],
        color=farben_rendite,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.axhline(0, color="#7F8C8D", linewidth=1)
    for b, wert in zip(balken, regime_df["gesamtrendite_pct"]):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            wert + (0.02 if wert >= 0 else -0.05),
            f"{wert:+.2f}%",
            ha="center",
            va="bottom" if wert >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylabel("Gesamtrendite (%)")
    ax1.set_title(f"{symbol} – Rendite nach Regime", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)
    ax1.set_facecolor("#F8F9FA")
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot 2: Win-Rate nach Regime
    farben_wr = ["#2ECC71" if w >= 50 else "#E74C3C" for w in regime_df["win_rate_pct"]]
    balken2 = ax2.bar(
        regimes,
        regime_df["win_rate_pct"],
        color=farben_wr,
        edgecolor="white",
        linewidth=1.5,
    )
    ax2.axhline(50, color="#7F8C8D", linewidth=1, linestyle="--", label="50% Baseline")
    for b, wert in zip(balken2, regime_df["win_rate_pct"]):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            wert + 0.5,
            f"{wert:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylabel("Win-Rate (%)")
    ax2.set_title(f"{symbol} – Win-Rate nach Regime", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_facecolor("#F8F9FA")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    # Anzahl Trades als Annotation (col wird nicht benötigt, daher _)
    for ax, _ in [(ax1, "gesamtrendite_pct"), (ax2, "win_rate_pct")]:
        for i, (_, row) in enumerate(regime_df.iterrows()):
            ax.text(
                i,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                f"N={row['n_trades']}",
                ha="center",
                fontsize=8,
                color="#7F8C8D",
            )

    plt.suptitle(
        f"{symbol} – Performance nach Market-Regime",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_backtest_regime.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Regime-Plot gespeichert: {pfad}")


# ============================================================
# 9. Monatliche Returns-Heatmap
# ============================================================


def monatliche_heatmap_plotten(  # pylint: disable=too-many-locals
    trades_df: pd.DataFrame, symbol: str
) -> None:
    """
    Erstellt eine Heatmap der monatlichen P&L-Werte.

    Grün = profitable Monate, Rot = Verlust-Monate.
    Zeigt ob das System saisonal stabil ist.

    Args:
        trades_df: Trade-Liste
        symbol: Handelssymbol
    """
    if trades_df.empty:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Monatliche P&L aggregieren
    monatlich = (
        trades_df["pnl_pct"].resample("ME").sum()  # Monatsende-Resampling
        * 100  # in Prozent
    )

    if len(monatlich) < 3:
        logger.info(f"[{symbol}] Zu wenige Monate für Heatmap – übersprungen")
        return

    # Pivot: Jahr × Monat
    monatlich_df = pd.DataFrame(
        {
            "Jahr": monatlich.index.year,
            "Monat": monatlich.index.month,
            "Rendite": monatlich.values,
        }
    )
    pivot = monatlich_df.pivot_table(
        values="Rendite", index="Jahr", columns="Monat", aggfunc="sum"
    )

    # Monatsnamen für x-Achse
    monat_namen = [
        "Jan",
        "Feb",
        "Mär",
        "Apr",
        "Mai",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Dez",
    ]
    pivot.columns = [monat_namen[int(m) - 1] for m in pivot.columns]

    # Heatmap erstellen (fig wird nicht direkt referenziert, daher _)
    _, ax = plt.subplots(figsize=(13, max(3, len(pivot) * 0.8 + 2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "P&L (%)"},
        ax=ax,
    )
    ax.set_title(
        f"{symbol} – Monatliche Returns (%)\n"
        f"Grün = profitabler Monat | Rot = Verlust-Monat",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Jahr")

    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_backtest_monatlich.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Monatliche Heatmap gespeichert: {pfad}")


# ============================================================
# 10. Jahres-Perioden-Vergleich plotten (Feature 3)
# ============================================================


def perioden_vergleich_plotten(  # pylint: disable=too-many-locals,too-many-statements
    trades_df: pd.DataFrame, symbol: str
) -> None:
    """
    Erstellt einen Jahresvergleich der Backtest-Performance.

    Teilt die Trade-Liste nach Jahren auf und zeigt für jedes Jahr:
    - Gesamtrendite (%)
    - Win-Rate (%)
    - Anzahl Trades

    So lassen sich schlechte Jahre (z.B. Trend-Monate) von guten unterscheiden.
    Das hilft herauszufinden ob das System in allen Marktphasen funktioniert.

    Args:
        trades_df: Trade-Liste aus trades_simulieren()
        symbol:    Handelssymbol
    """
    if trades_df.empty:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Jahres-Performance berechnen
    jahre = sorted(trades_df.index.year.unique())

    if len(jahre) < 2:
        logger.info(
            f"[{symbol}] Zu wenige Jahre für Perioden-Vergleich "
            f"({len(jahre)} Jahr(e)) – übersprungen"
        )
        return

    perioden_daten = []
    for jahr in jahre:
        jahres_trades = trades_df[trades_df.index.year == jahr]
        pnl = jahres_trades["pnl_pct"].values
        n = len(pnl)
        if n == 0:
            continue
        win_rate = float((pnl > 0).mean()) * 100
        rendite = float(pnl.sum()) * 100
        perioden_daten.append(
            {
                "periode": str(jahr),
                "n_trades": n,
                "gesamtrendite_pct": round(rendite, 2),
                "win_rate_pct": round(win_rate, 1),
            }
        )

    if len(perioden_daten) < 2:
        return

    perioden_df = pd.DataFrame(perioden_daten)

    # ---- Plot ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#F8F9FA")

    perioden = perioden_df["periode"]
    farben_rendite = [
        "#2ECC71" if r >= 0 else "#E74C3C"
        for r in perioden_df["gesamtrendite_pct"]
    ]
    farben_wr = [
        "#2ECC71" if w >= 50 else "#E74C3C"
        for w in perioden_df["win_rate_pct"]
    ]

    # Plot 1: Gesamtrendite pro Jahr
    balken1 = ax1.bar(
        perioden,
        perioden_df["gesamtrendite_pct"],
        color=farben_rendite,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.axhline(0, color="#7F8C8D", linewidth=1)
    for b, wert in zip(balken1, perioden_df["gesamtrendite_pct"]):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            wert + (0.01 if wert >= 0 else -0.03),
            f"{wert:+.2f}%",
            ha="center",
            va="bottom" if wert >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylabel("Gesamtrendite (%)")
    ax1.set_title(f"{symbol} – Rendite pro Jahr", fontweight="bold")
    ax1.set_facecolor("#F8F9FA")
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot 2: Win-Rate pro Jahr
    balken2 = ax2.bar(
        perioden,
        perioden_df["win_rate_pct"],
        color=farben_wr,
        edgecolor="white",
        linewidth=1.5,
    )
    ax2.axhline(50, color="#7F8C8D", linewidth=1, linestyle="--", label="50%")
    for b, wert in zip(balken2, perioden_df["win_rate_pct"]):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            wert + 0.5,
            f"{wert:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylabel("Win-Rate (%)")
    ax2.set_title(f"{symbol} – Win-Rate pro Jahr", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_facecolor("#F8F9FA")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    # Plot 3: Anzahl Trades pro Jahr
    ax3.bar(
        perioden,
        perioden_df["n_trades"],
        color="#3498DB",
        edgecolor="white",
        linewidth=1.5,
    )
    for idx, wert in enumerate(perioden_df["n_trades"]):
        ax3.text(
            idx,
            wert + 0.3,
            str(wert),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax3.set_ylabel("Anzahl Trades")
    ax3.set_title(f"{symbol} – Trades pro Jahr", fontweight="bold")
    ax3.set_facecolor("#F8F9FA")
    ax3.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"{symbol} – Jahresvergleich ({perioden.iloc[0]}–{perioden.iloc[-1]})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    pfad = PLOTS_DIR / f"{symbol}_backtest_perioden.png"
    plt.savefig(pfad, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Perioden-Vergleich gespeichert: {pfad}")

    # Log-Ausgabe der Jahres-Kennzahlen
    logger.info(f"\n[{symbol}] Performance nach Jahr:")
    for _, row in perioden_df.iterrows():
        icon = "✅" if row["gesamtrendite_pct"] >= 0 else "❌"
        logger.info(
            f"  {row['periode']}: {icon} "
            f"Rendite={row['gesamtrendite_pct']:+.2f}% | "
            f"Win-Rate={row['win_rate_pct']:.1f}% | "
            f"N={row['n_trades']}"
        )


# ============================================================
# 11. Vollständiger Backtest für ein Symbol
# ============================================================


def symbol_backtest(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-locals
    symbol: str,
    schwelle: float = 0.55,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    regime_erlaubt: Optional[list] = None,
    version: str = "v1",
    horizon: int = HORIZON,
    risiko_config: Optional[RisikoConfig] = None,
    zeitraum_von: Optional[str] = None,
    zeitraum_bis: Optional[str] = None,
    spread_faktor: float = 1.0,
    swap_aktiv: bool = False,
    timeframe: str = "H1",
) -> Optional[dict]:
    """
    Führt den vollständigen Backtest für ein Symbol durch.

    Schritte:
        1. Test-Set laden (2023+, optional gefiltert nach zeitraum_von/bis)
        2. Signale generieren (versioniertes Modell + Schwellenwert)
        3. Trades simulieren (TP/SL/Horizon + Spread + Regime-Filter + RisikoConfig)
        4. Kennzahlen berechnen (inkl. absoluter P&L wenn kapital > 0)
        5. Regime-Analyse
        6. Plots erstellen (Equity, Regime, Monatlich, Jahresvergleich)
        7. Trade-Log als CSV speichern

    Args:
        symbol:         Handelssymbol (z.B. "EURUSD")
        schwelle:       Wahrscheinlichkeits-Schwellenwert (Standard: 0.55)
        tp_pct:         Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:         Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        regime_erlaubt: Nur in diesen Regimes handeln (None = alle)
        version:        Versions-String für Modell- und Datei-Pfade (Standard: "v1")
        horizon:        Zeitschranke in Kerzen (muss mit labeling.py übereinstimmen!)
        risiko_config:  Dynamische Positionsgröße + ATR-SL (Standard: None = einfacher Modus)
        zeitraum_von:   Optionaler Startzeitpunkt (z.B. "2023-01-01")
        zeitraum_bis:   Optionaler Endzeitpunkt   (z.B. "2023-12-31")
        spread_faktor:  Spread-Multiplikator für Kosten-Stress-Test (Standard: 1.0 = normal)

    Returns:
        Dict mit Kennzahlen oder None bei Fehler.
    """
    # Info-String für Logging und Dateinamen
    regime_info = (
        f"Regimes=[{','.join(str(r) for r in regime_erlaubt)}]"
        if regime_erlaubt
        else "alle Regimes"
    )
    atr_info = (
        f" | ATR-SL={risiko_config.atr_faktor}×ATR"
        if risiko_config and risiko_config.atr_sl
        else ""
    )
    kapital_info = (
        f" | Kapital={risiko_config.kapital:,.0f} (Risiko={risiko_config.risiko_pct:.0%})"
        if risiko_config and risiko_config.kapital > 0
        else ""
    )
    logger.info(f"\n{'=' * 65}")
    logger.info(
        f"Backtest – {symbol} ({version}) | Schwelle: {schwelle:.0%} | "
        f"TP={tp_pct:.2%} / SL={sl_pct:.2%} | Horizon={horizon} | "
        f"{regime_info}{atr_info}{kapital_info}"
    )
    logger.info(f"{'=' * 65}")

    try:
        # Schritt 1: Test-Set laden (mit optionalem Zeitraum-Filter + Zeitrahmen)
        df = daten_laden(symbol, version, zeitraum_von, zeitraum_bis, timeframe)

        # Schritt 2: Signale generieren (versioniertes Modell, passend zum Zeitrahmen)
        df = signale_generieren(df, symbol, schwelle, version, timeframe)

        # Schritt 3: Trades simulieren (inkl. Risikomanagement + Spread-Faktor + Swap)
        trades_df = trades_simulieren(
            df, symbol, schwelle, tp_pct, sl_pct,
            regime_erlaubt, horizon, risiko_config,
            spread_faktor=spread_faktor,
            swap_aktiv=swap_aktiv,
            timeframe=timeframe,
        )

        if trades_df.empty:
            logger.warning(f"[{symbol}] Keine Trades – übersprungen!")
            return None

        # Schritt 4: Kennzahlen berechnen
        kennzahlen = kennzahlen_berechnen(trades_df, symbol)
        # Konfiguration in Kennzahlen speichern (für Zusammenfassung)
        kennzahlen["tp_pct"] = tp_pct
        kennzahlen["sl_pct"] = sl_pct
        kennzahlen["rrr"] = round(tp_pct / sl_pct, 2)
        kennzahlen["regime_filter"] = str(regime_erlaubt) if regime_erlaubt else "alle"
        kennzahlen["spread_faktor"] = spread_faktor  # Für Sensitivity-Test-Protokoll
        if risiko_config and risiko_config.kapital > 0:
            kennzahlen["kapital"] = risiko_config.kapital
            kennzahlen["risiko_pct"] = risiko_config.risiko_pct
            kennzahlen["atr_sl"] = risiko_config.atr_sl

        # Schritt 5: Regime-Analyse
        regime_df = regime_analyse(trades_df, symbol)

        # Schritt 6: Plots erstellen (inkl. neuem Jahres-Perioden-Vergleich)
        equity_kurve_plotten(trades_df, symbol, kennzahlen)
        regime_plotten(regime_df, symbol)
        monatliche_heatmap_plotten(trades_df, symbol)
        perioden_vergleich_plotten(trades_df, symbol)  # NEU: Jahresvergleich

        # Schritt 7: Trade-Log als CSV speichern
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        trade_pfad = BACKTEST_DIR / f"{symbol}_trades.csv"
        trades_df.to_csv(trade_pfad)
        logger.info(
            f"[{symbol}] Trade-Log gespeichert: "
            f"{trade_pfad} ({len(trades_df)} Trades)"
        )

        return kennzahlen

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"[{symbol}] Unerwarteter Fehler: {e}", exc_info=True)
        return None


# ============================================================
# 12. Hauptprogramm
# ============================================================


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Backtest für ein oder alle Symbole."""

    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – Backtest auf Test-Set (2023+)"
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        default=["EURUSD"],
        help=(
            "Ein oder mehrere Symbole (Standard: EURUSD) oder 'alle'. "
            "Beispiel: --symbol EURUSD USDCAD USDJPY"
        ),
    )
    parser.add_argument(
        "--schwelle",
        type=float,
        default=0.55,
        help=(
            "Wahrscheinlichkeits-Schwellenwert für Trade-Ausführung (Standard: 0.55). "
            "Höher = weniger aber zuverlässigere Trades."
        ),
    )
    parser.add_argument(
        "--tp_pct",
        type=float,
        default=TP_PCT,
        help=(
            f"Take-Profit als Anteil (Standard: {TP_PCT} = {TP_PCT*100:.1f}%%). "
            "Beispiel: --tp_pct 0.006 fuer 0.6%% TP bei 0.3%% SL (RRR=2:1)."
        ),
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=SL_PCT,
        help=(
            f"Stop-Loss als Anteil (Standard: {SL_PCT} = {SL_PCT*100:.1f}%%). "
            "Beispiel: --sl_pct 0.003 fuer 0.3%% SL."
        ),
    )
    parser.add_argument(
        "--regime_filter",
        type=str,
        default=None,
        help=(
            "Komma-getrennte Liste erlaubter Regime-Nummern (Standard: alle). "
            "0=Seitwärts, 1=Aufwärtstrend, 2=Abwärtstrend, 3=Hohe Volatilität. "
            "Beispiel: --regime_filter 1,2  (nur Trend-Phasen handeln)"
        ),
    )
    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Versions-Suffix für Modell- und Datei-Pfade (Standard: v1). "
            "Muss mit --version aus labeling.py und train_model.py übereinstimmen. "
            "v1 = Original | v2 = Horizon=10 | v3 = TP/SL=0.15%%"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help=(
            f"Zeitschranke in H1-Kerzen (Standard: {HORIZON}). "
            "Muss mit --horizon aus labeling.py übereinstimmen! "
            "Option A (v2): --horizon 10"
        ),
    )

    # ── Feature 1: Dynamische Positionsgrößenberechnung ──────────────────────
    parser.add_argument(
        "--kapital",
        type=float,
        default=0.0,
        help=(
            "Startkapital für Positionsgrößenberechnung (Standard: 0 = deaktiviert). "
            "Beispiel: --kapital 10000 für 10.000 EUR Konto."
        ),
    )
    parser.add_argument(
        "--risiko_pct",
        type=float,
        default=0.01,
        help=(
            "Max. Risiko pro Trade als Anteil des Kapitals (Standard: 0.01 = 1%%). "
            "Beispiel: --risiko_pct 0.02 für 2%% Risiko."
        ),
    )
    parser.add_argument(
        "--kontrakt_groesse",
        type=float,
        default=100_000.0,
        help=(
            "Kontraktgröße 1 Lot in Einheiten (Standard: 100.000 = Standard-Lot). "
            "Für Mini-Lot: --kontrakt_groesse 10000"
        ),
    )

    # ── Feature 2: ATR-basiertes Stop-Loss ───────────────────────────────────
    parser.add_argument(
        "--atr_sl",
        action="store_true",
        default=False,
        help=(
            "ATR-basiertes dynamisches Stop-Loss aktivieren (Standard: aus). "
            "SL = ATR_14 × atr_faktor statt festem --sl_pct."
        ),
    )
    parser.add_argument(
        "--atr_faktor",
        type=float,
        default=1.5,
        help=(
            "ATR-Multiplikator für das dynamische SL (Standard: 1.5). "
            "Beispiel: --atr_faktor 2.0 für 2× ATR als Stop-Loss."
        ),
    )

    # ── Feature 3: Zeitraum-Filterung ────────────────────────────────────────
    parser.add_argument(
        "--zeitraum_von",
        type=str,
        default=None,
        help=(
            "Startdatum für den Backtest-Zeitraum (Standard: 2023-01-01). "
            "Beispiel: --zeitraum_von 2024-01-01"
        ),
    )
    parser.add_argument(
        "--zeitraum_bis",
        type=str,
        default=None,
        help=(
            "Enddatum für den Backtest-Zeitraum (Standard: alle Daten). "
            "Beispiel: --zeitraum_bis 2024-12-31"
        ),
    )

    # ── Feature 5: Swap-Kosten (Overnight-Gebühr) ────────────────────────────
    parser.add_argument(
        "--swap_aktiv",
        action="store_true",
        default=False,
        help=(
            "Swap-Kosten (Overnight-Gebühr) aktivieren (Standard: aus). "
            "Zieht bei Trades die Mitternacht überschreiten den Broker-Swap ab. "
            "Beispiel: --swap_aktiv (Review-Punkt 5)"
        ),
    )

    # ── Feature 4: Transaction Cost Sensitivity Test ──────────────────────────
    parser.add_argument(
        "--spread_faktor",
        type=float,
        default=1.0,
        help=(
            "Spread-Multiplikator für Kosten-Stress-Test (Standard: 1.0 = reale Kosten). "
            "2.0 = doppelte Spreads (Review-Punkt 3). "
            "Beispiel: --spread_faktor 2.0 → Strategie noch profitabel bei 2× Kosten?"
        ),
    )

    # ── H4 Experiment: Zeitrahmen wählen ──────────────────────────────────────
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "H4"],
        help=(
            "Zeitrahmen für Daten und Modell (Standard: H1). "
            "H1 → SYMBOL_H1_labeled.csv + lgbm_SYMBOL_v1.pkl | "
            "H4 → SYMBOL_H4_labeled.csv + lgbm_SYMBOL_H4_v1.pkl. "
            "Vor H4 zuerst h4_pipeline.py + train_model.py --timeframe H4 ausführen!"
        ),
    )

    args = parser.parse_args()

    # Symbole bestimmen (Liste oder 'alle')
    if len(args.symbol) == 1 and args.symbol[0].lower() == "alle":
        ziel_symbole = SYMBOLE
    else:
        ziel_symbole = []
        for sym in args.symbol:
            if sym.upper() in SYMBOLE:
                ziel_symbole.append(sym.upper())
            else:
                print(f"Unbekanntes Symbol: {sym}")
                print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
                return

    # Regime-Filter parsen (z.B. "1,2" → [1, 2])
    regime_erlaubt = None
    if args.regime_filter:
        try:
            regime_erlaubt = [int(r.strip()) for r in args.regime_filter.split(",")]
        except ValueError:
            print(
                f"Ungültiger --regime_filter: '{args.regime_filter}'. "
                f"Erwartet: z.B. '1,2'"
            )
            return

    # RisikoConfig aus CLI-Argumenten zusammenbauen
    risiko_config = RisikoConfig(
        kapital=args.kapital,
        risiko_pct=args.risiko_pct,
        kontrakt_groesse=args.kontrakt_groesse,
        atr_sl=args.atr_sl,
        atr_faktor=args.atr_faktor,
    ) if (args.kapital > 0 or args.atr_sl) else None

    # Backtest-Ausgabe-Ordner anlegen
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    start_zeit = datetime.now()
    regime_info = (
        f"Regimes=[{','.join(REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt)}]"
        if regime_erlaubt
        else "alle Regimes"
    )
    logger.info("=" * 65)
    logger.info("⚠️  ACHTUNG: Teste auf dem heiligen Test-Set (2023+)!")
    logger.info("   Diese Auswertung ist FINAL – Modell danach nicht mehr anpassen!")
    logger.info("=" * 65)
    logger.info(
        f"Symbole: {', '.join(ziel_symbole)} | Version: {args.version} "
        f"| Zeitrahmen: {args.timeframe}"
    )
    logger.info(f"Schwellenwert: {args.schwelle:.0%}")
    logger.info(
        f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | "
        f"RRR={args.tp_pct/args.sl_pct:.1f}:1"
    )
    logger.info(f"Regime-Filter: {regime_info}")
    logger.info(f"Horizon: {args.horizon} {args.timeframe}-Barren")
    if risiko_config:
        if risiko_config.atr_sl:
            logger.info(
                f"ATR-SL: aktiv (Faktor={risiko_config.atr_faktor}×ATR_14)"
            )
        if risiko_config.kapital > 0:
            logger.info(
                f"Kapital: {risiko_config.kapital:,.0f} | "
                f"Risiko: {risiko_config.risiko_pct:.0%} pro Trade"
            )
    if args.zeitraum_von or args.zeitraum_bis:
        logger.info(
            f"Zeitraum-Filter: {args.zeitraum_von or TEST_VON} "
            f"bis {args.zeitraum_bis or 'heute'}"
        )
    if args.spread_faktor != 1.0:
        logger.info(
            f"⚡ SPREAD-STRESS-TEST: Faktor={args.spread_faktor}× "
            f"(Review-Punkt 3: Kosten-Sensitivität)"
        )
    if args.swap_aktiv:
        logger.info("🌙 SWAP-KOSTEN aktiv: Overnight-Gebühren werden eingerechnet")
    logger.info(f"Start: {start_zeit.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # Backtest für alle Symbole
    alle_kennzahlen = []
    for symbol in ziel_symbole:
        kennzahlen = symbol_backtest(
            symbol,
            args.schwelle,
            args.tp_pct,
            args.sl_pct,
            regime_erlaubt,
            args.version,
            args.horizon,
            risiko_config=risiko_config,
            zeitraum_von=args.zeitraum_von,
            zeitraum_bis=args.zeitraum_bis,
            spread_faktor=args.spread_faktor,
            swap_aktiv=args.swap_aktiv,
            timeframe=args.timeframe,
        )
        if kennzahlen:
            alle_kennzahlen.append(kennzahlen)

    # ── Survivorship-Bias Check (Review-Punkt 2) ─────────────────────────────
    # Prüft ob die "guten" Paare nur durch Data-Mining herausgepickt wurden.
    # Wenn das beste Paar deutlich über dem Durchschnitt liegt, besteht das
    # Risiko dass wir die Strategie auf Basis von Zufall optimiert haben.
    if len(alle_kennzahlen) >= 2:
        renditen  = [k["gesamtrendite_pct"] for k in alle_kennzahlen]
        sharpes   = [k["sharpe_ratio"]      for k in alle_kennzahlen]
        drawdowns = [k["max_drawdown_pct"]  for k in alle_kennzahlen]
        gfs       = [k["gewinnfaktor"]      for k in alle_kennzahlen]

        avg_rendite  = float(np.mean(renditen))
        avg_sharpe   = float(np.mean(sharpes))
        avg_gf       = float(np.mean(gfs))
        avg_drawdown = float(np.mean(drawdowns))

        print(f"\n{'─'*75}")
        print(
            f"SURVIVORSHIP-BIAS CHECK (Review-Punkt 2) "
            f"– {len(alle_kennzahlen)} Paare analysiert"
        )
        print(f"{'─'*75}")
        print(
            f"  Ø Alle {len(alle_kennzahlen)} Paare:  "
            f"Rendite={avg_rendite:+.2f}%  "
            f"Sharpe={avg_sharpe:.3f}  "
            f"GF={avg_gf:.3f}  "
            f"DD={avg_drawdown:+.2f}%"
        )
        # Die 2 besten Paare nach Sharpe Ratio hervorheben
        beste = sorted(
            alle_kennzahlen, key=lambda x: x["sharpe_ratio"], reverse=True
        )[:2]
        for b in beste:
            print(
                f"  Bestes [{b['symbol']:6}]:      "
                f"Rendite={b['gesamtrendite_pct']:+.2f}%  "
                f"Sharpe={b['sharpe_ratio']:.3f}  "
                f"GF={b['gewinnfaktor']:.3f}  "
                f"DD={b['max_drawdown_pct']:+.2f}%"
            )
        # Warnung: bestes Paar deutlich über Durchschnitt?
        if max(sharpes) > avg_sharpe * 1.5:
            print(
                "  ⚠️  RISIKO: Bestes Paar ist deutlich über Ø "
                "→ Data-Mining-Bias möglich!"
            )
        else:
            print(
                "  ✅  OK: Bestes Paar nur moderat besser als Ø "
                "→ System erscheint robust."
            )
        print(f"{'─'*75}")

    # ── Regime-Performance-Matrix (Phase 3 Optimierung) ─────────────────────────
    # Zeigt für jede Symbol/Regime-Kombination die Performance an.
    # Ermöglicht eine datengetriebene Entscheidung für --regime_filter.
    if len(ziel_symbole) >= 2:
        matrix_df = regime_matrix_erstellen(ziel_symbole)
        if matrix_df is not None:
            # Tabelle im Terminal ausgeben
            regime_matrix_drucken(matrix_df, regime_info)
            # Heatmap als PNG speichern
            regime_matrix_plotten(matrix_df)
            # Matrix als CSV für spätere Auswertung speichern
            matrix_pfad = BACKTEST_DIR / "regime_performance_matrix.csv"
            matrix_df.to_csv(matrix_pfad, index=False)
            print(f"Matrix CSV: {matrix_pfad}")
            print(f"Matrix PNG: plots/regime_performance_matrix.png")

    # Gesamtzusammenfassung
    ende_zeit = datetime.now()
    dauer_sek = int((ende_zeit - start_zeit).total_seconds())

    print("\n" + "=" * 75)
    stress_label = (
        f" – ⚡ SPREAD-STRESS-TEST ({args.spread_faktor}× Kosten)"
        if args.spread_faktor != 1.0 else ""
    )
    print(f"BACKTEST ABGESCHLOSSEN – Zusammenfassung ({args.version}){stress_label}")
    print(
        f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | "
        f"RRR={args.tp_pct/args.sl_pct:.1f}:1 | "
        f"Horizon={args.horizon} | Regime: {regime_info}"
    )
    if args.spread_faktor != 1.0:
        print(
            f"⚡ Spreads: {args.spread_faktor}× normal → "
            "Sind die Ergebnisse noch profitabel? (Review-Punkt 3)"
        )
    if risiko_config and risiko_config.atr_sl:
        print(f"ATR-SL: {risiko_config.atr_faktor}× ATR_14 (dynamisches Stop-Loss)")
    if risiko_config and risiko_config.kapital > 0:
        print(
            f"Kapital: {risiko_config.kapital:,.0f} | "
            f"Risiko/Trade: {risiko_config.risiko_pct:.0%}"
        )
    print("=" * 75)
    print(
        f"{'Symbol':8} {'Trades':7} {'Rendite%':9} "
        f"{'Win-Rate':9} {'GF':6} {'Sharpe':8} {'Max.DD':8}"
    )
    print(f"{'─' * 75}")

    for k in alle_kennzahlen:
        # Ziel-Icons
        gf_icon = "✅" if k["gewinnfaktor"] > 1.3 else "❌"
        sh_icon = "✅" if k["sharpe_ratio"] > 1.0 else "❌"
        dd_icon = "✅" if k["max_drawdown_pct"] > -20 else "❌"
        abs_str = (
            f" ({k['gesamtrendite_absolut']:+.0f}€)"
            if k.get("gesamtrendite_absolut", 0) != 0
            else ""
        )
        print(
            f"  {k['symbol']:8} {k['n_trades']:5d}   "
            f"{k['gesamtrendite_pct']:+7.2f}%{abs_str:<9}  "
            f"{k['win_rate_pct']:6.1f}%  "
            f"{k['gewinnfaktor']:5.2f}{gf_icon} "
            f"{k['sharpe_ratio']:6.3f}{sh_icon} "
            f"{k['max_drawdown_pct']:+7.2f}%{dd_icon}"
        )

    # Gesamtzusammenfassung als versioniertes CSV speichern
    if alle_kennzahlen:
        zusammenfassung_df = pd.DataFrame(alle_kennzahlen)
        # Versionsunabhängiger Name für v1 (rückwärtskompatibel), sonst mit Suffix
        if args.version == "v1":
            zusammenfassung_pfad = BACKTEST_DIR / "backtest_zusammenfassung.csv"
        else:
            zusammenfassung_pfad = (
                BACKTEST_DIR / f"backtest_zusammenfassung_{args.version}.csv"
            )
        zusammenfassung_df.to_csv(zusammenfassung_pfad, index=False)
        print(f"\nZusammenfassung gespeichert: {zusammenfassung_pfad}")

    print("\nPlots:    plots/SYMBOL_backtest_equity.png")
    print("          plots/SYMBOL_backtest_regime.png")
    print("          plots/SYMBOL_backtest_monatlich.png")
    print("          plots/SYMBOL_backtest_perioden.png")
    print("          plots/regime_performance_matrix.png  ← NEU: Regime-Vergleich")
    print("Trades:   backtest/SYMBOL_trades.csv")
    print("Matrix:   backtest/regime_performance_matrix.csv  ← NEU: Regime-Daten")
    print(f"Laufzeit: {dauer_sek // 60}m {dauer_sek % 60}s")

    # Legende
    print("\nLegende: ✅ Ziel erreicht | ❌ Ziel nicht erreicht")
    print(
        "         GF=Gewinnfaktor (Ziel >1.3) | Sharpe (Ziel >1.0) | Max.DD (Ziel >-20%)"
    )
    print("\nNächster Schritt: live_trader.py auf Windows Laptop einrichten (Phase 6)")
    print("=" * 75)


if __name__ == "__main__":
    main()
