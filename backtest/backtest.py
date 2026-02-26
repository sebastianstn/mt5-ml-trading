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

Eingabe:  models/lgbm_SYMBOL_v1.pkl
          data/SYMBOL_H1_labeled.csv
Ausgabe:  plots/SYMBOL_backtest_equity.png      ← Equity-Kurve
          plots/SYMBOL_backtest_regime.png      ← Performance nach Regime
          plots/SYMBOL_backtest_monatlich.png   ← Monatliche Returns-Heatmap
          backtest/SYMBOL_trades.csv            ← Alle Trades als CSV
          backtest/backtest_zusammenfassung.csv ← Kennzahlen aller Symbole
"""

# Standard-Bibliotheken
import argparse
import logging
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
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import seaborn as sns  # noqa: E402

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
# 1. Daten laden
# ============================================================


def labeled_pfad(symbol: str, version: str = "v1") -> Path:
    """
    Gibt den Pfad zum gelabelten CSV zurück (konsistent mit anderen Skripten).

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
    Lädt das gelabelte Feature-CSV und isoliert das Test-Set (2023+).

    Das Test-Set enthält sowohl die Features für das Modell als auch
    die Rohdaten (OHLC) für die Trade-Simulation.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versions-String für den Datei-Pfad (Standard: "v1")

    Returns:
        DataFrame mit Features, OHLC-Preisen, label und market_regime.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    pfad = labeled_pfad(symbol, version)
    if not pfad.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {pfad}\n"
            f"Zuerst labeling.py --version {version} ausführen!"
        )

    logger.info(f"[{symbol}] Lade {pfad.name} ...")
    df = pd.read_csv(pfad, index_col="time", parse_dates=True)

    # Test-Set isolieren (2023 bis heute)
    df_test = df[df.index >= TEST_VON].copy()

    if len(df_test) == 0:
        raise ValueError(
            f"[{symbol}] Keine Daten ab {TEST_VON}! Datei endet am {df.index[-1].date()}"
        )

    logger.info(
        f"[{symbol}] Test-Set: {len(df_test):,} Kerzen | "
        f"{df_test.index[0].date()} bis {df_test.index[-1].date()}"
    )
    return df_test


# ============================================================
# 2. Modell laden und Signale generieren
# ============================================================


def signale_generieren(
    df: pd.DataFrame,
    symbol: str,
    schwelle: float = 0.55,
    version: str = "v1",
) -> pd.DataFrame:
    """
    Lädt das LightGBM-Modell und generiert Trade-Signale mit Wahrscheinlichkeit.

    Nur Signale über dem Schwellenwert werden als Trade gewertet:
    - Long-Signal:  Modell sagt Klasse 2 UND prob_long  > schwelle
    - Short-Signal: Modell sagt Klasse 0 UND prob_short > schwelle
    - Kein Signal:  Klasse 1 (Neutral) ODER Wahrscheinlichkeit zu niedrig

    Args:
        df:       Test-Set DataFrame mit Features und OHLC
        symbol:   Handelssymbol
        schwelle: Mindest-Wahrscheinlichkeit für einen Trade (Standard: 0.55)
        version:  Versions-String für das Modell-File (Standard: "v1")

    Returns:
        DataFrame mit zusätzlichen Spalten: signal, prob_signal
    """
    # Modell laden (versioniertes Modell-File)
    modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{version}.pkl"
    if not modell_pfad.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Zuerst train_model.py --symbol {symbol} ausführen!"
        )
    logger.info(f"[{symbol}] Lade Modell: {modell_pfad.name}")
    modell = joblib.load(modell_pfad)

    # Features aufbereiten (gleiche Spalten wie beim Training)
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    X = df[feature_spalten].copy()

    # NaN-Werte mit Median auffüllen (Sicherheitsnetz)
    nan_anzahl = X.isna().sum().sum()
    if nan_anzahl > 0:
        logger.warning(f"[{symbol}] {nan_anzahl} NaN-Werte – werden mit Median gefüllt")
        X = X.fillna(X.median())

    logger.info(f"[{symbol}] Berechne Vorhersagen für {len(X):,} Kerzen ...")

    # Wahrscheinlichkeiten für alle 3 Klassen
    # proba[:,0] = Short-Wahrscheinlichkeit
    # proba[:,1] = Neutral-Wahrscheinlichkeit
    # proba[:,2] = Long-Wahrscheinlichkeit
    proba = modell.predict_proba(X)

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


def trade_simulieren(
    df: pd.DataFrame,
    eintritts_index: int,
    richtung: int,
    spread_kosten: float,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    horizon: int = HORIZON,
) -> dict:
    """
    Simuliert einen einzelnen Trade mit Double-Barrier Exit.

    Schritte:
        1. Eintrittspreis = close[T] (Market Order auf Kerzenschluss)
        2. TP/SL-Level berechnen (mit konfigurierbaren Schwellen)
        3. Nächste horizon Kerzen prüfen: welche Schranke wird ZUERST getroffen?
        4. P&L berechnen (inkl. Spread-Kosten)

    Asymmetrisches TP/SL: z.B. tp_pct=0.006, sl_pct=0.003 → RRR=2:1
    → Bei 40% Win-Rate schon profitabel: 0.4×0.6% − 0.6×0.3% = +0.06% p.Trade

    Args:
        df:              Test-Set DataFrame (mit OHLC und index)
        eintritts_index: Integer-Position des Eintrittsbalkens
        richtung:        2=Long, -1=Short
        spread_kosten:   Spread als Anteil des Preises (z.B. 0.0001)
        tp_pct:          Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:          Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        horizon:         Zeitschranke in Kerzen (Standard: HORIZON = 5)

    Returns:
        Dict mit Trade-Ergebnis: pnl_pct, exit_grund, n_bars_gehalten
    """
    n = len(df)

    # Eintrittspreis (Close der Signal-Kerze)
    eintrittspreis = df["close"].iloc[eintritts_index]

    # TP/SL-Level berechnen (mit den übergebenen, konfigurierbaren Schwellen)
    if richtung == 2:  # Long
        tp_level = eintrittspreis * (1.0 + tp_pct)  # Obere Schranke (Ziel)
        sl_level = eintrittspreis * (1.0 - sl_pct)  # Untere Schranke (Stop)
    else:  # Short
        tp_level = eintrittspreis * (1.0 - tp_pct)  # Untere Schranke (Ziel)
        sl_level = eintrittspreis * (1.0 + sl_pct)  # Obere Schranke (Stop)

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
                austritt_pnl = tp_pct
                exit_grund = "tp"
                n_bars = j
                break
            elif low_j <= sl_level:
                # Stop-Loss getroffen → Verlust = SL-Abstand
                austritt_pnl = -sl_pct
                exit_grund = "sl"
                n_bars = j
                break
        else:  # Short
            if low_j <= tp_level:
                # Take-Profit getroffen (Short) → Gewinn
                austritt_pnl = tp_pct
                exit_grund = "tp"
                n_bars = j
                break
            elif high_j >= sl_level:
                # Stop-Loss getroffen (Short) → Verlust
                austritt_pnl = -sl_pct
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
    spread_als_pct = (spread_kosten / eintrittspreis) if eintrittspreis > 10 else spread_kosten
    netto_pnl = austritt_pnl - 2 * spread_als_pct

    return {
        "pnl_pct": netto_pnl,
        "exit_grund": exit_grund,
        "n_bars": n_bars,
        "eintrittspreis": eintrittspreis,
    }


# ============================================================
# 4. Alle Trades für ein Symbol simulieren
# ============================================================


def trades_simulieren(
    df: pd.DataFrame,
    symbol: str,
    schwelle: float = 0.55,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    regime_erlaubt: Optional[list] = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Simuliert alle Trades auf dem Test-Set und gibt eine Trade-Liste zurück.

    Überlappende Trades werden vermieden: Nach einem Trade wird die nächste
    Signal-Kerze erst nach horizon Kerzen gesucht (no re-entry während Trade).

    Verbesserungs-Parameter:
        tp_pct / sl_pct:    Asymmetrisches TP/SL für besseres Chance-Risiko-Verhältnis.
                            Beispiel: tp=0.006, sl=0.003 → RRR=2:1 (nur 40% Win-Rate nötig)
        regime_erlaubt:     Nur in bestimmten Regimes handeln.
                            [1, 2] = nur Aufwärts-/Abwärtstrend (kein Seitwärts, keine Volatilität)
        horizon:            Zeitschranke in Kerzen (muss mit labeling.py übereinstimmen!)

    Args:
        df:             Test-Set DataFrame mit Signalen (aus signale_generieren())
        symbol:         Handelssymbol (für Spread-Kosten und Logging)
        schwelle:       Wahrscheinlichkeits-Schwellenwert (für Logging)
        tp_pct:         Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:         Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        regime_erlaubt: Liste erlaubter Regime-Nummern (None = alle Regimes handeln)
        horizon:        Zeitschranke in Kerzen (Standard: HORIZON = 5)

    Returns:
        DataFrame mit allen simulierten Trades.
    """
    spread_kosten = SPREAD_KOSTEN.get(symbol, 0.000150)  # Fallback: 1.5 Pips

    # Regime-Spalte vorbereiten (falls vorhanden)
    hat_regime = "market_regime" in df.columns

    trades = []  # Ergebnis-Liste
    n_gefiltert_regime = 0  # Zähler: übersprungene Trades wegen Regime-Filter
    i = 0  # Aktueller Balken-Index

    while i < len(df) - horizon:
        signal = df["signal"].iloc[i]

        # Nur Long (2) oder Short (-1) Signale handeln
        if signal in (2, -1):

            # Regime-Filter: Signal überspringen wenn aktuelles Regime nicht erlaubt ist
            if regime_erlaubt is not None and hat_regime:
                aktuelles_regime = df["market_regime"].iloc[i]
                if not np.isnan(aktuelles_regime) and int(aktuelles_regime) not in regime_erlaubt:
                    # Regime nicht erlaubt → Signal ignorieren
                    n_gefiltert_regime += 1
                    i += 1
                    continue

            # Trade simulieren (mit konfigurierbarem TP/SL und Horizon)
            ergebnis = trade_simulieren(df, i, signal, spread_kosten, tp_pct, sl_pct, horizon)

            # Trade-Details speichern
            regime_wert = df["market_regime"].iloc[i] if hat_regime else np.nan
            trades.append(
                {
                    "time": df.index[i],
                    "symbol": symbol,
                    "richtung": "Long" if signal == 2 else "Short",
                    "signal_klasse": signal,
                    "prob": df["prob_signal"].iloc[i],
                    "eintrittspreis": ergebnis["eintrittspreis"],
                    "exit_grund": ergebnis["exit_grund"],
                    "n_bars": ergebnis["n_bars"],
                    "pnl_pct": ergebnis["pnl_pct"],
                    "gewinn": ergebnis["pnl_pct"] > 0,
                    "market_regime": regime_wert,
                }
            )

            # Nächste Signal-Suche: nach dem Trade (keine überlappenden Trades)
            i += ergebnis["n_bars"] + 1
        else:
            i += 1

    if regime_erlaubt is not None and n_gefiltert_regime > 0:
        regime_namen_str = [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        logger.info(
            f"[{symbol}] Regime-Filter aktiv: {n_gefiltert_regime} Signale außerhalb "
            f"[{', '.join(regime_namen_str)}] übersprungen"
        )

    if not trades:
        logger.warning(f"[{symbol}] Keine Trades gefunden! Schwelle zu hoch oder Regime-Filter zu streng?")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df.set_index("time", inplace=True)

    logger.info(
        f"[{symbol}] {len(trades_df)} Trades simuliert | "
        f"Long: {(trades_df['richtung'] == 'Long').sum()} | "
        f"Short: {(trades_df['richtung'] == 'Short').sum()} | "
        f"TP={tp_pct:.1%} / SL={sl_pct:.1%} (RRR={tp_pct/sl_pct:.1f}:1)"
    )

    return trades_df


# ============================================================
# 5. Kennzahlen berechnen
# ============================================================


def kennzahlen_berechnen(
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
    }

    # Log-Ausgabe
    logger.info(f"\n{'─' * 55}")
    logger.info(f"[{symbol}] KENNZAHLEN (Test-Set {kennzahlen['zeitraum_von']} bis {kennzahlen['zeitraum_bis']})")
    logger.info(f"{'─' * 55}")
    logger.info(f"  Anzahl Trades:    {n_trades:6d} (Long: {kennzahlen['n_long']}, Short: {kennzahlen['n_short']})")
    logger.info(f"  Gesamtrendite:    {gesamtrendite:+7.2f}%")
    logger.info(f"  Win-Rate:         {win_rate:7.1f}%")
    logger.info(f"  Gewinnfaktor:     {gewinnfaktor:7.3f}  (Ziel: >1.3)")
    logger.info(f"  Sharpe Ratio:     {sharpe:7.3f}  (Ziel: >1.0)")
    logger.info(f"  Max. Drawdown:    {max_drawdown:+7.2f}%  (Ziel: >-20%)")
    logger.info(f"  Exits: TP={kennzahlen['tp_hits']} | SL={kennzahlen['sl_hits']} | Horizon={kennzahlen['horizon_exits']}")

    # Zielampel
    ziele = {
        "Gewinnfaktor > 1.3": gewinnfaktor > 1.3,
        "Sharpe > 1.0": sharpe > 1.0,
        "Drawdown > -20%": max_drawdown > -20,
        "Win-Rate > 45%": win_rate > 45,
    }
    logger.info(f"\n  Ziel-Check:")
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
        regime_trades = trades_mit_regime[trades_mit_regime["market_regime"] == regime_nr]
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
# 7. Equity-Kurve plotten
# ============================================================


def equity_kurve_plotten(trades_df: pd.DataFrame, symbol: str, kennzahlen: dict) -> None:
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


def regime_plotten(regime_df: pd.DataFrame, symbol: str) -> None:
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
    farben_wr = [
        "#2ECC71" if w >= 50 else "#E74C3C" for w in regime_df["win_rate_pct"]
    ]
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

    # Anzahl Trades als Annotation
    for ax, col in [(ax1, "gesamtrendite_pct"), (ax2, "win_rate_pct")]:
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


def monatliche_heatmap_plotten(trades_df: pd.DataFrame, symbol: str) -> None:
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
        trades_df["pnl_pct"]
        .resample("ME")  # Monatsende-Resampling
        .sum()
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
    pivot = monatlich_df.pivot_table(values="Rendite", index="Jahr", columns="Monat", aggfunc="sum")

    # Monatsnamen für x-Achse
    monat_namen = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                   "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    pivot.columns = [monat_namen[int(m) - 1] for m in pivot.columns]

    # Heatmap erstellen
    fig, ax = plt.subplots(figsize=(13, max(3, len(pivot) * 0.8 + 2)))
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
# 10. Vollständiger Backtest für ein Symbol
# ============================================================


def symbol_backtest(
    symbol: str,
    schwelle: float = 0.55,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    regime_erlaubt: Optional[list] = None,
    version: str = "v1",
    horizon: int = HORIZON,
) -> Optional[dict]:
    """
    Führt den vollständigen Backtest für ein Symbol durch.

    Schritte:
        1. Test-Set laden (2023+, versioniertes CSV)
        2. Signale generieren (versioniertes Modell + Schwellenwert)
        3. Trades simulieren (TP/SL/Horizon + Spread + Regime-Filter)
        4. Kennzahlen berechnen
        5. Regime-Analyse
        6. Plots erstellen
        7. Trade-Log als CSV speichern

    Args:
        symbol:         Handelssymbol (z.B. "EURUSD")
        schwelle:       Wahrscheinlichkeits-Schwellenwert (Standard: 0.55)
        tp_pct:         Take-Profit-Abstand (Standard: TP_PCT = 0.3%)
        sl_pct:         Stop-Loss-Abstand   (Standard: SL_PCT = 0.3%)
        regime_erlaubt: Nur in diesen Regimes handeln (None = alle)
        version:        Versions-String für Modell- und Datei-Pfade (Standard: "v1")
        horizon:        Zeitschranke in Kerzen (muss mit labeling.py übereinstimmen!)

    Returns:
        Dict mit Kennzahlen oder None bei Fehler.
    """
    # Info-String für Logging und Dateinamen
    regime_info = (
        f"Regimes=[{','.join(str(r) for r in regime_erlaubt)}]"
        if regime_erlaubt else "alle Regimes"
    )
    logger.info(f"\n{'=' * 65}")
    logger.info(
        f"Backtest – {symbol} ({version}) | Schwelle: {schwelle:.0%} | "
        f"TP={tp_pct:.2%} / SL={sl_pct:.2%} | Horizon={horizon} | {regime_info}"
    )
    logger.info(f"{'=' * 65}")

    try:
        # Schritt 1: Test-Set laden (versioniertes CSV)
        df = daten_laden(symbol, version)

        # Schritt 2: Signale generieren (versioniertes Modell)
        df = signale_generieren(df, symbol, schwelle, version)

        # Schritt 3: Trades simulieren (mit TP/SL + Regime-Filter + Horizon)
        trades_df = trades_simulieren(df, symbol, schwelle, tp_pct, sl_pct, regime_erlaubt, horizon)

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

        # Schritt 5: Regime-Analyse
        regime_df = regime_analyse(trades_df, symbol)

        # Schritt 6: Plots erstellen
        equity_kurve_plotten(trades_df, symbol, kennzahlen)
        regime_plotten(regime_df, symbol)
        monatliche_heatmap_plotten(trades_df, symbol)

        # Schritt 7: Trade-Log als CSV speichern
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        trade_pfad = BACKTEST_DIR / f"{symbol}_trades.csv"
        trades_df.to_csv(trade_pfad)
        logger.info(f"[{symbol}] Trade-Log gespeichert: {trade_pfad} ({len(trades_df)} Trades)")

        return kennzahlen

    except FileNotFoundError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"[{symbol}] Unerwarteter Fehler: {e}", exc_info=True)
        return None


# ============================================================
# 11. Hauptprogramm
# ============================================================


def main() -> None:
    """Backtest für ein oder alle Symbole."""

    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – Backtest auf Test-Set (2023+)"
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
            f"Take-Profit als Anteil (Standard: {TP_PCT} = {TP_PCT:.1%}). "
            "Beispiel: --tp_pct 0.006 für 0.6% TP bei 0.3% SL (RRR=2:1)."
        ),
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=SL_PCT,
        help=(
            f"Stop-Loss als Anteil (Standard: {SL_PCT} = {SL_PCT:.1%}). "
            "Beispiel: --sl_pct 0.003 für 0.3% SL."
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

    # Regime-Filter parsen (z.B. "1,2" → [1, 2])
    regime_erlaubt = None
    if args.regime_filter:
        try:
            regime_erlaubt = [int(r.strip()) for r in args.regime_filter.split(",")]
        except ValueError:
            print(f"Ungültiger --regime_filter: '{args.regime_filter}'. Erwartet: z.B. '1,2'")
            return

    # Backtest-Ausgabe-Ordner anlegen
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    start_zeit = datetime.now()
    regime_info = (
        f"Regimes=[{','.join(REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt)}]"
        if regime_erlaubt else "alle Regimes"
    )
    logger.info("=" * 65)
    logger.info("⚠️  ACHTUNG: Teste auf dem heiligen Test-Set (2023+)!")
    logger.info("   Diese Auswertung ist FINAL – Modell danach nicht mehr anpassen!")
    logger.info("=" * 65)
    logger.info(f"Symbole: {', '.join(ziel_symbole)} | Version: {args.version}")
    logger.info(f"Schwellenwert: {args.schwelle:.0%}")
    logger.info(f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | RRR={args.tp_pct/args.sl_pct:.1f}:1")
    logger.info(f"Regime-Filter: {regime_info}")
    logger.info(f"Horizon: {args.horizon} H1-Barren")
    logger.info(f"Start: {start_zeit.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    # Backtest für alle Symbole
    alle_kennzahlen = []
    for symbol in ziel_symbole:
        kennzahlen = symbol_backtest(
            symbol, args.schwelle, args.tp_pct, args.sl_pct,
            regime_erlaubt, args.version, args.horizon
        )
        if kennzahlen:
            alle_kennzahlen.append(kennzahlen)

    # Gesamtzusammenfassung
    ende_zeit = datetime.now()
    dauer_sek = int((ende_zeit - start_zeit).total_seconds())

    print("\n" + "=" * 75)
    print(f"BACKTEST ABGESCHLOSSEN – Zusammenfassung ({args.version})")
    print(f"TP={args.tp_pct:.2%} | SL={args.sl_pct:.2%} | RRR={args.tp_pct/args.sl_pct:.1f}:1 | Horizon={args.horizon} | Regime: {regime_info}")
    print("=" * 75)
    print(f"{'Symbol':8} {'Trades':7} {'Rendite':9} {'Win-Rate':9} {'GF':6} {'Sharpe':8} {'Max.DD':8}")
    print(f"{'─' * 75}")

    for k in alle_kennzahlen:
        # Ziel-Icons
        gf_icon = "✅" if k["gewinnfaktor"] > 1.3 else "❌"
        sh_icon = "✅" if k["sharpe_ratio"] > 1.0 else "❌"
        dd_icon = "✅" if k["max_drawdown_pct"] > -20 else "❌"
        print(
            f"  {k['symbol']:8} {k['n_trades']:5d}   "
            f"{k['gesamtrendite_pct']:+7.2f}%  "
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
            zusammenfassung_pfad = BACKTEST_DIR / f"backtest_zusammenfassung_{args.version}.csv"
        zusammenfassung_df.to_csv(zusammenfassung_pfad, index=False)
        print(f"\nZusammenfassung gespeichert: {zusammenfassung_pfad}")

    print(f"\nPlots:    plots/SYMBOL_backtest_equity.png")
    print(f"          plots/SYMBOL_backtest_regime.png")
    print(f"          plots/SYMBOL_backtest_monatlich.png")
    print(f"Trades:   backtest/SYMBOL_trades.csv")
    print(f"Laufzeit: {dauer_sek // 60}m {dauer_sek % 60}s")

    # Legende
    print("\nLegende: ✅ Ziel erreicht | ❌ Ziel nicht erreicht")
    print("         GF=Gewinnfaktor (Ziel >1.3) | Sharpe (Ziel >1.0) | Max.DD (Ziel >-20%)")
    print("\nNächster Schritt: live_trader.py auf Windows Laptop einrichten (Phase 6)")
    print("=" * 75)


if __name__ == "__main__":
    main()
