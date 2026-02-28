"""
h4_pipeline.py – Vollständige H4 Daten-Pipeline (Resample + Features + Labels)

Warum H4?
    - H4-Kerzen sind 4× größer als H1 → weniger Rauschen, stärkere Signale
    - 0.3% TP/SL-Bewegung ist in 4 Stunden leichter zu erreichen als in 1 Stunde
    - Der Markt "atmet" auf H4 klarer: Trends und Umkehrungen besser erkennbar
    - Weniger Trades, aber potenziell bessere Trefferquote (Signal-to-Noise)

Ablauf für jedes Symbol:
    1. H1-Rohdaten lesen (data/SYMBOL_H1.csv)
    2. H1 → H4 resamplen (OHLCV-Aggregation: open=first, high=max, low=min,
       close=last, volume=sum, spread=mean)
    3. Technische Indikatoren berechnen (importiert aus feature_engineering.py)
    4. H4-spezifische Multi-Timeframe-Features: D1 und W1 Trends
    5. Externe Features: Fear & Greed + BTC-Funding-Rate (mit .shift(1))
    6. Regime Detection: ADX(14) + SMA50 + ATR% → market_regime 0–3
    7. Double-Barrier Labeling: TP=SL=0.3%, Horizon=5 H4-Barren (= 20 Stunden)
    8. NaN bereinigen + als data/SYMBOL_H4_labeled.csv speichern

Läuft auf: Linux-Server (/mnt/1T-Data/XGBoost-LightGBM/)

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/h4_pipeline.py                   # alle 7 Symbole
    python features/h4_pipeline.py --symbol EURUSD   # ein Symbol

Eingabe:  data/SYMBOL_H1.csv
          data/fear_greed.csv
          data/btc_funding_rate.csv
Ausgabe:  data/SYMBOL_H4_labeled.csv
"""

# pylint: disable=duplicate-code

# Standard-Bibliotheken
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Datenverarbeitung
import numpy as np
import pandas as pd

# Technische Indikatoren (pandas_ta für ADX)
import pandas_ta  # noqa: F401  # pylint: disable=unused-import

# Feature-Funktionen aus feature_engineering.py importieren
# Vermeidet Code-Duplizierung – selbe bewährten Funktionen für H4
_FE_DIR = Path(__file__).parent
sys.path.insert(0, str(_FE_DIR))

from feature_engineering import (  # noqa: E402  # pylint: disable=wrong-import-position
    ind_sma,
    ind_rsi,
    trend_features,
    momentum_features,
    volatility_features,
    volume_features,
    kerzenmuster_features,
    zeitbasierte_features,
    nan_bereinigung,
)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("h4_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Alle 7 Forex-Hauptpaare
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# ============================================================
# Labeling-Parameter (auf H4 optimiert)
# ============================================================
# H4 ATR ≈ 0.3–0.5% pro Bar → TP/SL=0.3% ist gut erreichbar
# Horizon=5 H4-Barren = 20 Stunden = ausreichend Zeit für den Move
TP_PCT = 0.003   # Take-Profit: 0.3% des Close-Preises
SL_PCT = 0.003   # Stop-Loss:   0.3% symmetrisch
HORIZON = 5      # Zeitschranke: 5 H4-Kerzen (= 20 Stunden)


# ============================================================
# 1. H1 → H4 Resampling
# ============================================================


def h1_zu_h4_resamplen(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert H1-OHLCV-Daten auf H4 (4-Stunden-Kerzen).

    OHLCV-Aggregationsregeln:
        Open   = erste H1-Kerze des H4-Blocks
        High   = Maximum aller 4 H1-Kerzen
        Low    = Minimum aller 4 H1-Kerzen
        Close  = letzte H1-Kerze des H4-Blocks
        Volume = Summe aller 4 H1-Kerzen
        Spread = Mittelwert aller 4 H1-Kerzen (Schätzung des H4-Spreads)

    Leere Blöcke (Wochenende, Feiertage) werden entfernt.

    Args:
        df_h1: H1 OHLCV DataFrame mit DatetimeIndex

    Returns:
        H4 OHLCV DataFrame (~1/4 der Zeilen)
    """
    agg_regeln = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Spread nur aggregieren wenn vorhanden
    if "spread" in df_h1.columns:
        agg_regeln["spread"] = "mean"

    df_h4 = df_h1.resample("4h").agg(agg_regeln)

    # Leere Blöcke entfernen (Wochenende hat kein Forex-Volumen)
    df_h4 = df_h4.dropna(subset=["close"])

    logger.info(
        "Resampling: %s H1-Kerzen → %s H4-Kerzen (%s%%)",
        f"{len(df_h1):,}",
        f"{len(df_h4):,}",
        f"{len(df_h4) / len(df_h1) * 100:.1f}",
    )

    return df_h4


# ============================================================
# 2. H4-spezifische Multi-Timeframe-Features
# ============================================================


def multitimeframe_h4_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Multi-Timeframe-Features für H4-Daten.

    Für H4 sind die relevanten höheren Zeitrahmen D1 und W1:
        - trend_d1: Tages-Trend (SMA20 > SMA50 auf D1-Daten)
        - rsi_d1:   Tages-RSI(14) als Momentum-Indikator
        - trend_w1: Wochen-Trend (SMA10 > SMA20 auf W1-Daten)

    WICHTIG – kein Look-Ahead-Bias:
        Alle höheren Zeitrahmen werden mit .shift(1) verzögert, bevor
        sie per forward-fill auf H4-Kerzen übertragen werden.

    Args:
        df: H4 OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit drei neuen Multi-Timeframe-Spalten.
    """
    result = df.copy()
    close = df["close"]

    # --- Tages-Trend (D1) ---
    close_d1 = close.resample("1d").last().dropna()
    sma_d1_20 = ind_sma(close_d1, 20)
    sma_d1_50 = ind_sma(close_d1, 50)

    # +1 = SMA20 > SMA50 (bullish), -1 = bearish, 0 = kein klarer Trend
    trend_d1 = np.sign(sma_d1_20 - sma_d1_50).fillna(0)
    # .shift(1): Kerze von heute → erst morgen als Feature
    trend_d1_delayed = trend_d1.shift(1)
    result["trend_d1"] = trend_d1_delayed.reindex(result.index, method="ffill")

    # D1-RSI(14) als Momentum-Feature auf Tagesbasis
    rsi_d1 = ind_rsi(close_d1, length=14)
    rsi_d1_delayed = rsi_d1.shift(1)
    result["rsi_d1"] = rsi_d1_delayed.reindex(result.index, method="ffill")

    # --- Wochen-Trend (W1) ---
    close_w1 = close.resample("1W").last().dropna()
    sma_w1_10 = ind_sma(close_w1, 10)
    sma_w1_20 = ind_sma(close_w1, 20)

    # +1 = Aufwärtstrend auf Wochenbasis, -1 = Abwärtstrend
    trend_w1 = np.sign(sma_w1_10 - sma_w1_20).fillna(0)
    trend_w1_delayed = trend_w1.shift(1)
    result["trend_w1"] = trend_w1_delayed.reindex(result.index, method="ffill")

    logger.info("Multi-Timeframe H4-Features: D1-Trend, D1-RSI, W1-Trend ✓")
    return result


# ============================================================
# 3. Externe Features hinzufügen
# ============================================================


def externe_features_hinzufuegen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt externe Marktdaten als Features hinzu.

    Fear & Greed Index (täglich):
        Tages-Stimmungsindikator 0–100. Mit .shift(1) verschoben → heute
        wird erst morgen genutzt. Tageswert → H4 per forward-fill.

    BTC Funding Rate (8-stündlich):
        Proxy für Risk-On/Risk-Off. Mit .shift(1) verschoben.
        Gut geeignet für H4 da 8h-Daten auf H4-Barren passen.

    Wenn Dateien nicht vorhanden → Spalten werden mit 0 gefüllt.

    Args:
        df: H4 DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit bis zu 3 neuen Spalten.
    """
    result = df.copy()

    # --- Fear & Greed Index (täglich) ---
    fg_pfad = DATA_DIR / "fear_greed.csv"
    if fg_pfad.exists():
        fg = pd.read_csv(fg_pfad, index_col="time", parse_dates=True)
        fg = fg.sort_index()

        # .shift(1): Stimmung von heute → erst morgen als Feature
        # (verhindert Look-Ahead-Bias)
        fg = fg.shift(1)

        # Täglich → H4: Merge über datetime-Index, dann forward-fill
        result = result.merge(
            fg[["fear_greed_value", "fear_greed_class"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        result["fear_greed_value"] = result["fear_greed_value"].ffill().fillna(50.0)
        result["fear_greed_class"] = result["fear_greed_class"].ffill().fillna(1.0)
        logger.info("Externe Features: Fear & Greed Index geladen ✓")
    else:
        # Fallback: neutrale Standardwerte
        result["fear_greed_value"] = 50.0
        result["fear_greed_class"] = 1.0
        logger.warning("fear_greed.csv nicht gefunden – Standardwerte (50.0 / 1.0)")

    # --- BTC Funding Rate (8-stündlich) ---
    btc_pfad = DATA_DIR / "btc_funding_rate.csv"
    if btc_pfad.exists():
        btc = pd.read_csv(btc_pfad, index_col="time", parse_dates=True)
        btc = btc.sort_index()

        # .shift(1): Funding-Rate von 08:00 → erst 16:00 als Feature
        btc = btc.shift(1)

        result = result.merge(
            btc[["btc_funding_rate"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        result["btc_funding_rate"] = result["btc_funding_rate"].ffill().fillna(0.0)
        logger.info("Externe Features: BTC Funding Rate geladen ✓")
    else:
        result["btc_funding_rate"] = 0.0
        logger.warning("btc_funding_rate.csv nicht gefunden – Standardwert (0.0)")

    return result


# ============================================================
# 4. Regime Detection (ADX + ATR% + SMA50)
# ============================================================


def regime_berechnen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Klassifiziert jede H4-Kerze in eine der 4 Marktphasen.

    Identische Logik wie regime_detection.py für H1-Daten:
        0 = Seitwärts     (kein klarer Trend)
        1 = Aufwärtstrend (ADX>25, Close > SMA50, normal volatile)
        2 = Abwärtstrend  (ADX>25, Close < SMA50, normal volatile)
        3 = Hohe Vola     (ATR% > 1.5 × rollender Median ATR%)

    Priorität: Hohe Vola > Trend > Seitwärts

    Auf H4-Daten ist die Regime-Erkennung stabiler als auf H1,
    da weniger Rauschen → weniger falsche Regime-Wechsel.

    Args:
        df: H4 DataFrame (benötigt 'close', 'sma_50', 'atr_pct')

    Returns:
        DataFrame mit neuen Spalten: 'adx_14', 'market_regime'
    """
    result = df.copy()

    # ADX(14) via pandas_ta
    # pandas_ta erwartet DataFrame mit high/low/close
    adx_df = result.ta.adx(length=14)
    adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
    result["adx_14"] = adx_df[adx_col]

    # ATR% prüfen (aus volatility_features() berechnet)
    if "atr_pct" not in result.columns:
        logger.error("atr_pct fehlt – volatility_features() zuerst ausführen!")
        result["adx_14"] = np.nan
        result["market_regime"] = -1
        return result

    # Rollende Volatilitätsschwelle (50 H4-Barren ≈ 8.3 Handelstage)
    median_atr = result["atr_pct"].rolling(window=50, min_periods=50).median()

    # Regime-Klassifikation (vektorisiert = schnell)
    regime = pd.Series(0, index=result.index, dtype=int)  # Standard: Seitwärts

    hoch_vol = result["atr_pct"] > (1.5 * median_atr)
    aufwaerts = (
        (result["adx_14"] > 25)
        & (result["close"] > result["sma_50"])
        & ~hoch_vol
    )
    abwaerts = (
        (result["adx_14"] > 25)
        & (result["close"] < result["sma_50"])
        & ~hoch_vol
    )

    regime[aufwaerts] = 1
    regime[abwaerts] = 2
    regime[hoch_vol] = 3

    # Anfangsperiode ohne gültige Indikatoren als -1 (NaN-Markierung)
    nan_maske = result["adx_14"].isna() | median_atr.isna()
    regime[nan_maske] = -1

    result["market_regime"] = regime

    # Regime-Verteilung loggen
    gueltig = regime[regime >= 0]
    namen = {0: "Seitwärts", 1: "Aufwärts", 2: "Abwärts", 3: "Hohe Vola"}
    for reg_nr in [0, 1, 2, 3]:
        anteil = (gueltig == reg_nr).sum() / len(gueltig) * 100
        logger.info("  Regime %s (%s): %s%%", reg_nr, namen[reg_nr], f"{anteil:.1f}")

    logger.info("Regime Detection auf H4: ADX(14) + ATR%% + SMA50 ✓")
    return result


# ============================================================
# 5. Double-Barrier Labeling
# ============================================================


def double_barrier_label_h4(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    horizon: int = HORIZON,
) -> np.ndarray:
    """
    Double-Barrier Labels für H4-Kerzen.

    Für jede H4-Kerze T: schaut in die nächsten `horizon` H4-Kerzen.
        - Wenn High[T+j] >= close[T] * (1 + tp_pct) zuerst: Label  1 (Long)
        - Wenn Low[T+j]  <= close[T] * (1 - sl_pct) zuerst: Label -1 (Short)
        - Weder noch bis Horizon:                             Label  0 (Neutral)

    Mit H4-Kerzen:
        TP=SL=0.3%, Horizon=5 H4 = 20 Stunden Lookahead
        Erwartete Verteilung: ≈25% Long, ≈25% Short, ≈50% Neutral
        (besser als H1 mit 18-22% Long/Short wegen größerer Bars)

    Args:
        close:   Close-Preise (numpy Array)
        high:    High-Preise
        low:     Low-Preise
        tp_pct:  Take-Profit Schwelle (Standard: 0.3%)
        sl_pct:  Stop-Loss Schwelle   (Standard: 0.3%)
        horizon: H4-Kerzen voraus schauen (Standard: 5)

    Returns:
        numpy Array mit Labels (-1, 0, 1), letzte `horizon` Werte = NaN
    """
    n = len(close)
    labels = np.full(n, np.nan)  # NaN als Platzhalter

    for i in range(n - horizon):
        # Barrieren ab Close-Preis zum Einstiegszeitpunkt T
        tp_level = close[i] * (1.0 + tp_pct)   # Obere Barriere (Long-TP)
        sl_level = close[i] * (1.0 - sl_pct)   # Untere Barriere (Short-TP)

        label = 0  # Standard: keine Barriere getroffen

        # Welche Barriere wird ZUERST berührt?
        for j in range(1, horizon + 1):
            if high[i + j] >= tp_level:
                label = 1   # Obere Barriere → Long Signal
                break
            if low[i + j] <= sl_level:
                label = -1  # Untere Barriere → Short Signal
                break

        labels[i] = label

    return labels


def label_verteilung_pruefen(df: pd.DataFrame, symbol: str) -> None:
    """
    Zeigt Label-Verteilung im Log. Warnt bei starkem Ungleichgewicht.

    Args:
        df:     DataFrame mit 'label'-Spalte
        symbol: Symbol-Name (für Log-Ausgabe)
    """
    gueltig = df["label"].dropna()
    anz = len(gueltig)
    verteilung = gueltig.value_counts().sort_index()

    logger.info("\n[%s] H4 Label-Verteilung (%s gültige Kerzen):", symbol, f"{anz:,}")
    namen = {-1: "Short (-1)", 0: "Kein Signal (0)", 1: "Long  (+1)"}
    for label_nr in [-1, 0, 1]:
        anzahl = verteilung.get(label_nr, 0)
        anteil = anzahl / anz if anz > 0 else 0
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

    # Warnung wenn Klasse < 10%
    for label_nr in [-1, 0, 1]:
        anteil = verteilung.get(label_nr, 0) / anz if anz > 0 else 0
        if anteil < 0.10:
            logger.warning(
                "[%s] Klasse %s nur %s%% – TP/SL oder Horizon anpassen?",
                symbol,
                label_nr,
                f"{anteil * 100:.1f}",
            )


# ============================================================
# 6. Hauptfunktion: Symbol verarbeiten
# ============================================================


def symbol_h4_verarbeiten(symbol: str) -> bool:
    """
    Vollständige H4-Pipeline für ein Symbol.

    Schritte:
        1. H1-Daten laden
        2. H1 → H4 resamplen
        3. Technische Features berechnen
        4. H4-spezifische MTF-Features (D1, W1)
        5. Externe Features (Fear&Greed, BTC-Funding)
        6. Regime Detection
        7. NaN-Bereinigung (Warm-Up entfernen)
        8. Double-Barrier Labels berechnen
        9. Speichern als SYMBOL_H4_labeled.csv

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    h1_pfad = DATA_DIR / f"{symbol}_H1.csv"
    ausgabe_pfad = DATA_DIR / f"{symbol}_H4_labeled.csv"

    # Eingabedatei prüfen
    if not h1_pfad.exists():
        logger.error("[%s] H1-Datei nicht gefunden: %s", symbol, h1_pfad)
        return False

    try:
        # --- 1. H1-Daten laden ---
        logger.info("[%s] Lade H1-Daten: %s ...", symbol, h1_pfad.name)
        df_h1 = pd.read_csv(h1_pfad, index_col="time", parse_dates=True)
        df_h1 = df_h1.sort_index()  # Sicherheitshalber sortieren
        logger.info(
            "[%s] H1: %s Kerzen | %s bis %s",
            symbol,
            f"{len(df_h1):,}",
            df_h1.index[0].date(),
            df_h1.index[-1].date(),
        )

        # --- 2. H1 → H4 resamplen ---
        df = h1_zu_h4_resamplen(df_h1)

        # --- 3. Technische Features (aus feature_engineering.py) ---
        logger.info("[%s] Berechne technische Features ...", symbol)
        df = trend_features(df)            # SMA, EMA, MACD, Preis-Ratios
        df = momentum_features(df)         # RSI, Stochastic, Williams %R, ROC
        df = volatility_features(df)       # ATR, Bollinger Bands, Hist-Vol
        df = volume_features(df)           # OBV, Volume-Ratio, Volume-ROC
        df = kerzenmuster_features(df)     # Body, Wicks, Hammer, Doji etc.
        df = zeitbasierte_features(df)     # Stunde, Wochentag, Sitzungen

        # --- 4. H4-spezifische MTF-Features ---
        df = multitimeframe_h4_features(df)   # D1-Trend, D1-RSI, W1-Trend

        # --- 5. Externe Features ---
        df = externe_features_hinzufuegen(df)

        # --- 6. Regime Detection ---
        df = regime_berechnen(df)

        # --- 7. NaN-Bereinigung (Warm-Up-Perioden entfernen) ---
        # SMA(200) braucht 200 Bars → erste 199 H4-Kerzen haben NaN
        # Aber: NaN-Zeilen mit market_regime=-1 separat behandeln
        n_vorher = len(df)
        # Regime=-1 Zeilen entfernen (Anfangsperiode ohne gültige Indikatoren)
        df = df[df["market_regime"] >= 0]
        # Dann vollständige NaN-Bereinigung
        df = nan_bereinigung(df)
        logger.info(
            "[%s] NaN-Bereinigung: %s → %s Kerzen (%s entfernt)",
            symbol,
            f"{n_vorher:,}",
            f"{len(df):,}",
            f"{n_vorher - len(df):,}",
        )

        if len(df) < 500:
            logger.error(
                "[%s] Zu wenige Kerzen nach Bereinigung: %s (min. 500 nötig)",
                symbol,
                len(df),
            )
            return False

        # --- 8. Double-Barrier Labels ---
        logger.info(
            "[%s] Berechne H4-Labels (TP=SL=%s, Horizon=%s H4-Barren = %s Stunden) ...",
            symbol,
            f"{TP_PCT:.2%}",
            HORIZON,
            HORIZON * 4,
        )
        labels = double_barrier_label_h4(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            tp_pct=TP_PCT,
            sl_pct=SL_PCT,
            horizon=HORIZON,
        )
        df["label"] = labels

        # Letzte HORIZON Zeilen (kein vollständiger Vorausblick) entfernen
        n_vor_label = len(df)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)   # float → int
        logger.info(
            "[%s] Labels: %s Zeilen entfernt, %s Kerzen final",
            symbol,
            n_vor_label - len(df),
            f"{len(df):,}",
        )

        # Label-Verteilung analysieren
        label_verteilung_pruefen(df, symbol)

        # --- 9. Speichern ---
        df.to_csv(ausgabe_pfad)
        groesse_mb = ausgabe_pfad.stat().st_size / 1024 / 1024
        logger.info(
            "[%s] Gespeichert: %s | %s Kerzen | %s Spalten | %.1f MB",
            symbol,
            ausgabe_pfad.name,
            f"{len(df):,}",
            len(df.columns),
            groesse_mb,
        )

    except (ValueError, KeyError, pd.errors.EmptyDataError) as e:
        logger.error("[%s] Fehler in der Pipeline: %s", symbol, e)
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[%s] Unerwarteter Fehler: %s", symbol, e)
        return False

    return True


# ============================================================
# 7. Hauptprogramm (CLI)
# ============================================================


def main() -> None:
    """Startet die H4-Pipeline für alle oder ausgewählte Symbole."""

    parser = argparse.ArgumentParser(
        description="MT5 ML-Trading – H4 Daten-Pipeline"
    )
    parser.add_argument(
        "--symbol",
        default="alle",
        help=(
            "Handelssymbol oder 'alle' (Standard: alle). "
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

    logger.info("=" * 65)
    logger.info("H4 Pipeline – H1→H4 Resample + Features + Labels")
    logger.info("Parameter: TP=SL=%s, Horizon=%s H4-Barren (=%s Stunden)",
                f"{TP_PCT:.2%}", HORIZON, HORIZON * 4)
    logger.info("Symbole: %s", ", ".join(ziel_symbole))
    logger.info("=" * 65)

    ergebnisse = []
    for symbol in ziel_symbole:
        logger.info("\n%s", "─" * 45)
        logger.info("Verarbeite: %s", symbol)
        logger.info("%s", "─" * 45)
        erfolg = symbol_h4_verarbeiten(symbol)
        ergebnisse.append((symbol, "OK" if erfolg else "FEHLER"))

    # Zusammenfassung
    print("\n" + "=" * 65)
    print("H4 PIPELINE – ABGESCHLOSSEN")
    print(f"Parameter: TP=SL={TP_PCT:.2%} | Horizon={HORIZON} H4-Barren ({HORIZON*4} Stunden)")
    print("=" * 65)
    for symbol, status in ergebnisse:
        zeichen = "✓" if status == "OK" else "✗"
        print(f"  {zeichen} {symbol}: {status}")

    erfolge = [r for r in ergebnisse if r[1] == "OK"]
    print(f"\n{len(erfolge)}/{len(ziel_symbole)} Symbole erfolgreich verarbeitet.")
    print(f"\nGelabelte H4-Daten: data/SYMBOL_H4_labeled.csv")
    print(
        f"\nNächster Schritt: python train_model.py --symbol alle "
        f"--timeframe H4 --trials 50"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
