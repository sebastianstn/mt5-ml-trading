"""
feature_engineering.py – Technische Indikatoren und Features berechnen.

Läuft auf: Linux-Server
Input:     data/EURUSD_H1.csv
Output:    data/EURUSD_H1_features.csv

WICHTIG – Look-Ahead-Bias-Prävention:
    Alle Indikatoren verwenden NUR Daten bis zur aktuellen Kerze (t).
    Das bedeutet: Feature[t] darf KEINE Information aus Kerze t+1 enthalten.
    - pandas_ta-Indikatoren sind standardmäßig korrekt (kein Future-Leak)
    - Multi-Timeframe-Features werden mit .shift(1) gesichert

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/feature_engineering.py
"""

# Standard-Bibliotheken
import logging
from pathlib import Path

# Datenverarbeitung
import numpy as np
import pandas as pd
import pandas_ta as ta

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 1. Daten laden
# ============================================================


def daten_laden(dateipfad: Path) -> pd.DataFrame:
    """
    Lädt OHLCV-Rohdaten aus CSV.

    Args:
        dateipfad: Pfad zur EURUSD_H1.csv

    Returns:
        DataFrame mit DatetimeIndex (UTC), sortiert aufsteigend.
    """
    df = pd.read_csv(dateipfad, index_col="time", parse_dates=True)
    df.sort_index(inplace=True)
    logger.info(f"Rohdaten geladen: {len(df):,} Kerzen ({df.index[0]} – {df.index[-1]})")
    return df


# ============================================================
# 2. Trend-Features
# ============================================================


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Trend-Indikatoren.

    SMA/EMA zeigen die Marktrichtung, MACD misst die Dynamik des Trends.
    Price-to-SMA-Ratio: Wie weit ist der Preis vom Durchschnitt entfernt?

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit neuen Trend-Spalten.
    """
    result = df.copy()

    # --- Simple Moving Averages ---
    result["sma_20"] = ta.sma(result["close"], length=20)
    result["sma_50"] = ta.sma(result["close"], length=50)
    result["sma_200"] = ta.sma(result["close"], length=200)

    # Preis-zu-SMA Ratio: positive Werte = Preis über dem SMA (bullish)
    result["price_sma20_ratio"] = (result["close"] - result["sma_20"]) / result["sma_20"]
    result["price_sma50_ratio"] = (result["close"] - result["sma_50"]) / result["sma_50"]
    result["price_sma200_ratio"] = (result["close"] - result["sma_200"]) / result["sma_200"]

    # SMA-Crossover: 1 = SMA20 über SMA50 (bullish), -1 = bearish
    result["sma_20_50_cross"] = np.sign(result["sma_20"] - result["sma_50"])
    result["sma_50_200_cross"] = np.sign(result["sma_50"] - result["sma_200"])

    # --- Exponential Moving Averages ---
    result["ema_12"] = ta.ema(result["close"], length=12)
    result["ema_26"] = ta.ema(result["close"], length=26)

    # EMA Crossover: 1 = EMA12 über EMA26 (bullish Signal)
    result["ema_cross"] = np.sign(result["ema_12"] - result["ema_26"])

    # --- MACD (12, 26, 9) ---
    # macd_line = EMA12 - EMA26 (Momentum)
    # macd_signal = EMA9 der MACD-Linie
    # macd_hist = macd_line - macd_signal (Histogramm)
    macd = ta.macd(result["close"], fast=12, slow=26, signal=9)
    result["macd_line"] = macd["MACD_12_26_9"]
    result["macd_signal"] = macd["MACDs_12_26_9"]
    result["macd_hist"] = macd["MACDh_12_26_9"]

    logger.info("Trend-Features: SMA(20,50,200), EMA(12,26), MACD(12,26,9) ✓")
    return result


# ============================================================
# 3. Momentum-Features
# ============================================================


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Momentum-Indikatoren.

    RSI und Stochastic messen Überverkauft/Überkauft-Zustände.
    Williams %R ist ein weiterer Überkauft-Indikator.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Momentum-Spalten.
    """
    result = df.copy()

    # --- RSI (14) ---
    # >70 = überkauft (Verkaufssignal), <30 = überverkauft (Kaufsignal)
    result["rsi_14"] = ta.rsi(result["close"], length=14)

    # RSI zentriert um 50 (für lineares Modell besser)
    result["rsi_centered"] = result["rsi_14"] - 50

    # --- Stochastic Oscillator (14, 3) ---
    # %K = aktuelle Position innerhalb der letzten 14 Kerzen
    # %D = 3-Perioden-Durchschnitt von %K (Signal-Linie)
    stoch = ta.stoch(result["high"], result["low"], result["close"], k=14, d=3)
    result["stoch_k"] = stoch["STOCHk_14_3_3"]
    result["stoch_d"] = stoch["STOCHd_14_3_3"]
    result["stoch_cross"] = np.sign(result["stoch_k"] - result["stoch_d"])

    # --- Williams %R (14) ---
    # -100 bis 0: Werte < -80 = überverkauft, > -20 = überkauft
    result["williams_r"] = ta.willr(result["high"], result["low"], result["close"], length=14)

    # --- Rate of Change (10 Perioden) ---
    # Prozentuale Preisveränderung über 10 Kerzen
    result["roc_10"] = ta.roc(result["close"], length=10)

    logger.info("Momentum-Features: RSI(14), Stochastic(14,3), Williams %R(14), ROC(10) ✓")
    return result


# ============================================================
# 4. Volatilitäts-Features
# ============================================================


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Volatilitäts-Indikatoren.

    ATR misst die absolute Volatilität in Pips.
    Bollinger Bands zeigen relative Volatilität und Preis-Position.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Volatilitäts-Spalten.
    """
    result = df.copy()

    # --- ATR (14) – Average True Range ---
    # Misst die durchschnittliche Schwankungsbreite pro Kerze (in Pips)
    result["atr_14"] = ta.atr(result["high"], result["low"], result["close"], length=14)

    # ATR normalisiert (als % des Preises, für verschiedene Instrumente vergleichbar)
    result["atr_pct"] = result["atr_14"] / result["close"]

    # --- Bollinger Bands (20, 2 Standardabweichungen) ---
    bbands = ta.bbands(result["close"], length=20, std=2)
    # Spaltennamen dynamisch ermitteln (versions-unabhängig)
    bb_cols = {c[2]: c for c in bbands.columns}  # {'L': 'BBL_...', 'M': 'BBM_...', ...}
    result["bb_upper"] = bbands[bb_cols["U"]]
    result["bb_mid"] = bbands[bb_cols["M"]]
    result["bb_lower"] = bbands[bb_cols["L"]]

    # BB Width: Breite der Bänder (hoch = volatile Phase)
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]

    # BB %B: Wo liegt der Preis? 0 = am unteren Band, 1 = am oberen Band
    band_range = result["bb_upper"] - result["bb_lower"]
    result["bb_pct"] = (result["close"] - result["bb_lower"]) / band_range.replace(0, np.nan)

    # --- Historische Volatilität (Rolling 20 Perioden) ---
    # Annualisierte Volatilität der Log-Returns
    log_returns = np.log(result["close"] / result["close"].shift(1))
    result["hist_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252 * 24)

    logger.info("Volatilitäts-Features: ATR(14), Bollinger Bands(20,2), Hist.Vol(20) ✓")
    return result


# ============================================================
# 5. Volumen-Features
# ============================================================


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Volumen-basierte Indikatoren.

    Hinweis: MT5 liefert Tick-Volumen, kein echtes Forex-Volumen.
    Tick-Volumen korreliert jedoch gut mit der echten Handelsaktivität.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame mit Volumen-Spalten.
    """
    result = df.copy()

    # --- OBV: On-Balance Volume ---
    # Steigt wenn close > close(t-1), fällt sonst → Trendbestätigung durch Volumen
    result["obv"] = ta.obv(result["close"], result["volume"])

    # OBV normalisiert (Z-Score, Rolling 50) für stationären Wert
    obv_mean = result["obv"].rolling(50).mean()
    obv_std = result["obv"].rolling(50).std()
    result["obv_zscore"] = (result["obv"] - obv_mean) / obv_std.replace(0, np.nan)

    # --- Volumen Rate of Change (14 Perioden) ---
    result["volume_roc"] = ta.roc(result["volume"], length=14)

    # --- Volumen-Ratio: aktuelles Volumen vs. 20-Perioden-Durchschnitt ---
    # >1 = überdurchschnittliche Aktivität (oft bei Ausbrüchen)
    vol_sma_20 = result["volume"].rolling(20).mean()
    result["volume_ratio"] = result["volume"] / vol_sma_20.replace(0, np.nan)

    logger.info("Volumen-Features: OBV, OBV Z-Score, Volume ROC(14), Volume Ratio ✓")
    return result


# ============================================================
# 6. Kerzenmuster-Features
# ============================================================


def kerzenmuster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Features basierend auf der Kerzenstruktur.

    Alle Werte werden durch den ATR normalisiert, damit sie
    für verschiedene Volatilitätsphasen vergleichbar sind.

    Args:
        df: OHLCV DataFrame (muss atr_14 bereits enthalten!)

    Returns:
        DataFrame mit Kerzenstruktur-Spalten.
    """
    result = df.copy()
    atr = result["atr_14"]

    # --- Log-Returns ---
    result["return_1h"] = np.log(result["close"] / result["close"].shift(1))
    result["return_4h"] = np.log(result["close"] / result["close"].shift(4))
    result["return_24h"] = np.log(result["close"] / result["close"].shift(24))

    # --- Kerzenstruktur (normalisiert durch ATR) ---
    body_top = result[["close", "open"]].max(axis=1)
    body_bot = result[["close", "open"]].min(axis=1)

    result["candle_body"] = (body_top - body_bot) / atr.replace(0, np.nan)
    result["upper_wick"] = (result["high"] - body_top) / atr.replace(0, np.nan)
    result["lower_wick"] = (body_bot - result["low"]) / atr.replace(0, np.nan)

    # Kerzenrichtung: 1 = bullish (Kurs gestiegen), -1 = bearish
    result["candle_dir"] = np.sign(result["close"] - result["open"])

    # High-Low-Range (normalisiert durch Close-Preis)
    result["hl_range"] = (result["high"] - result["low"]) / result["close"]

    logger.info("Kerzenmuster-Features: Returns(1h,4h,24h), Körper, Wicks, HL-Range ✓")
    return result


# ============================================================
# 7. Multi-Timeframe-Features (H4 + D1)
# ============================================================


def multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Multi-Timeframe Trend-Features (H4 und D1).

    LOOK-AHEAD-BIAS PRÄVENTION:
    H4/D1-Kerzen werden mit .shift(1) verzögert, bevor sie auf H1
    forward-gefüllt werden. Eine H4-Kerze die um 08:00 schließt,
    wird erst ab der nächsten H1-Kerze (09:00) als Feature genutzt.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Multi-Timeframe-Spalten.
    """
    result = df.copy()
    close = df["close"]

    # --- H4 (4-Stunden) Trend ---
    close_h4 = close.resample("4h").last().dropna()
    sma_h4_20 = ta.sma(close_h4, length=20)
    sma_h4_50 = ta.sma(close_h4, length=50)

    # 1 = SMA20 > SMA50 (H4 bullish), -1 = bearish
    trend_h4 = np.sign(sma_h4_20 - sma_h4_50)
    # .shift(1): verhindert Look-Ahead (aktuelle H4-Kerze erst NACH dem Schließen nutzbar)
    trend_h4_delayed = trend_h4.shift(1)
    result["trend_h4"] = trend_h4_delayed.reindex(result.index, method="ffill")

    # H4 RSI als zusätzliches Momentum-Feature
    rsi_h4 = ta.rsi(close_h4, length=14)
    rsi_h4_delayed = rsi_h4.shift(1)
    result["rsi_h4"] = rsi_h4_delayed.reindex(result.index, method="ffill")

    # --- D1 (Täglich) Trend ---
    close_d1 = close.resample("1D").last().dropna()
    sma_d1_20 = ta.sma(close_d1, length=20)
    sma_d1_50 = ta.sma(close_d1, length=50)

    trend_d1 = np.sign(sma_d1_20 - sma_d1_50)
    trend_d1_delayed = trend_d1.shift(1)
    result["trend_d1"] = trend_d1_delayed.reindex(result.index, method="ffill")

    logger.info("Multi-Timeframe-Features: H4-Trend, H4-RSI, D1-Trend ✓")
    return result


# ============================================================
# 8. Zeitbasierte Features
# ============================================================


def zeitbasierte_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet zeitbasierte Features (Handelssitzungen, Wochentag).

    Forex-Märkte zeigen starke Intraday-Muster:
    - London-Session (08–17 UTC): Hohe Volatilität
    - New York-Session (13–22 UTC): Ebenfalls hohe Aktivität
    - London/NY-Overlap (13–17 UTC): Maximale Liquidität
    - Asien-Session (00–09 UTC): Geringe Volatilität

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)

    Returns:
        DataFrame mit Zeit-Spalten.
    """
    result = df.copy()

    # Stunde (0–23 UTC) und Wochentag (0=Montag, 4=Freitag)
    result["hour"] = result.index.hour
    result["day_of_week"] = result.index.dayofweek

    # Handelssitzungen (Binary: 1 = aktiv, 0 = inaktiv)
    h = result["hour"]
    result["session_london"] = ((h >= 8) & (h < 17)).astype(int)
    result["session_ny"] = ((h >= 13) & (h < 22)).astype(int)
    result["session_asia"] = ((h >= 0) & (h < 9)).astype(int)
    result["session_overlap"] = ((h >= 13) & (h < 17)).astype(int)  # London+NY Overlap

    logger.info("Zeitbasierte Features: Stunde, Wochentag, 4 Sitzungen ✓")
    return result


# ============================================================
# 9. NaN-Bereinigung
# ============================================================


def nan_bereinigung(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Zeilen mit NaN-Werten (entstehen durch Warm-Up-Perioden).

    SMA(200) braucht 200 Kerzen → erste 199 Zeilen haben NaN.
    Diese werden entfernt, damit das Modell saubere Daten erhält.

    Args:
        df: DataFrame mit möglichen NaN-Werten

    Returns:
        DataFrame ohne NaN-Werte.
    """
    vor = len(df)
    df_clean = df.dropna()
    entfernt = vor - len(df_clean)
    logger.info(
        f"NaN-Bereinigung: {entfernt} Zeilen entfernt (Warm-Up-Period), "
        f"{len(df_clean):,} Kerzen übrig"
    )
    return df_clean


# ============================================================
# 10. Hauptfunktion
# ============================================================


def main() -> None:
    """Hauptablauf: Daten laden → Features berechnen → validieren → speichern."""
    logger.info("=" * 60)
    logger.info("Feature Engineering – gestartet")
    logger.info("Gerät: Linux-Server")
    logger.info("=" * 60)

    # 1. Rohdaten laden
    eingabe = DATA_DIR / "EURUSD_H1.csv"
    df = daten_laden(eingabe)

    # 2. Features in korrekter Reihenfolge berechnen
    #    (volatility_features muss vor kerzenmuster_features kommen,
    #     da ATR für die Normalisierung benötigt wird!)
    df = trend_features(df)
    df = momentum_features(df)
    df = volatility_features(df)       # ← ATR wird hier erstellt
    df = volume_features(df)
    df = kerzenmuster_features(df)     # ← ATR wird hier verwendet
    df = multitimeframe_features(df)
    df = zeitbasierte_features(df)

    # 3. NaN-Zeilen entfernen
    df = nan_bereinigung(df)

    # 4. Feature-Korrelation grob prüfen (Warnungen bei >0.95)
    ohlcv_cols = ["open", "high", "low", "close", "volume", "spread"]
    feature_cols = [c for c in df.columns if c not in ohlcv_cols]
    korr = df[feature_cols].corr().abs()
    hoch_korreliert = [
        (c1, c2, korr.loc[c1, c2])
        for i, c1 in enumerate(feature_cols)
        for c2 in feature_cols[i + 1 :]
        if korr.loc[c1, c2] > 0.95
    ]
    if hoch_korreliert:
        logger.warning(f"{len(hoch_korreliert)} Feature-Paare mit Korrelation >0.95:")
        for c1, c2, val in hoch_korreliert[:5]:
            logger.warning(f"  {c1} ↔ {c2}: {val:.3f}")

    # 5. Ergebnis speichern
    ausgabe = DATA_DIR / "EURUSD_H1_features.csv"
    df.to_csv(ausgabe)
    groesse_mb = ausgabe.stat().st_size / (1024 * 1024)

    # 6. Zusammenfassung
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING – ABGESCHLOSSEN")
    print("=" * 60)
    print(f"\nKerzen:   {len(df):,}")
    print(f"Spalten:  {len(df.columns)} (inkl. OHLCV)")
    print(f"Features: {len(feature_cols)}")
    print(f"Zeitraum: {df.index[0]}  –  {df.index[-1]}")
    print(f"Datei:    {ausgabe} ({groesse_mb:.1f} MB)")
    print(f"\nAlle Feature-Spalten ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    print("=" * 60)


if __name__ == "__main__":
    main()
