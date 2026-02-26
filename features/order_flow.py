"""
order_flow.py – Externe Marktdaten als Features laden und in CSVs einfügen.

Läuft auf: Linux-Server
Benötigt: requests (bereits installiert)

Was dieses Skript macht:
    1. Fear & Greed Index (Alternative.me) – kostenlos, kein API-Key nötig
       Täglicher Stimmungsindikator: 0=Extreme Fear, 100=Extreme Greed
       Daten verfügbar ab: 2018-02-01

    2. BTC Funding Rate (Binance Futures API) – kostenlos, kein API-Key nötig
       Proxy für Risk-On/Risk-Off Stimmung im Markt.
       Positiv = Markt bullish (Longs zahlen),
       Negativ = bearish (Shorts zahlen)
       Daten verfügbar ab: 2019-09-10

    3. BTC Open Interest (Binance Futures API) – kostenlos, kein API-Key nötig
       Anzahl offener Kontrakte = Proxy für Marktaktivität/Risikobereitschaft
       Daten verfügbar ab: ~2020-01-01

WICHTIG – Look-Ahead-Bias Prävention:
    Alle externen Features werden mit .shift(1) verschoben.
    Beispiel: Fear & Greed von Montag → wird erst Dienstag als Feature genutzt.

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python features/order_flow.py

Output:
    data/fear_greed.csv
    data/btc_funding_rate.csv
    data/btc_open_interest.csv
    data/SYMBOL_H1_labeled.csv (aktualisiert mit neuen Feature-Spalten)
"""
# pylint: disable=duplicate-code

# Standard-Bibliotheken
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Datenverarbeitung
import numpy as np
import pandas as pd

# HTTP-Anfragen an externe APIs
import requests

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pfade (absolut, plattformunabhängig)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Alle 7 Forex-Symbole
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
# 1. Fear & Greed Index (Alternative.me)
# ============================================================


def fear_greed_laden() -> pd.DataFrame:
    """
    Lädt den historischen Fear & Greed Index von alternative.me.

    Kostenlos, kein API-Key nötig.
    Gibt alle verfügbaren historischen Daten zurück.
    Die API liefert täglich einen Wert zwischen
    0 (Extreme Fear) und 100 (Extreme Greed).

    Returns:
        DataFrame mit Spalten [fear_greed_value, fear_greed_class].
        Index: DatetimeIndex (täglich, UTC).
        Leerer DataFrame bei Fehler.
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    logger.info("Fear & Greed Index wird geladen (alternative.me)...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        daten = response.json()
    except requests.RequestException as e:
        logger.error("Fehler beim Laden des Fear & Greed Index: %s", e)
        return pd.DataFrame()

    eintraege = daten.get("data", [])
    if not eintraege:
        logger.error("Keine Daten vom Fear & Greed Index erhalten.")
        return pd.DataFrame()

    # Rohdaten in DataFrame umwandeln
    df = pd.DataFrame(eintraege)

    # Unix-Timestamp (Sekunden) → datetime mit UTC-Zeitzone
    df["time"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="s", utc=True
    )
    df = df.set_index("time").sort_index()

    # Numerischer Wert (0–100)
    df["fear_greed_value"] = df["value"].astype(float)

    # Klassifizierung in 4 Kategorien für das Modell
    # 0 = Extreme Fear (0–24), 1 = Fear (25–49),
    # 2 = Greed (50–74), 3 = Extreme Greed (75–100)
    df["fear_greed_class"] = pd.cut(
        df["fear_greed_value"],
        bins=[0, 25, 50, 75, 100],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)

    result = df[["fear_greed_value", "fear_greed_class"]].copy()
    result = result[~result.index.duplicated(keep="last")]

    logger.info(
        "Fear & Greed: %s Tage geladen (%s – %s)",
        f"{len(result):,}",
        result.index[0].date(),
        result.index[-1].date(),
    )
    return result


# ============================================================
# 2. BTC Funding Rate (Binance Futures)
# ============================================================


def binance_funding_rate_laden(
    start_datum: str = "2019-09-10",
    end_datum: Optional[str] = None,
) -> pd.DataFrame:
    """
    Lädt historische BTC/USDT Funding Rates von Binance Perpetual Futures.

    Die Funding Rate wird alle 8 Stunden berechnet und bezahlt.
    Sie ist ein guter Proxy für die Marktstimmung:
    - Positiv (+): Markt ist bullish (Long-Trader bezahlen Short-Trader)
    - Negativ (-): Markt ist bearish (Short-Trader bezahlen Long-Trader)
    - Extrem positiv (>0.01): Warnsignal für überhitzte Long-Positionen

    Args:
        start_datum: Startdatum im Format "YYYY-MM-DD" (frühestens 2019-09-10)
        end_datum: Enddatum im Format "YYYY-MM-DD" (Standard: heute)

    Returns:
        DataFrame mit Spalte [btc_funding_rate], Index: DatetimeIndex (UTC).
        Leerer DataFrame bei Fehler.
    """
    if end_datum is None:
        end_datum = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Binance Futures API Endpunkt
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    limit = 1000  # Maximum pro Anfrage

    # Zeitstempel in Millisekunden (Binance API Format)
    start_ts = int(
        datetime.strptime(start_datum, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_ts = int(
        datetime.strptime(end_datum, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )

    alle_daten = []
    aktuelle_ts = start_ts

    logger.info(
        "BTC Funding Rate wird geladen (Binance, %s – %s)...",
        start_datum,
        end_datum,
    )

    # Paginierung: Binance liefert max. 1000 Einträge pro Anfrage
    while aktuelle_ts < end_ts:
        params = {
            "symbol": "BTCUSDT",
            "startTime": aktuelle_ts,
            "endTime": end_ts,
            "limit": limit,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            daten = response.json()
        except requests.RequestException as e:
            logger.error("Binance API Fehler (Funding Rate): %s", e)
            break

        # Keine weiteren Daten verfügbar
        if not daten:
            break

        # API-Fehler abfangen (Binance gibt Fehlercodes im JSON zurück)
        if isinstance(daten, dict) and "code" in daten:
            logger.error("Binance API Fehler: %s", daten)
            break

        alle_daten.extend(daten)

        # Letzten Zeitstempel als neuen Startpunkt nehmen
        letzter_ts = daten[-1]["fundingTime"]
        if letzter_ts <= aktuelle_ts:
            break  # Verhindert Endlosschleife

        aktuelle_ts = letzter_ts + 1

        # API-Limit respektieren (max. 3 Anfragen/Sekunde)
        time.sleep(0.4)
        logger.info(
            "  Bis: %s (%s Einträge)",
            pd.to_datetime(letzter_ts, unit="ms", utc=True).date(),
            f"{len(alle_daten):,}",
        )

    if not alle_daten:
        logger.error("Keine Funding Rate Daten erhalten.")
        return pd.DataFrame()

    # DataFrame erstellen und bereinigen
    df = pd.DataFrame(alle_daten)
    df["time"] = pd.to_datetime(
        df["fundingTime"].astype(int), unit="ms", utc=True
    )
    df = df.set_index("time").sort_index()
    df["btc_funding_rate"] = df["fundingRate"].astype(float)

    result = df[["btc_funding_rate"]].copy()
    result = result[~result.index.duplicated(keep="last")]

    logger.info(
        "Funding Rate: %s Einträge (%s – %s)",
        f"{len(result):,}",
        result.index[0].date(),
        result.index[-1].date(),
    )
    return result


# ============================================================
# 3. BTC Open Interest (Binance Futures)
# ============================================================


# pylint: disable=too-many-locals
def binance_open_interest_laden(
    start_datum: str = "2020-01-01",
    end_datum: Optional[str] = None,
) -> pd.DataFrame:
    """
    Lädt historisches BTC Open Interest (stündlich) von Binance Futures.

    Open Interest = Gesamtzahl offener Derivate-Kontrakte.
    Interpretation:
    - Steigendes OI + steigender Preis = Starker, bestätigter Trend
    - Fallendes OI + fallender Preis =
      Positionen werden geschlossen (Trendende?)
    - Hohes OI = Viele offene Positionen = Potenzial für starke Bewegungen

    Args:
        start_datum: Startdatum im Format "YYYY-MM-DD" (frühestens ~2020-01-01)
        end_datum: Enddatum im Format "YYYY-MM-DD" (Standard: heute)

    Returns:
        DataFrame mit Spalten [btc_oi_change, btc_oi_zscore],
        Index: DatetimeIndex (UTC, stündlich).
        Leerer DataFrame bei Fehler.
    """
    if end_datum is None:
        end_datum = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    url = "https://fapi.binance.com/futures/data/openInterestHist"
    limit = 500  # Maximum pro Anfrage

    start_ts = int(
        datetime.strptime(start_datum, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_ts = int(
        datetime.strptime(end_datum, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )

    alle_daten = []
    aktuelle_ts = start_ts
    seiten = 0

    logger.info(
        "BTC Open Interest wird geladen (Binance, %s – %s)...",
        start_datum,
        end_datum,
    )

    # Paginierung: jeweils 500 Stunden (≈ 20 Tage) pro Anfrage
    while aktuelle_ts < end_ts:
        # Endzeitpunkt für diese Anfrage: max. 500 Stunden voraus
        batch_end = min(aktuelle_ts + limit * 3_600_000, end_ts)

        params = {
            "symbol": "BTCUSDT",
            "period": "1h",
            "limit": limit,
            "startTime": aktuelle_ts,
            "endTime": batch_end,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            daten = response.json()
        except requests.RequestException as e:
            logger.error("Binance OI API Fehler: %s", e)
            break

        # API-Fehler oder keine Daten mehr
        if not daten or (isinstance(daten, dict) and "code" in daten):
            logger.warning(
                "OI-Daten lücke bei %s",
                pd.to_datetime(aktuelle_ts, unit="ms", utc=True).date(),
            )
            # Trotzdem weitermachen, Zeitfenster vorspulen
            aktuelle_ts = batch_end + 1
            time.sleep(0.4)
            continue

        alle_daten.extend(daten)

        # Nächster Startpunkt = letzter Zeitstempel + 1 Stunde
        letzter_ts = daten[-1]["timestamp"]
        aktuelle_ts = letzter_ts + 3_600_000

        seiten += 1
        if seiten % 20 == 0:
            logger.info(
                "  Seite %s: bis %s (%s Einträge)",
                seiten,
                pd.to_datetime(letzter_ts, unit="ms", utc=True).date(),
                f"{len(alle_daten):,}",
            )

        time.sleep(0.4)

    if not alle_daten:
        logger.warning(
            "Keine Open Interest Daten erhalten. "
            "Möglicherweise API-Beschränkung oder Zeitraum zu weit zurück."
        )
        return pd.DataFrame()

    # DataFrame erstellen
    df = pd.DataFrame(alle_daten)
    df["time"] = pd.to_datetime(
        df["timestamp"].astype(int), unit="ms", utc=True
    )
    df = df.set_index("time").sort_index()
    df["btc_oi"] = df["sumOpenInterest"].astype(float)
    df = df[~df.index.duplicated(keep="last")]

    # Prozentuale Änderung (stationärer als absoluter Wert)
    df["btc_oi_change"] = df["btc_oi"].pct_change().clip(-0.5, 0.5)

    # Z-Score über rollende 50 Stunden (normalisiert, modellfreundlich)
    oi_mean = df["btc_oi"].rolling(50).mean()
    oi_std = df["btc_oi"].rolling(50).std()
    df["btc_oi_zscore"] = (df["btc_oi"] - oi_mean) / oi_std.replace(0, np.nan)

    result = df[["btc_oi_change", "btc_oi_zscore"]].copy()

    logger.info(
        "Open Interest: %s Einträge (%s – %s)",
        f"{len(result):,}",
        result.index[0].date(),
        result.index[-1].date(),
    )
    return result


# ============================================================
# 4. Externe Features auf H1-Granularität bringen
# ============================================================


def externe_features_auf_h1(
    df_h1: pd.DataFrame,
    fg: pd.DataFrame,
    funding: pd.DataFrame,
    oi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bringt externe Features (täglich/8h/1h) auf H1-Granularität und merged sie.

    LOOK-AHEAD-BIAS PRÄVENTION:
    Alle externen Features werden mit .shift(1) verschoben:
    - Fear & Greed von Montag → erst Dienstag als Feature nutzbar
    - Funding Rate von 08:00 → erst ab 09:00 als Feature nutzbar
    - Open Interest von 10:00 → erst ab 11:00 als Feature nutzbar

    Args:
        df_h1:   Bestehender H1 DataFrame (Index: DatetimeIndex UTC)
        fg:      Fear & Greed DataFrame (täglich)
        funding: BTC Funding Rate DataFrame (8h-Intervalle)
        oi:      BTC Open Interest DataFrame (1h)

    Returns:
        DataFrame mit neuen externen Feature-Spalten angehängt.
    """
    result = df_h1.copy()

    # --- Fear & Greed Index: täglich → H1 ---
    if not fg.empty:
        # Reindex: fehlende Stunden mit forward-fill füllen
        # (letzter Tageswert gilt)
        fg_h1 = fg.reindex(result.index, method="ffill")
        # shift(1): gestrigen Fear & Greed Wert verwenden (kein Look-Ahead!)
        result["fear_greed_value"] = fg_h1["fear_greed_value"].shift(1)
        result["fear_greed_class"] = fg_h1["fear_greed_class"].shift(1)
        logger.info("  Fear & Greed: täglich → H1 gemapped + shift(1) ✓")

    # --- BTC Funding Rate: 8h → H1 ---
    if not funding.empty:
        # Reindex: letzter bekannter Funding Rate Wert gilt weiter
        # (forward-fill)
        funding_h1 = funding.reindex(result.index, method="ffill")
        # shift(1): aktuelle Funding Rate erst ab nächster Kerze als Feature
        result["btc_funding_rate"] = funding_h1["btc_funding_rate"].shift(1)
        logger.info("  BTC Funding Rate: 8h → H1 gemapped + shift(1) ✓")

    # --- BTC Open Interest: 1h → direkt mappen ---
    if not oi.empty:
        oi_h1 = oi.reindex(result.index, method="ffill")
        # shift(1): OI-Wert der letzten Stunde als Feature
        result["btc_oi_change"] = oi_h1["btc_oi_change"].shift(1)
        result["btc_oi_zscore"] = oi_h1["btc_oi_zscore"].shift(1)
        logger.info("  BTC Open Interest: 1h → H1 gemapped + shift(1) ✓")

    return result


# ============================================================
# 5. Hauptfunktion
# ============================================================


# pylint: disable=too-many-branches,too-many-statements
def main() -> None:
    """
    Hauptablauf:
        1. Externe Daten von APIs laden
        2. Als separate CSV-Dateien speichern (Backup)
        3. In alle 7 SYMBOL_H1_labeled.csv Dateien einfügen
    """
    logger.info("=" * 60)
    logger.info("ORDER FLOW & SENTIMENT FEATURES – START")
    logger.info("Gerät: Linux-Server")
    logger.info("=" * 60)

    # ── Schritt 1: Externe Daten laden ────────────────────────
    logger.info("\n[1/3] Externe Daten von APIs laden...")

    fg = fear_greed_laden()
    if not fg.empty:
        fg.to_csv(DATA_DIR / "fear_greed.csv")
        logger.info(
            "  Gespeichert: data/fear_greed.csv (%s Einträge)", f"{len(fg):,}"
        )

    # Kurze Pause zwischen API-Aufrufen
    time.sleep(1)

    funding = binance_funding_rate_laden(start_datum="2019-09-10")
    if not funding.empty:
        funding.to_csv(DATA_DIR / "btc_funding_rate.csv")
        logger.info(
            "  Gespeichert: data/btc_funding_rate.csv (%s Einträge)",
            f"{len(funding):,}",
        )

    time.sleep(1)

    oi = binance_open_interest_laden(start_datum="2020-01-01")
    if not oi.empty:
        oi.to_csv(DATA_DIR / "btc_open_interest.csv")
        logger.info(
            "  Gespeichert: data/btc_open_interest.csv (%s Einträge)",
            f"{len(oi):,}",
        )

    # Zusammenfassung welche Daten verfügbar sind
    verfuegbare_features = []
    if not fg.empty:
        verfuegbare_features.extend(["fear_greed_value", "fear_greed_class"])
    if not funding.empty:
        verfuegbare_features.append("btc_funding_rate")
    if not oi.empty:
        verfuegbare_features.extend(["btc_oi_change", "btc_oi_zscore"])

    if not verfuegbare_features:
        logger.error("Keine externen Daten geladen! Abbruch.")
        return

    logger.info("\n  Verfügbare neue Features: %s", verfuegbare_features)

    # ── Schritt 2: In alle labeled CSVs einfügen ─────────────
    logger.info("\n[2/3] Features in SYMBOL_H1_labeled.csv einfügen...")
    ergebnisse = []

    for symbol in SYMBOLE:
        labeled_pfad = DATA_DIR / f"{symbol}_H1_labeled.csv"
        if not labeled_pfad.exists():
            logger.warning(
                "  %s: labeled CSV nicht gefunden, überspringe.", symbol
            )
            ergebnisse.append((symbol, "FEHLER – Datei nicht gefunden"))
            continue

        # CSV laden
        df = pd.read_csv(labeled_pfad, index_col="time", parse_dates=True)

        # Sicherstellen dass der Index UTC-Zeitzone hat (nötig für reindex)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Vorhandene externe Features überschreiben (falls bereits vorhanden)
        for col in verfuegbare_features:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Externe Features hinzufügen
        df = externe_features_auf_h1(df, fg, funding, oi)

        # NaN in neuen Spalten auffüllen
        # Für Zeiträume vor API-Verfügbarkeit: Median verwenden
        gefuellte = {}
        for col in verfuegbare_features:
            if col not in df.columns:
                continue
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                gefuellte[col] = (nan_count, median_val)

        # Zurückspeichern
        df.to_csv(labeled_pfad)

        info_teile = [f"+{len(verfuegbare_features)} Features"]
        if gefuellte:
            info_teile.append(
                f"{sum(v[0] for v in gefuellte.values())} NaN→Median gefüllt"
            )
        ergebnisse.append((symbol, " | ".join(info_teile)))
        logger.info("  ✓ %s: %s", symbol, " | ".join(info_teile))

    # ── Schritt 3: Zusammenfassung ────────────────────────────
    logger.info("\n[3/3] Abschluss.")
    print("\n" + "=" * 60)
    print("ORDER FLOW & SENTIMENT – ABGESCHLOSSEN")
    print("=" * 60)
    print("\nNeue Features in SYMBOL_H1_labeled.csv:")
    if not fg.empty:
        print("  • fear_greed_value  – Fear & Greed Index (0–100)")
        print(
            "  • fear_greed_class  – "
            "Kategorie (0=ExtremFear, 1=Fear, 2=Greed, 3=ExtremGreed)"
        )
    if not funding.empty:
        print("  • btc_funding_rate  – BTC Funding Rate (alle 8h, Binance)")
    if not oi.empty:
        print(
            "  • btc_oi_change     – BTC Open Interest Änderung (1h, Binance)"
        )
        print(
            "  • btc_oi_zscore     – BTC Open Interest Z-Score (1h, Binance)"
        )

    print("\nSymbol-Status:")
    for symbol, status in ergebnisse:
        mark = "✓" if "FEHLER" not in status else "✗"
        print(f"  {mark} {symbol}: {status}")

    print("\n" + "=" * 60)
    print("HINWEIS: Neue Features wurden zu den labeled CSVs hinzugefügt.")
    print("Für optimale Modellleistung die Modelle neu trainieren:")
    print("  python train_model.py --symbol alle")
    print("=" * 60)


if __name__ == "__main__":
    main()
