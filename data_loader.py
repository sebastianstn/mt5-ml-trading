"""
data_loader.py – MT5 historische Daten laden und als CSV speichern.

WICHTIG: Dieses Skript läuft NUR auf dem Windows 11 Laptop!
         MetaTrader5-Bibliothek funktioniert nicht auf Linux.

Verwendung:
    python data_loader.py

Ergebnis:
    data/EURUSD_H1.csv  (relativ zum Skript-Verzeichnis)
"""

# Standard-Bibliotheken
import logging
import sys
from pathlib import Path
from typing import Optional

# Datenverarbeitung
import pandas as pd

# MT5-Verbindung (NUR auf Windows!)
try:
    import MetaTrader5 as mt5
except ImportError:
    print("FEHLER: MetaTrader5-Bibliothek nicht gefunden.")
    print("       Dieses Skript läuft nur auf Windows!")
    print("       Installieren: pip install MetaTrader5")
    sys.exit(1)

# Umgebungsvariablen aus .env laden
import os
from dotenv import load_dotenv

load_dotenv()

# Logging konfigurieren (Ausgabe ins Terminal + Datei)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_loader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Ausgabepfad: data/-Ordner relativ zum Skript
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 1. MT5-Verbindung
# ============================================================


def mt5_verbinden() -> bool:
    """
    Verbindet sich mit dem laufenden MetaTrader 5 Terminal.

    Die Zugangsdaten werden aus der .env-Datei gelesen.

    Returns:
        True wenn Verbindung erfolgreich, False bei Fehler.
    """
    # MT5-Bibliothek initialisieren
    if not mt5.initialize():
        logger.error(f"MT5-Initialisierung fehlgeschlagen: {mt5.last_error()}")
        return False

    # Zugangsdaten aus .env lesen
    login = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")

    if login == 0 or not password or not server:
        logger.error(
            "MT5-Zugangsdaten fehlen in .env! "
            "Bitte MT5_LOGIN, MT5_PASSWORD und MT5_SERVER eintragen."
        )
        mt5.shutdown()
        return False

    # Mit Konto einloggen
    if not mt5.login(login, password=password, server=server):
        logger.error(f"MT5-Login fehlgeschlagen: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Erfolgreiche Verbindung bestätigen
    konto = mt5.account_info()
    logger.info(
        f"Verbunden: Konto {konto.login} | "
        f"Balance: {konto.balance:.2f} {konto.currency} | "
        f"Server: {konto.server}"
    )
    return True


# ============================================================
# 2. Daten laden
# ============================================================


def daten_laden(
    symbol: str = "EURUSD",
    timeframe: int = mt5.TIMEFRAME_H1,
    anzahl_kerzen: int = 50000,
) -> Optional[pd.DataFrame]:
    """
    Lädt historische OHLCV-Daten aus MetaTrader 5.

    Args:
        symbol: Handelssymbol (Standard: "EURUSD")
        timeframe: MT5-Timeframe-Konstante (Standard: TIMEFRAME_H1)
        anzahl_kerzen: Maximale Anzahl Kerzen (Standard: 50000 ≈ 5+ Jahre auf H1)

    Returns:
        DataFrame mit Spalten [open, high, low, close, volume, spread]
        und DatetimeIndex in UTC. None bei Fehler.
    """
    # Symbol in MT5 prüfen
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol '{symbol}' nicht in MT5 gefunden!")
        return None

    # Symbol im Market Watch aktivieren (falls noch nicht sichtbar)
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Symbol '{symbol}' konnte nicht aktiviert werden!")
            return None

    # Rohdaten abrufen (von Pos 0 = aktuellste Kerze rückwärts)
    logger.info(f"Lade {anzahl_kerzen:,} Kerzen für {symbol} H1 ...")
    rohdaten = mt5.copy_rates_from_pos(symbol, timeframe, 0, anzahl_kerzen)

    if rohdaten is None or len(rohdaten) == 0:
        logger.error(f"Keine Daten erhalten: {mt5.last_error()}")
        return None

    # NumPy-Structured-Array → pandas DataFrame
    df = pd.DataFrame(rohdaten)

    # Unix-Timestamp → UTC-Datetime (wichtig für korrekte Zeitzonenverwaltung)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.index.name = "time"

    # Nur benötigte Spalten behalten, tick_volume → volume umbenennen
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume", "spread"]]

    # Chronologisch sortieren (älteste Kerze zuerst)
    df.sort_index(inplace=True)

    logger.info(
        f"Daten geladen: {len(df):,} Kerzen | "
        f"Von: {df.index[0]} | Bis: {df.index[-1]}"
    )
    return df


# ============================================================
# 3. Datenqualität prüfen
# ============================================================


def daten_validieren(df: pd.DataFrame, symbol: str) -> bool:
    """
    Prüft die Qualität der geladenen OHLCV-Daten.

    Warnt bei NaN-Werten, falscher OHLC-Logik oder zu wenig Daten.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex
        symbol: Symbol-Name (nur für Logging)

    Returns:
        True wenn alle Prüfungen bestanden, False wenn Probleme gefunden.
    """
    probleme_gefunden = False

    # Prüfung 1: NaN-Werte
    nan_count = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"[{symbol}] {nan_count} NaN-Werte gefunden!")
        probleme_gefunden = True
    else:
        logger.info(f"[{symbol}] NaN-Check: OK (keine NaN-Werte)")

    # Prüfung 2: Mindest-Datenmenge (3 Jahre H1 ≈ 26.000 Kerzen)
    min_kerzen = 26000
    if len(df) < min_kerzen:
        logger.warning(
            f"[{symbol}] Nur {len(df):,} Kerzen – erwartet: min. {min_kerzen:,}"
        )
        probleme_gefunden = True
    else:
        logger.info(f"[{symbol}] Datenmenge: OK ({len(df):,} Kerzen)")

    # Prüfung 3: OHLC-Logik (High muss >= Low, Open, Close sein)
    ungueltig = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()
    if ungueltig > 0:
        logger.warning(f"[{symbol}] {ungueltig} Kerzen mit ungültiger OHLC-Logik!")
        probleme_gefunden = True
    else:
        logger.info(f"[{symbol}] OHLC-Logik: OK")

    # Prüfung 4: Duplizierte Zeitstempel
    duplikate = df.index.duplicated().sum()
    if duplikate > 0:
        logger.warning(f"[{symbol}] {duplikate} duplizierte Zeitstempel!")
        probleme_gefunden = True
    else:
        logger.info(f"[{symbol}] Zeitstempel: OK (keine Duplikate)")

    return not probleme_gefunden


# ============================================================
# 4. Als CSV speichern
# ============================================================


def daten_speichern(
    df: pd.DataFrame,
    symbol: str,
    timeframe_name: str = "H1",
) -> Path:
    """
    Speichert den OHLCV DataFrame als CSV-Datei.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex (UTC)
        symbol: Handelssymbol (für Dateiname)
        timeframe_name: Timeframe als String (für Dateiname)

    Returns:
        Absoluter Pfad zur gespeicherten CSV-Datei.
    """
    # data/-Ordner anlegen falls nicht vorhanden
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dateiname: z.B. EURUSD_H1.csv
    dateiname = f"{symbol}_{timeframe_name}.csv"
    dateipfad = DATA_DIR / dateiname

    # CSV speichern (Index = Zeitstempel wird mitgespeichert)
    df.to_csv(dateipfad)

    groesse_kb = dateipfad.stat().st_size / 1024
    logger.info(f"CSV gespeichert: {dateipfad} ({groesse_kb:.0f} KB)")
    return dateipfad


# ============================================================
# 5. Hauptfunktion
# ============================================================


def main() -> None:
    """Hauptablauf: MT5 verbinden → Daten laden → prüfen → speichern."""
    logger.info("=" * 60)
    logger.info("MT5 Data Loader – gestartet")
    logger.info("Gerät: Windows 11 Laptop (MT5 muss geöffnet sein!)")
    logger.info("=" * 60)

    # Schritt 1: MT5 verbinden
    if not mt5_verbinden():
        logger.error("Verbindung fehlgeschlagen – Abbruch.")
        sys.exit(1)

    try:
        # Schritt 2: Daten laden (50.000 H1-Kerzen ≈ 5+ Jahre)
        df = daten_laden(
            symbol="EURUSD",
            timeframe=mt5.TIMEFRAME_H1,
            anzahl_kerzen=50000,
        )

        if df is None:
            logger.error("Datenladen fehlgeschlagen – Abbruch.")
            sys.exit(1)

        # Schritt 3: Datenqualität prüfen
        qualitaet_ok = daten_validieren(df, "EURUSD")
        if not qualitaet_ok:
            logger.warning("Datenqualitäts-Probleme gefunden – trotzdem weiter.")

        # Schritt 4: Als CSV speichern
        dateipfad = daten_speichern(df, "EURUSD", "H1")

        # Schritt 5: Kurzvorschau ausgeben
        print("\n" + "=" * 60)
        print("ERFOLG – Daten geladen und gespeichert!")
        print("=" * 60)
        print(f"\nZeitraum: {df.index[0]}  bis  {df.index[-1]}")
        print(f"Kerzen:   {len(df):,}")
        print(f"Datei:    {dateipfad}")
        print("\nErste 3 Zeilen:")
        print(df.head(3).to_string())
        print("\nLetzte 3 Zeilen:")
        print(df.tail(3).to_string())
        print("\n" + "=" * 60)
        print("Nächster Schritt: CSV auf den Linux-Server übertragen.")
        print("Befehl (in PowerShell ausführen):")
        print(
            f"  scp {dateipfad} "
            "stnsebi@<linux-server-ip>:/mnt/1T-Data/XGBoost-LightGBM/data/"
        )
        print("=" * 60)

    finally:
        # MT5-Verbindung immer sauber schließen
        mt5.shutdown()
        logger.info("MT5-Verbindung getrennt.")


if __name__ == "__main__":
    main()
