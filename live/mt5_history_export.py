"""
mt5_history_export.py – MT5 Trade-Historie als strukturierte CSV exportieren

Liest die NATIVE MT5-Trade-Datenbank über die Python-API und exportiert:
  - Alle Deals (Entries + Exits) mit exakten Fill-Preisen
  - Kommissionen, Swap, Netto-PnL pro Trade
  - Magic Number, Kommentare, Zeitstempel

Datenquelle: MT5 Terminal-Datenbank (die genaueste Quelle!)
→ Enthält alle Daten die im Tab "Historie" sichtbar sind.

Verwendung (Windows Laptop, venv aktiviert):
    python mt5_history_export.py
    python mt5_history_export.py --magic 20260101 --tage 30 --symbol USDCAD
    python mt5_history_export.py --output reports/mt5_trades_export.csv

Die exportierte CSV kann dann auf dem Linux-Server analysiert werden.

Läuft auf: Windows 11 Laptop (MetaTrader5-Bibliothek NUR auf Windows!)
"""

import argparse
import importlib.util
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Projekt-Module robust laden (funktioniert als Skript und ohne Pylance-Importwarnung)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_config_path = Path(__file__).resolve().with_name("config.py")
_config_spec = importlib.util.spec_from_file_location("config", _config_path)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Konnte config.py nicht laden: {_config_path}")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def mt5_deals_exportieren(
    magic_number: int = config.MAGIC_NUMBER,
    tage: int = 90,
    symbol: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Exportiert alle Deals aus der MT5-Trade-Historie als DataFrame.

    Deals sind die tatsächlich ausgeführten Transaktionen – jeder Entry
    und Exit hat einen eigenen Deal mit Fill-Preis, Kommission und Swap.

    Args:
        magic_number: Nur Deals mit dieser Magic Number exportieren.
        tage: Zeitraum in Tagen rückwärts ab jetzt.
        symbol: Optional – nur Deals für dieses Symbol.

    Returns:
        DataFrame mit allen Deal-Feldern oder None bei Fehler.
    """
    if not config.MT5_VERFUEGBAR:
        logger.error("MetaTrader5 nicht installiert – nur auf Windows Laptop ausfuehrbar!")
        return None

    mt5_api = config.mt5_api()

    # Zeitraum berechnen
    bis = datetime.now(timezone.utc)
    von = bis - timedelta(days=tage)

    # Deals abrufen
    if symbol:
        deals = mt5_api.history_deals_get(von, bis, group=f"*{symbol}*")
    else:
        deals = mt5_api.history_deals_get(von, bis)

    if deals is None or len(deals) == 0:
        logger.info("Keine Deals im Zeitraum %s bis %s gefunden.", von.date(), bis.date())
        return None

    # In DataFrame umwandeln
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

    # Magic Number filtern
    if magic_number > 0:
        df = df[df["magic"] == magic_number]
        if df.empty:
            logger.info("Keine Deals mit Magic Number %d gefunden.", magic_number)
            return None

    # Zeitstempel in lesbares Format umwandeln
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)

    # Typ-Mapping für Lesbarkeit
    deal_types = {0: "BUY", 1: "SELL", 2: "BALANCE", 3: "CREDIT",
                  4: "CHARGE", 5: "CORRECTION", 6: "BONUS"}
    entry_types = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}

    df["type_str"] = df["type"].map(deal_types).fillna("OTHER")
    df["entry_str"] = df["entry"].map(entry_types).fillna("OTHER")

    # Netto-PnL berechnen (Profit + Kommission + Swap)
    df["pnl_net"] = df["profit"] + df["commission"] + df["swap"]

    # Sortieren nach Zeit
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(
        "Exportiert: %d Deals | Zeitraum: %s bis %s | Magic: %d | "
        "Symbole: %s",
        len(df), von.date(), bis.date(), magic_number,
        ", ".join(sorted(df["symbol"].unique())) if len(df) > 0 else "-",
    )

    return df


def trades_zusammenfuehren(deals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verknüpft Entry- und Exit-Deals zu vollständigen Roundtrip-Trades.

    Jede Zeile im Ergebnis ist ein kompletter Trade mit:
      - Entry-Zeit, Entry-Preis
      - Exit-Zeit, Exit-Preis
      - Dauer, PnL, Kommission, Swap
      - Richtung (Long/Short)

    Args:
        deals_df: DataFrame aus mt5_deals_exportieren().

    Returns:
        DataFrame mit einem Trade pro Zeile (Roundtrips).
    """
    if deals_df is None or deals_df.empty:
        return pd.DataFrame()

    # Nur Kauf/Verkauf-Deals (keine Balance, Credit, etc.)
    trades = deals_df[deals_df["type"].isin([0, 1])].copy()

    # Entry-Deals (IN)
    entries = trades[trades["entry"] == 0].copy()
    entries = entries.rename(columns={
        "time": "entry_time",
        "price": "entry_price",
        "volume": "entry_lot",
        "commission": "entry_commission",
        "ticket": "entry_deal",
    })

    # Exit-Deals (OUT)
    exits = trades[trades["entry"].isin([1, 2])].copy()
    exits = exits.rename(columns={
        "time": "exit_time",
        "price": "exit_price",
        "volume": "exit_lot",
        "profit": "trade_profit",
        "commission": "exit_commission",
        "swap": "trade_swap",
        "ticket": "exit_deal",
    })

    # Über position_id zusammenführen (Entry und Exit desselben Trades)
    merged = pd.merge(
        entries[["entry_time", "entry_price", "entry_lot", "entry_commission",
                 "entry_deal", "position_id", "symbol", "type", "magic", "comment"]],
        exits[["exit_time", "exit_price", "exit_lot", "exit_commission",
               "trade_profit", "trade_swap", "exit_deal", "position_id"]],
        on="position_id",
        how="left",
    )

    # Richtung bestimmen (type=0=BUY → Long, type=1=SELL → Short)
    merged["richtung"] = merged["type"].map({0: "Long", 1: "Short"})

    # Gesamt-Kommission und Netto-PnL
    merged["total_commission"] = merged["entry_commission"].fillna(0) + merged["exit_commission"].fillna(0)
    merged["pnl_net"] = merged["trade_profit"].fillna(0) + merged["total_commission"] + merged["trade_swap"].fillna(0)

    # Dauer berechnen
    merged["dauer_min"] = (
        (merged["exit_time"] - merged["entry_time"]).dt.total_seconds() / 60.0
    ).round(1)

    # Status: offen oder geschlossen
    merged["status"] = merged["exit_time"].apply(lambda x: "geschlossen" if pd.notna(x) else "offen")

    # Spalten auswählen und sortieren
    result_cols = [
        "symbol", "richtung", "entry_time", "entry_price", "exit_time", "exit_price",
        "entry_lot", "trade_profit", "total_commission", "trade_swap", "pnl_net",
        "dauer_min", "status", "magic", "position_id", "entry_deal", "exit_deal", "comment",
    ]
    existing_cols = [c for c in result_cols if c in merged.columns]
    result = merged[existing_cols].sort_values("entry_time").reset_index(drop=True)

    # Zusammenfassung loggen
    geschlossene = result[result["status"] == "geschlossen"]
    if not geschlossene.empty:
        total_pnl = geschlossene["pnl_net"].sum()
        win_rate = (geschlossene["pnl_net"] > 0).mean() * 100
        avg_dauer = geschlossene["dauer_min"].mean()
        logger.info(
            "Roundtrips: %d trades (%d geschlossen, %d offen) | "
            "PnL: %.2f | Win-Rate: %.1f%% | Ø Dauer: %.0f Min",
            len(result), len(geschlossene), len(result) - len(geschlossene),
            total_pnl, win_rate, avg_dauer,
        )

    return result


def main() -> None:
    """Hauptprogramm: MT5-Historie exportieren."""
    parser = argparse.ArgumentParser(
        description="MT5 Trade-Historie exportieren (für Auswertung)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--magic", type=int, default=config.MAGIC_NUMBER,
        help=f"Magic Number filtern (Standard: {config.MAGIC_NUMBER}). 0 = alle.",
    )
    parser.add_argument(
        "--tage", type=int, default=90,
        help="Zeitraum in Tagen rückwärts (Standard: 90).",
    )
    parser.add_argument(
        "--symbol", type=str, default="",
        help="Optional: nur für ein Symbol exportieren (z.B. USDCAD).",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Ausgabe-CSV-Pfad (Standard: logs/mt5_trade_history.csv).",
    )
    parser.add_argument(
        "--roundtrips", type=int, default=1, choices=[0, 1],
        help="1 = zusätzlich Roundtrip-CSV erstellen (Standard: 1).",
    )
    args = parser.parse_args()

    # MT5 verbinden (muss bereits geöffnet und eingeloggt sein)
    if not config.MT5_VERFUEGBAR:
        print("MetaTrader5 nicht installiert! Nur auf Windows Laptop ausführbar.")
        return

    mt5_api = config.mt5_api()
    if not mt5_api.initialize():
        print(f"MT5-Verbindung fehlgeschlagen: {mt5_api.last_error()}")
        print("Sicherstellen dass MT5 Terminal geöffnet und eingeloggt ist!")
        return

    konto = mt5_api.account_info()
    if konto:
        logger.info("MT5 verbunden | Konto: %d | Saldo: %.2f %s",
                    konto.login, konto.balance, konto.currency)

    # Deals exportieren
    symbol_filter = args.symbol.upper() if args.symbol else None
    deals_df = mt5_deals_exportieren(
        magic_number=args.magic,
        tage=args.tage,
        symbol=symbol_filter,
    )

    if deals_df is None or deals_df.empty:
        print("Keine Deals gefunden. Prüfe Magic Number und Zeitraum.")
        mt5_api.shutdown()
        return

    # Ausgabepfad bestimmen
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.LOG_DIR / "mt5_trade_history.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deals-CSV speichern
    deals_df.to_csv(output_path, index=False)
    logger.info("Deals exportiert → %s (%d Zeilen)", output_path, len(deals_df))
    print(f"\nDeals-CSV: {output_path} ({len(deals_df)} Zeilen)")

    # Roundtrips erstellen
    if args.roundtrips:
        roundtrips_df = trades_zusammenfuehren(deals_df)
        if not roundtrips_df.empty:
            rt_path = output_path.parent / output_path.name.replace(".csv", "_roundtrips.csv")
            roundtrips_df.to_csv(rt_path, index=False)
            print(f"Roundtrips-CSV: {rt_path} ({len(roundtrips_df)} Trades)")

            # Kurze Zusammenfassung auf dem Terminal
            geschlossene = roundtrips_df[roundtrips_df["status"] == "geschlossen"]
            if not geschlossene.empty:
                print("\n--- Zusammenfassung ---")
                print(f"Geschlossene Trades: {len(geschlossene)}")
                print(f"Netto-PnL:           {geschlossene['pnl_net'].sum():.2f}")
                print(f"Win-Rate:            {(geschlossene['pnl_net'] > 0).mean() * 100:.1f}%")
                print(f"Avg PnL/Trade:       {geschlossene['pnl_net'].mean():.2f}")
                print(f"Best Trade:          {geschlossene['pnl_net'].max():.2f}")
                print(f"Worst Trade:         {geschlossene['pnl_net'].min():.2f}")
                if "dauer_min" in geschlossene.columns:
                    print(f"Avg Dauer:           {geschlossene['dauer_min'].mean():.0f} Min")

    mt5_api.shutdown()
    logger.info("MT5 getrennt. Export abgeschlossen.")


if __name__ == "__main__":
    main()
