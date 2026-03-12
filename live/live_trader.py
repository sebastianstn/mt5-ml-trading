"""
live_trader.py – Automatisches Live-Trading mit MetaTrader 5 und LightGBM

Läuft auf: Windows 11 Laptop (MetaTrader5-Bibliothek NUR auf Windows!)

PAPER-TRADING-MODUS (Standard):
    PAPER_TRADING = True  → Signale werden NUR geloggt, KEIN echtes Geld!
    PAPER_TRADING = False → Echte Orders – erst nach 2 Wochen Testlauf!

Ablauf pro H1-Kerze:
    1. Neue H1-Daten von MT5 abrufen (letzten 500 Barren)
    2. Alle 45+ Features berechnen (identisch mit feature_engineering.py)
    3. Externe Features holen (Fear & Greed Index, BTC Funding Rate)
    4. Marktregime erkennen (identisch mit regime_detection.py)
    5. LightGBM-Vorhersage + Wahrscheinlichkeits-Filter (Schwelle)
       - Optional: Shadow-Mode mit Two-Stage (HTF H1 + LTF M5) für USDCAD/USDJPY
    6. Regime-Filter anwenden (z.B. nur im Aufwärtstrend handeln)
    7. Order senden (Paper: nur loggen / Live: echte MT5-Order)

Shadow-Mode (Two-Stage):
    --two_stage_enable 1 aktiviert das Two-Stage-System (USDCAD + USDJPY):
        - HTF-Bias-Modell (H1): Bestimmt Marktrichtung (Short/Neutral/Long)
        - LTF-Entry-Modell (M5): Generiert Entry-Signal basierend auf HTF-Bias
        - Beide Signale (Single-Stage vs. Two-Stage) werden geloggt für Vergleich
        - Hard Fallback zu Single-Stage bei jedem Fehler

Modell-Übertragung vom Linux-Server auf den Windows Laptop:
    scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_usdcad_v4.pkl USER@LAPTOP:./models/
    scp /mnt/1Tb-Data/XGBoost-LightGBM/models/lgbm_usdjpy_v4.pkl USER@LAPTOP:./models/

Verwendung (Windows, venv aktiviert):
    python live_trader.py --symbol USDCAD --schwelle 0.48 --regime_filter 0,1,2,3 --atr_sl 1

    # Two-Stage Shadow-Mode:
    python live_trader.py --symbol USDJPY --schwelle 0.48 --atr_sl 1 \\
        --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4

    python live_trader.py --help

Voraussetzungen:
    pip install MetaTrader5 pandas numpy pandas_ta joblib requests python-dotenv
"""

# pylint: disable=logging-fstring-interpolation

# Standard-Bibliotheken
import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

# Datenverarbeitung
import pandas as pd

# Modell laden
import joblib

# ---- Projekt-Module (alle ausgelagerten Funktionen) ----
import config
from config import (
    BASE_DIR,
    FEATURE_SPALTEN,
    HEARTBEAT_LOG_DEFAULT,
    KILL_SWITCH_DD_DEFAULT,
    LOT,
    MODEL_DIR,
    MT5_VERFUEGBAR,
    REGIME_NAMEN,
    STANDARD_SCHWELLE,
    STANDARD_STARTUP_OBSERVATION_BARS,
    STANDARD_TWO_STAGE_COOLDOWN_BARS,
    SYMBOLE,
    AKTIVE_SYMBOLE,
    TIMEFRAME_CONFIG,
    mt5_api as _mt5_api,
)
from external_api import externe_features_einfuegen, externe_features_holen
from feature_builder import features_berechnen
from mt5_connector import (
    alle_positionen_schliessen,
    mt5_daten_holen,
    mt5_letzte_kerze_uhrzeit,
    mt5_offene_position,
    mt5_verbinden,
    order_senden,
)
from paper_trading import _letzte_geschlossene_kerze, paper_trade_pruefen_und_loggen
from risk_manager import kill_switch_pruefen, offenen_trade_pruefen
from signal_engine import _modell_feature_namen, shadow_signal_generieren
from trade_logger import trade_loggen


# ============================================================
# Logging-Konfiguration
# ============================================================


def _resolve_log_dir(
    base_dir: Path,
    cli_log_dir: str = "",
    cli_log_subdir: str = "",
) -> Path:
    """
    Bestimmt den effektiven Log-Ordner optional über Umgebungsvariablen.

    Priorität: CLI > Umgebungsvariable MT5_TRADING_LOG_DIR > Unterordner > Standard.

    Args:
        base_dir: Projekt-Basisverzeichnis.
        cli_log_dir: Optionaler absoluter Log-Pfad aus der CLI.
        cli_log_subdir: Optionaler Unterordner unter ``base_dir / "logs"`` aus der CLI.

    Returns:
        Aufgelöster Log-Pfad als Path.
    """
    # CLI-Override hat höchste Priorität
    cli_log_dir = cli_log_dir.strip()
    if cli_log_dir:
        return Path(cli_log_dir).expanduser()

    cli_log_subdir = cli_log_subdir.strip()
    if cli_log_subdir:
        normalized_cli_subdir = cli_log_subdir.replace("\\", "/").strip("/ ")
        if normalized_cli_subdir:
            return base_dir / "logs" / Path(normalized_cli_subdir)

    # Absoluter Override per Umgebung
    env_log_dir = os.environ.get("MT5_TRADING_LOG_DIR", "").strip()
    if env_log_dir:
        return Path(env_log_dir).expanduser()

    # Relativer Unterordner unterhalb des Standard-Logpfads
    env_log_subdir = os.environ.get("MT5_TRADING_LOG_SUBDIR", "").strip()
    if env_log_subdir:
        normalized_subdir = env_log_subdir.replace("\\", "/").strip("/ ")
        if normalized_subdir:
            return base_dir / "logs" / Path(normalized_subdir)

    # Fallback: Standardordner
    return base_dir / "logs"


def configure_logging(log_dir: Path) -> Path:
    """
    Konfiguriert Logging für Terminal + Datei neu und aktualisiert config.LOG_DIR.

    Args:
        log_dir: Zielordner für ``live_trader.log`` und CSV-Dateien.

    Returns:
        Finaler Log-Ordner.
    """
    resolved_log_dir = log_dir.expanduser()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    # Mutable LOG_DIR in config aktualisieren – alle Module lesen es von dort
    config.LOG_DIR = resolved_log_dir

    # force=True stellt sicher, dass ein CLI-Override alte Handler sauber ersetzt
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(resolved_log_dir / "live_trader.log", encoding="utf-8"),
        ],
        force=True,
    )
    return resolved_log_dir


# Logging vorab initialisieren (wird in main() mit korrektem Pfad neu gesetzt)
configure_logging(_resolve_log_dir(BASE_DIR))
logger = logging.getLogger(__name__)


# ============================================================
# Haupt-Loop-Hilfsfunktion
# ============================================================


def neue_kerze_abwarten(
    symbol: str,
    letzte_kerzen_zeit: Optional[datetime],
    timeframe: str = "H1",
) -> bool:
    """
    Prüft ob eine neue Kerze im gewählten Zeitrahmen geöffnet wurde.

    Args:
        symbol:            Handelssymbol
        letzte_kerzen_zeit: Zeitstempel der letzten verarbeiteten Kerze
        timeframe:         Zeitrahmen (H1, M30, M15, M5)

    Returns:
        True wenn neue Kerze verfügbar.
    """
    aktuelle_kerze = mt5_letzte_kerze_uhrzeit(symbol, timeframe)
    if aktuelle_kerze is None:
        logger.debug(f"[{symbol}] mt5_letzte_kerze_uhrzeit() gibt None zurück")
        return False

    # Debug-Logging (alle 60 Sekunden, um Spam zu vermeiden)
    jetzt = time.time()
    if jetzt - float(config._MT5_RUNTIME_STATE.get("kerzen_debug_last_ts", 0.0)) > 60:
        logger.debug(
            f"[{symbol}] Kerzen-Check: Letzte={letzte_kerzen_zeit} | "
            f"Aktuell={aktuelle_kerze}"
        )
        config._MT5_RUNTIME_STATE["kerzen_debug_last_ts"] = jetzt

    return letzte_kerzen_zeit is None or aktuelle_kerze != letzte_kerzen_zeit


# ============================================================
# Trading-Loop
# ============================================================


def trading_loop(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
    symbol: str,
    schwelle: float,
    short_schwelle: Optional[float],
    decision_mapping: str,
    regime_spalte: str,
    two_stage_kongruenz: bool,
    regime_erlaubt: Optional[list],
    paper_trading: bool,
    lot: float,
    modell: object,
    kill_switch_dd: float = KILL_SWITCH_DD_DEFAULT,
    kapital_start: float = 10000.0,
    heartbeat_log: bool = HEARTBEAT_LOG_DEFAULT,
    timeframe: str = "H1",
    atr_sl_aktiv: bool = True,
    atr_sl_faktor: float = 1.5,
    two_stage_config: Optional[dict] = None,
    tp_pct: float = 0.006,
    sl_pct: float = 0.003,
    cooldown_bars: int = 12,
    startup_observation_bars: int = STANDARD_STARTUP_OBSERVATION_BARS,
) -> None:
    """
    Haupt-Schleife: Läuft dauerhaft und wartet auf neue Kerzen im gewählten Zeitrahmen.

    Bei jeder neuen Kerze:
    1. MT5-Daten holen
    2. Features berechnen
    3. Signal generieren (Shadow-Mode: Single-Stage vs. Two-Stage)
    4. Kill-Switch prüfen (Drawdown-Limit!)
    5. Trade ausführen (falls Signal stark genug)

    Args:
        symbol:          Handelssymbol
        schwelle:        Wahrscheinlichkeits-Schwelle (z.B. 0.60)
        short_schwelle:  Optionale Short-Schwelle
        decision_mapping: Mapping-Modus ("class" oder "long_prob")
        regime_spalte:   Regime-Quelle ("market_regime" oder "market_regime_hmm")
        two_stage_kongruenz: True=Kongruenzfilter aktiv, False=deaktiviert
        regime_erlaubt:  Erlaubte Regime oder None für alle
        paper_trading:   True = Paper-Modus (kein echtes Geld!)
        lot:             Lot-Größe
        modell:          Geladenes LightGBM-Modell (Single-Stage)
        kill_switch_dd:  Max. Drawdown bis zum automatischen Stopp (Standard: 0.15 = 15%)
        kapital_start:   Startkapital für Paper-Tracking und Kill-Switch-Berechnung
        heartbeat_log:   True = schreibe pro neuer Kerze einen CSV-Heartbeat
        atr_sl_aktiv:    True = ATR-basiertes SL (dynamisch), False = festes SL
        atr_sl_faktor:   ATR-Multiplikator für SL (Standard: 1.5)
        two_stage_config: Two-Stage-Konfiguration für Shadow-Mode (optional)
        tp_pct:          Take-Profit in Dezimal (Standard: 0.006 = 0.6%)
        sl_pct:          Stop-Loss Fallback in Dezimal (Standard: 0.003 = 0.3%)
        cooldown_bars:   Mindest-Bars zwischen Trades (Standard: 12, bei M5 = 1h)
        startup_observation_bars: Anzahl neuer Kerzen, die erst nur beobachtet werden.
    """
    modus_str = "PAPER-TRADING" if paper_trading else "⚠️  LIVE-TRADING MIT ECHTEM GELD!"
    regime_str = (
        [REGIME_NAMEN.get(r, str(r)) for r in regime_erlaubt]
        if regime_erlaubt
        else "alle"
    )

    # Two-Stage M5-Takt: Flag vorab setzen (wird für Logging + Loop gebraucht)
    ts_aktiv = two_stage_config is not None and two_stage_config.get("enable", False)
    ts_cfg: dict[str, Any] = cast(dict[str, Any], two_stage_config) if ts_aktiv else {}

    # RRR (Risk-Reward-Ratio) aus tp_pct/sl_pct berechnen
    rrr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    logger.info("=" * 65)
    logger.info(f"LIVE-TRADER GESTARTET – {symbol}")
    logger.info(f"Modus:          {modus_str}")
    logger.info(f"Long-Schwelle:  {schwelle:.0%}")
    if short_schwelle is None:
        logger.info("Short-Schwelle: auto")
    else:
        logger.info(f"Short-Schwelle: {short_schwelle:.0%}")
    logger.info(f"Mapping-Modus:  {decision_mapping}")
    logger.info(f"Regime-Quelle:  {regime_spalte}")
    logger.info(
        f"Kongruenz-Filter: {'aktiv' if two_stage_kongruenz else 'deaktiviert (aggressiv)'}"
    )
    logger.info(f"Regime-Filter:  {regime_str}")
    if atr_sl_aktiv:
        logger.info(
            f"Stop-Loss:      ATR-SL aktiv ({atr_sl_faktor}× ATR_14, dynamisch)"
        )
        logger.info(f"TP/SL-RRR:      {rrr:.1f}:1 (TP = SL × {rrr:.1f})")
    else:
        logger.info(f"TP/SL:          {tp_pct:.1%} / {sl_pct:.1%} (RRR={rrr:.1f}:1)")
    logger.info(f"Cooldown:       {cooldown_bars} Bars zwischen Trades")
    logger.info(
        f"Beobachtung:    {startup_observation_bars} neue Kerzen nur beobachten"
    )
    logger.info(f"Lot-Größe:      {lot}")
    logger.info(
        f"Kill-Switch:    Drawdown > {kill_switch_dd:.0%} → automatischer Stopp"
    )
    logger.info(f"Startkapital:   {kapital_start:,.2f} (für Kill-Switch-Berechnung)")
    logger.info(
        f"Heartbeat-Log:  {'aktiv' if heartbeat_log else 'aus'} "
        "(CSV-Update pro Kerze)"
    )
    logger.info(f"Logs:           {config.LOG_DIR}")
    logger.info(f"Zeitrahmen:     {timeframe}")
    if ts_aktiv:
        logger.info(
            f"M5-Takt:        AKTIV → Loop auf {ts_cfg.get('ltf_timeframe', 'M5')}, "
            f"HTF-Bias auf H1 (gecached)"
        )
        logger.info(f"Warte auf neue {ts_cfg.get('ltf_timeframe', 'M5')}-Kerze ...")
    else:
        logger.info(f"Warte auf neue {timeframe}-Kerze ...")
    logger.info("=" * 65)

    letzte_kerzen_zeit: Optional[datetime] = None
    n_signale = 0  # Gesamt-Signale
    n_trades = 0  # Ausgeführte Trades
    bars_seit_letztem_trade = cooldown_bars  # Cooldown-Zähler (startet "bereit")

    # Letzter eröffneter Trade (für Schließungs-Erkennung und PnL-Logging)
    letzter_trade: Optional[dict] = None
    verarbeitete_kerzen = 0  # Anzahl seit Start verarbeiteter neuer Kerzen

    # ---- Kill-Switch: Startkapital ermitteln ----
    if not paper_trading and MT5_VERFUEGBAR:
        account = _mt5_api().account_info()
        if account:
            start_equity = account.equity
            logger.info(
                f"[{symbol}] MT5-Startkapital: {start_equity:,.2f} {account.currency}"
            )
        else:
            logger.warning(
                f"[{symbol}] MT5-Kontodaten nicht lesbar – Kill-Switch nutzt Startkapital {kapital_start}"
            )
            start_equity = kapital_start
    else:
        start_equity = kapital_start
        logger.info(f"[{symbol}] Paper-Startkapital: {start_equity:,.2f} (simuliert)")

    # Simuliertes Kapital für Paper-Modus (wird nach jedem Trade aktualisiert)
    paper_kapital = start_equity

    # ---- Two-Stage M5-Takt: effektiver Zeitrahmen und HTF-Cache ----
    if ts_aktiv:
        effektiver_tf = ts_cfg.get("ltf_timeframe", "M5")
        logger.info(
            f"[{symbol}] ⚡ M5-TAKT AKTIV: Loop-Intervall = {effektiver_tf} "
            f"(alle {TIMEFRAME_CONFIG[effektiver_tf]['minutes_per_bar']} Min) | "
            f"HTF-Bias = H1 (gecached, Update nur bei neuer H1-Kerze)"
        )
    else:
        effektiver_tf = timeframe

    # HTF-Cache Variablen (nur für Two-Stage M5-Takt)
    letzte_htf_kerzen_zeit: Optional[datetime] = None
    cached_h1_df_clean: Optional[pd.DataFrame] = None
    externe_feature_cache: dict = {}

    while True:
        try:
            # ---- Trade-Schließung prüfen (PnL-Logging) ----
            if letzter_trade is not None:
                letzter_trade = offenen_trade_pruefen(
                    symbol, letzter_trade, paper_trading
                )

            # Neue Kerze abwarten (M5 bei Two-Stage, sonst H1)
            if not neue_kerze_abwarten(symbol, letzte_kerzen_zeit, effektiver_tf):
                time.sleep(15)
                continue

            # Neue Kerze erkannt!
            aktuelle_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
            if aktuelle_kerzen_zeit is not None:
                kerzen_ts = aktuelle_kerzen_zeit.strftime("%Y-%m-%d %H:%M:%S")
            else:
                kerzen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\n[{symbol}] Neue {effektiver_tf}-Kerze | {kerzen_ts} UTC")
            verarbeitete_kerzen += 1

            # Cooldown-Zähler bei jeder neuen Kerze erhöhen
            bars_seit_letztem_trade += 1

            externe_features = externe_features_holen(cache=externe_feature_cache)

            # ---- Kill-Switch prüfen ----
            if not paper_trading and MT5_VERFUEGBAR:
                account = _mt5_api().account_info()
                if account:
                    aktuell_equity = account.equity
                else:
                    aktuell_equity = start_equity
            else:
                aktuell_equity = paper_kapital

            if kill_switch_pruefen(
                symbol, start_equity, aktuell_equity, kill_switch_dd, paper_trading
            ):
                # Kill-Switch ausgelöst: Positionen schließen und stoppen
                if not paper_trading:
                    alle_positionen_schliessen(symbol)
                break

            # ==============================================================
            # DATEN LADEN
            # ==============================================================

            if ts_aktiv:
                # ── Two-Stage M5-Takt ──────────────────────────────────
                # A) HTF-Bias (H1) nur aktualisieren wenn neue H1-Kerze da
                neue_h1 = neue_kerze_abwarten(symbol, letzte_htf_kerzen_zeit, "H1")
                if neue_h1 or cached_h1_df_clean is None:
                    htf_df_raw = mt5_daten_holen(symbol, timeframe="H1")
                    if htf_df_raw is not None and len(htf_df_raw) >= 250:
                        htf_df = features_berechnen(htf_df_raw, timeframe="H1")
                        htf_df = externe_features_einfuegen(
                            htf_df, external_features=externe_features
                        )
                        cached_h1_df_clean = htf_df.dropna(subset=FEATURE_SPALTEN)
                        ts_cfg["htf_df"] = cached_h1_df_clean
                        letzte_htf_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, "H1")
                        logger.info(
                            f"[{symbol}] 📊 HTF-Bias aktualisiert (neue H1-Kerze) | "
                            f"HTF-Bars={len(cached_h1_df_clean)}"
                        )
                    else:
                        logger.warning(
                            f"[{symbol}] HTF-H1-Daten unzureichend "
                            f"({len(htf_df_raw) if htf_df_raw is not None else 0} Bars)"
                        )
                        if cached_h1_df_clean is None:
                            letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(
                                symbol, effektiver_tf
                            )
                            time.sleep(30)
                            continue

                # B) LTF-Daten (M5) frisch laden bei jeder M5-Kerze
                ltf_tf = ts_cfg["ltf_timeframe"]
                ltf_df_raw = mt5_daten_holen(symbol, timeframe=ltf_tf)
                ltf_min_bars = (
                    250
                    * TIMEFRAME_CONFIG.get(ltf_tf, {"bars_per_hour": 1})[
                        "bars_per_hour"
                    ]
                )
                if ltf_df_raw is None or len(ltf_df_raw) < ltf_min_bars:
                    logger.warning(
                        f"[{symbol}] LTF-{ltf_tf}-Daten unzureichend "
                        f"({len(ltf_df_raw) if ltf_df_raw is not None else 0}) – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    time.sleep(30)
                    continue

                ltf_df = features_berechnen(ltf_df_raw, timeframe=ltf_tf)
                ltf_df = externe_features_einfuegen(
                    ltf_df, external_features=externe_features
                )
                ltf_df_clean = ltf_df.dropna(subset=FEATURE_SPALTEN)
                ts_cfg["ltf_df"] = ltf_df_clean

                if len(ltf_df_clean) < 10:
                    logger.warning(
                        f"[{symbol}] LTF nach NaN-Bereinigung zu wenig Zeilen – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    continue

                # df_clean = gecachte H1-Daten (Single-Stage Baseline-Vergleich)
                df_clean = cached_h1_df_clean

            else:
                # ── Single-Stage (H1) ────────────────────────────────
                df = mt5_daten_holen(symbol, timeframe=timeframe)
                min_bars = (
                    250
                    * TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["H1"])[
                        "bars_per_hour"
                    ]
                )
                if df is None or len(df) < min_bars:
                    logger.warning(
                        f"[{symbol}] Zu wenige Daten ({len(df) if df is not None else 0}) – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    time.sleep(30)
                    continue

                df = features_berechnen(df, timeframe=timeframe)
                df = externe_features_einfuegen(df, external_features=externe_features)

                # NaN-Zeilen am Anfang (Warm-Up) entfernen
                modell_feature_namen = _modell_feature_namen(modell)
                dropna_features = [f for f in modell_feature_namen if f in df.columns]
                df_clean = df.dropna(subset=dropna_features)
                if len(df_clean) < 10:
                    logger.warning(
                        f"[{symbol}] Nach NaN-Bereinigung zu wenige Zeilen – übersprungen"
                    )
                    letzte_kerzen_zeit = mt5_letzte_kerze_uhrzeit(symbol, effektiver_tf)
                    continue

            # Paper-Trades gegen die inzwischen abgeschlossenen Kerzen prüfen
            markt_df_raw = ltf_df_raw if ts_aktiv else cast(pd.DataFrame, df)
            if paper_trading and letzter_trade is not None:
                letzter_trade, paper_kapital = paper_trade_pruefen_und_loggen(
                    symbol=symbol,
                    letzter_trade=letzter_trade,
                    markt_df=markt_df_raw,
                    paper_kapital=paper_kapital,
                    timeframe=effektiver_tf,
                )

            # ---- Signal generieren (Shadow-Mode) ----
            signal, prob, regime, atr_pct = shadow_signal_generieren(
                symbol=symbol,
                df=df_clean,
                modell=modell,
                schwelle=schwelle,
                short_schwelle=short_schwelle,
                decision_mapping=decision_mapping,
                regime_spalte=regime_spalte,
                two_stage_kongruenz=two_stage_kongruenz,
                regime_erlaubt=regime_erlaubt,
                two_stage_config=two_stage_config,
            )
            regime_name = REGIME_NAMEN.get(regime, "?")

            # ATR-basiertes Stop-Loss berechnen (dynamisch!)
            if atr_sl_aktiv and atr_pct > 0:
                sl_aktuell = atr_pct * atr_sl_faktor
                tp_aktuell = sl_aktuell * rrr
                sl_info = (
                    f"ATR-SL={sl_aktuell:.2%} | TP={tp_aktuell:.2%} (RRR {rrr:.1f}:1)"
                )
            else:
                sl_aktuell = sl_pct
                tp_aktuell = tp_pct
                sl_info = f"Fix-SL={sl_aktuell:.1%} | Fix-TP={tp_aktuell:.1%}"

            logger.info(
                f"[{symbol}] Signal={signal} | Prob={prob:.1%} | "
                f"Regime={regime} ({regime_name}) | {sl_info}"
            )

            # Signal/Heartbeat in CSV loggen
            signal_kerze = _letzte_geschlossene_kerze(df_clean)
            close_preis = float(signal_kerze["close"])
            signal_kerzen_zeit = cast(pd.Timestamp, signal_kerze.name).to_pydatetime()
            if signal != 0 and close_preis > 0:
                if signal == 2:  # Long
                    log_sl = round(close_preis * (1.0 - sl_aktuell), 5)
                    log_tp = round(close_preis * (1.0 + tp_aktuell), 5)
                else:  # Short
                    log_sl = round(close_preis * (1.0 + sl_aktuell), 5)
                    log_tp = round(close_preis * (1.0 - tp_aktuell), 5)
            else:
                log_sl = 0.0
                log_tp = 0.0

            if signal != 0:
                n_signale += 1
            if heartbeat_log or signal != 0:
                log_htf_bias = (
                    two_stage_config.get("last_htf_bias") if two_stage_config else None
                )
                log_ltf_signal = (
                    two_stage_config.get("last_ltf_signal")
                    if two_stage_config
                    else None
                )
                trade_loggen(
                    symbol,
                    signal,
                    prob,
                    regime,
                    paper_trading,
                    entry_price=close_preis,
                    sl_price=log_sl,
                    tp_price=log_tp,
                    htf_bias=log_htf_bias,
                    ltf_signal=log_ltf_signal,
                )

            # ---- Trade ausführen ----
            if signal != 0:
                if verarbeitete_kerzen <= startup_observation_bars:
                    logger.info(
                        f"[{symbol}] 👀 Beobachtungsphase aktiv – Kerze {verarbeitete_kerzen}/{startup_observation_bars}. "
                        f"Signal wird nur beobachtet, noch kein Trade."
                    )
                elif bars_seit_letztem_trade < cooldown_bars:
                    logger.info(
                        f"[{symbol}] Cooldown aktiv – noch {cooldown_bars - bars_seit_letztem_trade} "
                        f"Bars warten (von {cooldown_bars})"
                    )
                elif paper_trading and letzter_trade is not None:
                    logger.info(
                        f"[{symbol}] Bereits offener Paper-Trade – kein neuer Trade"
                    )
                elif mt5_offene_position(symbol):
                    logger.info(
                        f"[{symbol}] Bereits offene Position – kein neuer Trade"
                    )
                else:
                    order_info = order_senden(
                        symbol, signal, lot, tp_aktuell, sl_aktuell, paper_trading
                    )
                    if order_info is not None:
                        n_trades += 1
                        bars_seit_letztem_trade = 0
                        richtung_str = "Long" if signal == 2 else "Short"
                        logger.info(
                            f"[{symbol}] Trade #{n_trades}: {richtung_str} | "
                            f"Prob={prob:.1%} | Regime={regime_name}"
                        )

                        trade_htf = (
                            two_stage_config.get("last_htf_bias")
                            if two_stage_config
                            else None
                        )
                        trade_ltf = (
                            two_stage_config.get("last_ltf_signal")
                            if two_stage_config
                            else None
                        )
                        if not paper_trading and MT5_VERFUEGBAR:
                            letzter_trade = {
                                "position_ticket": int(
                                    order_info.get("position_ticket", 0) or 0
                                ),
                                "deal_ticket": int(
                                    order_info.get("deal_ticket", 0) or 0
                                ),
                                "richtung": signal,
                                "entry_price": float(
                                    order_info.get("entry_price", close_preis)
                                ),
                                "open_zeit": datetime.now(timezone.utc),
                                "htf_bias": trade_htf,
                                "ltf_signal": trade_ltf,
                            }
                        else:
                            # Paper-Modus: Trade bis TP/SL über Kerzen verfolgen
                            letzter_trade = {
                                "position_ticket": 0,
                                "deal_ticket": 0,
                                "richtung": signal,
                                "entry_price": close_preis,
                                "sl_price": log_sl,
                                "tp_price": log_tp,
                                "lot": lot,
                                "entry_bar_time": signal_kerzen_zeit,
                                "open_zeit": datetime.now(timezone.utc),
                                "htf_bias": trade_htf,
                                "ltf_signal": trade_ltf,
                            }
            else:
                logger.info(f"[{symbol}] Kein Trade-Signal (Details siehe oben)")

            # Zeitstempel der verarbeiteten Kerze speichern
            letzte_kerzen_zeit = aktuelle_kerzen_zeit

            # Statistik (bei M5 ≈ alle 8 Stunden, bei H1 ≈ alle 4 Tage)
            stat_intervall = 100 if ts_aktiv else 24
            if n_trades > 0 and n_trades % stat_intervall == 0:
                dd_aktuell = (
                    (start_equity - paper_kapital) / start_equity
                    if paper_trading
                    else 0.0
                )
                logger.info(
                    f"[{symbol}] Status: {n_trades} Trades | {n_signale} Signale | "
                    f"Modus: {'Paper' if paper_trading else 'LIVE'} | "
                    f"Sim-Kapital: {paper_kapital:,.2f} | DD: {dd_aktuell:.1%}"
                )

        except KeyboardInterrupt:
            logger.info(f"\n[{symbol}] Trader gestoppt (Ctrl+C)")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"[{symbol}] Fehler in Haupt-Schleife: {e}", exc_info=True)
            logger.info("Warte 60 Sekunden vor Neustart ...")
            time.sleep(60)


# ============================================================
# Hauptprogramm
# ============================================================


def main() -> None:  # pylint: disable=too-many-locals,too-many-branches
    """Startet den Live-Trader für ein Symbol."""

    parser = argparse.ArgumentParser(
        description=(
            "MT5 ML-Trading – Live-Trader (Phase 7)\n"
            "Läuft auf: Windows 11 Laptop mit MT5-Terminal\n\n"
            "Aktuelle Konfiguration (Paper-Test-Phase):\n"
            "  USDCAD: --symbol USDCAD --schwelle 0.48 --regime_filter 0,1,2,3 --atr_sl 1\n"
            "  USDJPY: --symbol USDJPY --schwelle 0.45 --regime_filter 0,1,2,3 --atr_sl 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default="USDCAD",
        choices=SYMBOLE,
        help=(
            "Handelssymbol (Standard: USDCAD). "
            "Aktive Betriebs-Symbole laut Policy: USDCAD, USDJPY"
        ),
    )
    parser.add_argument(
        "--schwelle",
        type=float,
        default=STANDARD_SCHWELLE,
        help=(
            "Long-Schwelle für Trade-Ausführung (Standard: 0.45). "
            "Bewusst etwas lockerer, damit im Paper-Betrieb mehr Entries sichtbar werden."
        ),
    )
    parser.add_argument(
        "--short_schwelle",
        type=float,
        default=-1.0,
        help=(
            "Optionale Short-Schwelle. "
            "Bei --decision_mapping class: proba_short >= short_schwelle. "
            "Bei --decision_mapping long_prob: proba_long <= short_schwelle. "
            "Standard: -1 (auto)."
        ),
    )
    parser.add_argument(
        "--decision_mapping",
        type=str,
        choices=["class", "long_prob"],
        default="class",
        help=(
            "Signal-Mapping: class (klassische Klassen-Proba) oder long_prob "
            "(Long bei >= long_schwelle, Short bei <= short_schwelle)."
        ),
    )
    parser.add_argument(
        "--regime_source",
        type=str,
        choices=["market_regime", "market_regime_hmm"],
        default="market_regime",
        help=(
            "Quelle für Regime-Filter. "
            "market_regime_hmm ist oft reaktiver (mehr Regime-Wechsel)."
        ),
    )
    parser.add_argument(
        "--two_stage_kongruenz",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "1 = HTF/LTF Kongruenzfilter aktiv (sicherer, weniger Trades), "
            "0 = Kongruenzfilter aus (aggressiver, mehr Trades, Standard)."
        ),
    )
    parser.add_argument(
        "--two_stage_allow_neutral_htf",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "1 = neutraler H1-Bias darf starke M5-Entries trotzdem zulassen (Standard), "
            "0 = neutraler H1-Bias blockiert aktive M5-Signale."
        ),
    )
    parser.add_argument(
        "--regime_filter",
        type=str,
        default="0,1,2,3",
        help=(
            "Komma-getrennte Regime-Nummern (Standard: '0,1,2,3'). "
            "0=Seitwärts, 1=Aufwärtstrend, 2=Abwärtstrend, 3=Hohe Vola. "
            "Paper-Test-Phase: Standardmäßig alle Regime erlaubt für mehr Feedback."
        ),
    )
    parser.add_argument(
        "--lot",
        type=float,
        default=LOT,
        help=f"Lot-Größe pro Trade (Standard: {LOT} = Micro-Lot). NICHT erhöhen ohne Erfahrung!",
    )
    parser.add_argument(
        "--paper_trading",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = Paper-Modus (Standard, empfohlen!), "
            "0 = Live-Modus (erst nach 2 Wochen Testlauf!)"
        ),
    )
    parser.add_argument(
        "--mt5_server",
        default="",
        help="MT5 Broker-Server (z.B. 'ICMarkets-Demo')",
    )
    parser.add_argument(
        "--mt5_login",
        type=int,
        default=0,
        help="MT5 Kontonummer",
    )
    parser.add_argument(
        "--mt5_password",
        default="",
        help="MT5 Passwort",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help=(
            "Optionaler absoluter Log-Ordner. Überschreibt MT5_TRADING_LOG_DIR und "
            "den Standardpfad. Ideal für dedizierte Testläufe wie Test 128."
        ),
    )
    parser.add_argument(
        "--log_subdir",
        default="",
        help=(
            "Optionaler Unterordner unter BASE_DIR/logs. Beispiel: paper_test128. "
            "Wird nur genutzt wenn --log_dir leer ist."
        ),
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Modell-Versions-Suffix (Standard: v4). Muss mit train_model.py übereinstimmen.",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["H1", "M30", "M15"],
        help=(
            "Zeitrahmen für Datenabruf und Modell (Standard: H1). "
            "M30/M15 erfordern separat trainierte Modelle."
        ),
    )
    parser.add_argument(
        "--allow_research_symbol",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 = Policy erzwingen (nur USDCAD/USDJPY handelbar, Standard), "
            "1 = Forschungs-Override fuer andere Symbole (nur Paper-Modus empfohlen)."
        ),
    )
    parser.add_argument(
        "--kill_switch_dd",
        type=float,
        default=KILL_SWITCH_DD_DEFAULT,
        help=(
            f"Kill-Switch: Maximaler Drawdown in Dezimal (Standard: {KILL_SWITCH_DD_DEFAULT} = 15%%). "
            "Wenn Verlust diesen Wert überschreitet, stoppt der Trader automatisch. "
            "Empfehlung: 0.15 (15%%). Nie über 0.20 (20%%) setzen!"
        ),
    )
    parser.add_argument(
        "--kapital_start",
        type=float,
        default=10000.0,
        help=(
            "Startkapital für Kill-Switch-Berechnung im Paper-Modus (Standard: 10000.0 EUR). "
            "Im Live-Modus wird das echte MT5-Kontokapital automatisch ausgelesen."
        ),
    )
    parser.add_argument(
        "--heartbeat_log",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = pro neuer Kerze Heartbeat in CSV schreiben (Standard), "
            "0 = nur bei Signal/Trade schreiben."
        ),
    )
    parser.add_argument(
        "--atr_sl",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "1 = ATR-basiertes Stop-Loss (dynamisch, Standard), "
            "0 = festes Stop-Loss (0.3%%). "
            "ATR-SL verbessert Sharpe um +1.5 bis +2.0 im Backtest."
        ),
    )
    parser.add_argument(
        "--atr_faktor",
        type=float,
        default=1.5,
        help=(
            "ATR-Multiplikator für SL-Berechnung (Standard: 1.5). "
            "SL = ATR_14 × Faktor. Empfehlung aus Backtest: 1.5"
        ),
    )
    parser.add_argument(
        "--two_stage_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0 = Single-Stage (Standard), "
            "1 = Shadow-Mode für Two-Stage (USDCAD/USDJPY, v4-Modelle erforderlich)."
        ),
    )
    parser.add_argument(
        "--two_stage_ltf_timeframe",
        default="M5",
        choices=["M5", "M15"],
        help=(
            "LTF-Zeitrahmen für Two-Stage (Standard: M5). "
            "HTF ist immer H1. Nur relevant wenn --two_stage_enable 1"
        ),
    )
    parser.add_argument(
        "--two_stage_version",
        default="v4",
        help=(
            "Modellversion für Two-Stage-Modelle (Standard: v4). "
            "Erwartet: lgbm_htf_bias_SYMBOL_H1_VERSION.pkl und "
            "lgbm_ltf_entry_SYMBOL_LTF-TF_VERSION.pkl"
        ),
    )
    parser.add_argument(
        "--tp_pct",
        type=float,
        default=0.006,
        help=(
            "Take-Profit in Dezimal (Standard: 0.006 = 0.6%%). "
            "Wird als RRR-Faktor relativ zum SL verwendet wenn ATR-SL aktiv. "
            "Backtest-Optimum: 0.006 (RRR 2:1 mit SL=0.3%%)."
        ),
    )
    parser.add_argument(
        "--sl_pct",
        type=float,
        default=0.003,
        help=(
            "Stop-Loss in Dezimal (Standard: 0.003 = 0.3%%). "
            "Fallback wenn ATR-SL deaktiviert ist."
        ),
    )
    parser.add_argument(
        "--two_stage_cooldown_bars",
        type=int,
        default=STANDARD_TWO_STAGE_COOLDOWN_BARS,
        help=(
            "Cooldown in Bars nach einem Trade (Standard: 3). "
            "Verhindert Overtrading bei dauerhaftem Signal. "
            "Bei M5: 3 Bars = 15 Minuten Pause zwischen Trades."
        ),
    )
    parser.add_argument(
        "--startup_observation_bars",
        type=int,
        default=STANDARD_STARTUP_OBSERVATION_BARS,
        help=(
            "Anzahl neuer Kerzen direkt nach Start, die erst nur beobachtet werden (Standard: 5). "
            "Signale werden geloggt, aber noch nicht gehandelt."
        ),
    )
    args = parser.parse_args()

    # ---- Logging-Ziel möglichst früh fixieren ----
    configure_logging(
        _resolve_log_dir(
            BASE_DIR,
            cli_log_dir=args.log_dir,
            cli_log_subdir=args.log_subdir,
        )
    )

    # ---- ATR-SL Konfiguration aus CLI ----
    atr_sl_aktiv = bool(args.atr_sl)
    atr_sl_faktor = args.atr_faktor
    short_schwelle: Optional[float] = (
        None if float(args.short_schwelle) < 0.0 else float(args.short_schwelle)
    )
    two_stage_kongruenz = bool(args.two_stage_kongruenz)

    # ---- Regime-Filter parsen ----
    regime_erlaubt = None
    if args.regime_filter:
        try:
            regime_erlaubt = [int(r.strip()) for r in args.regime_filter.split(",")]
        except ValueError:
            print(
                f"Ungültiger --regime_filter: '{args.regime_filter}'. Erwartet: z.B. '1,2'"
            )
            return

    # ---- Paper-Trading sicher stellen ----
    paper_trading = bool(args.paper_trading)
    if not paper_trading:
        print("\n" + "⚠️ " * 20)
        print("WARNUNG: Live-Trading Modus aktiv!")
        print("  → Echte Orders mit echtem Geld!")
        print("  → Nur aktivieren nach 2 Wochen erfolgreichen Paper-Tradings!")
        print("  → Weiter? (ja eingeben zum Bestätigen)")
        bestaetigung = input("Bestätigung: ").strip().lower()
        if bestaetigung != "ja":
            print("Abgebrochen. Starte mit --paper_trading 1")
            return
        print("⚠️ " * 20 + "\n")

    # ---- Symbol-Policy prüfen (nur 2 aktive Paare) ----
    symbol = args.symbol.upper()
    if symbol not in AKTIVE_SYMBOLE and not bool(args.allow_research_symbol):
        print(
            "\nPolicy-Block: Dieses Symbol ist aktuell Research-only.\n"
            f"Aktive Betriebs-Symbole: {', '.join(AKTIVE_SYMBOLE)}\n"
            "Wenn du bewusst testen willst, setze --allow_research_symbol 1 "
            "(empfohlen nur mit --paper_trading 1)."
        )
        return

    # Forschungs-Override nur im Paper-Modus zulassen
    if (
        symbol not in AKTIVE_SYMBOLE
        and bool(args.allow_research_symbol)
        and not paper_trading
    ):
        print(
            "Research-Override darf nicht im Live-Modus laufen. "
            "Bitte --paper_trading 1 verwenden."
        )
        return

    # ---- Modell laden ----
    timeframe = args.timeframe.upper()
    if timeframe == "H1":
        modell_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{args.version}.pkl"
    else:
        modell_pfad = (
            MODEL_DIR / f"lgbm_{symbol.lower()}_{timeframe}_{args.version}.pkl"
        )
    if not modell_pfad.exists():
        modell_name_hinweis = (
            f"lgbm_{symbol.lower()}_{args.version}.pkl"
            if timeframe == "H1"
            else f"lgbm_{symbol.lower()}_{timeframe}_{args.version}.pkl"
        )
        print(
            f"Modell nicht gefunden: {modell_pfad}\n"
            f"Bitte Modell vom Linux-Server übertragen:\n"
            f"  scp SERVER:/mnt/1Tb-Data/XGBoost-LightGBM/models/"
            f"{modell_name_hinweis} ./models/"
        )
        return

    logger.info(f"Lade Modell: {modell_pfad.name}")
    modell = joblib.load(modell_pfad)
    logger.info("Modell geladen ✓")

    # ---- MT5 verbinden (wenn Zugangsdaten vorhanden) ----
    if args.mt5_server and args.mt5_login and args.mt5_password:
        verbunden = mt5_verbinden(args.mt5_server, args.mt5_login, args.mt5_password)
        if not verbunden:
            logger.warning(
                "MT5-Verbindung fehlgeschlagen – starte im simulierten Modus.\n"
                "Stelle sicher dass MT5 Terminal geöffnet und eingeloggt ist!"
            )
            if not paper_trading:
                print("MT5-Verbindung für Live-Trading erforderlich! Abgebrochen.")
                return
    else:
        logger.info(
            "Keine MT5-Zugangsdaten angegeben – Paper-Trading ohne MT5-Verbindung.\n"
            "Für echte Verbindung: --mt5_server SERVER --mt5_login NUMMER --mt5_password PW"
        )

    # ---- MT5-Check für Live-Modus ----
    if not paper_trading and not MT5_VERFUEGBAR:
        print("MT5 nicht installiert! Live-Trading nicht möglich.")
        return

    # ---- Kill-Switch-Parameter validieren ----
    if args.kill_switch_dd > 0.20:
        logger.warning(
            f"⚠️  Kill-Switch-Limit {args.kill_switch_dd:.0%} ist sehr hoch! "
            "Empfehlung: max. 20% (0.20). Bitte überprüfen."
        )

    # ---- Two-Stage-Konfiguration vorbereiten (Shadow-Mode) ----
    two_stage_config = None
    TWO_STAGE_APPROVED = {"USDCAD", "USDJPY"}
    if bool(args.two_stage_enable):
        if symbol.upper() not in TWO_STAGE_APPROVED:
            logger.info(
                f"[{symbol}] Two-Stage-Shadow-Mode nur für {TWO_STAGE_APPROVED} freigeschaltet. "
                "Verwende Single-Stage."
            )
        else:
            logger.info(
                f"[{symbol}] Two-Stage-Shadow-Mode aktiv: HTF=H1, LTF={args.two_stage_ltf_timeframe}, "
                f"Version={args.two_stage_version}"
            )
            # Feature-Listen aus JSON-Metadatei laden (exakt wie beim Training)
            meta_pfad = (
                MODEL_DIR
                / f"two_stage_{symbol.lower()}_{args.two_stage_ltf_timeframe}_{args.two_stage_version}.json"
            )
            if meta_pfad.exists():
                with open(meta_pfad, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                htf_feats = meta["htf_features"]
                ltf_feats = meta["ltf_features"]
                logger.info(
                    f"[{symbol}] Metadaten geladen: HTF={len(htf_feats)} Features, "
                    f"LTF={len(ltf_feats)} Features"
                )
            else:
                logger.warning(
                    f"[{symbol}] Metadatei {meta_pfad.name} nicht gefunden – "
                    "verwende Standard-Feature-Liste"
                )
                htf_feats = FEATURE_SPALTEN
                ltf_feats = FEATURE_SPALTEN

            two_stage_config = {
                "enable": True,
                "ltf_timeframe": args.two_stage_ltf_timeframe,
                "version": args.two_stage_version,
                "allow_neutral_htf_entries": bool(args.two_stage_allow_neutral_htf),
                "htf_features": htf_feats,
                "ltf_features": ltf_feats,
                # HTF/LTF DataFrames werden in trading_loop dynamisch geladen
                "htf_df": None,
                "ltf_df": None,
            }

    # ---- Hauptschleife starten ----
    trading_loop(
        symbol=symbol,
        schwelle=args.schwelle,
        short_schwelle=short_schwelle,
        decision_mapping=args.decision_mapping,
        regime_spalte=args.regime_source,
        two_stage_kongruenz=two_stage_kongruenz,
        regime_erlaubt=regime_erlaubt,
        paper_trading=paper_trading,
        lot=args.lot,
        modell=modell,
        kill_switch_dd=args.kill_switch_dd,
        kapital_start=args.kapital_start,
        heartbeat_log=bool(args.heartbeat_log),
        timeframe=timeframe,
        atr_sl_aktiv=atr_sl_aktiv,
        atr_sl_faktor=atr_sl_faktor,
        two_stage_config=two_stage_config,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        cooldown_bars=args.two_stage_cooldown_bars,
        startup_observation_bars=args.startup_observation_bars,
    )

    # MT5 Verbindung beenden
    if MT5_VERFUEGBAR and _mt5_api().terminal_info() is not None:
        _mt5_api().shutdown()
        logger.info("MT5-Verbindung getrennt.")


if __name__ == "__main__":
    main()
