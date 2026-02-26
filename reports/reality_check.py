"""
reality_check.py – Out-of-Sample Reality-Check für die letzten N Monate

Analysiert ob die Backtest-Trades der letzten Monate sinnvoll und robust wirken.
Dies ist kein Backtest (der ist in backtest.py), sondern eine Post-Hoc-Analyse
der bereits gespeicherten Trades aus backtest/SYMBOL_trades.csv.

Analysiert (Review-Punkt 10):
    1. Signal-Verteilung (Long vs. Short – ausgeglichen?)
    2. Durchschnittliche Wahrscheinlichkeit (über 0.60 = gutes Vertrauen?)
    3. Regime-Verteilung bei Einstieg (handeln wir in den richtigen Phasen?)
    4. Monatlicher P&L-Verlauf (konsistentes Ergebnis oder Einmaleffekte?)
    5. Exit-Typen (TP vs. SL vs. Horizon – zu viele SL = schlechte Signale?)

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python reports/reality_check.py                          # Alle Symbole, 3 Monate
    python reports/reality_check.py --symbol USDCAD USDJPY  # Nur diese Symbole
    python reports/reality_check.py --symbol alle --monate 6

Eingabe:  backtest/SYMBOL_trades.csv (aus backtest.py)
Ausgabe:  reports/SYMBOL_reality_check.txt  (Text-Bericht)
          plots/SYMBOL_reality_check.png    (4 Subplots)
"""

# Standard-Bibliotheken
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Datenverarbeitung
import pandas as pd

# Visualisierung
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position
import seaborn as sns  # noqa: E402  # pylint: disable=wrong-import-position

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Pfade und Konstanten
# ============================================================

BASE_DIR     = Path(__file__).parent.parent
BACKTEST_DIR = BASE_DIR / "backtest"
REPORTS_DIR  = Path(__file__).parent
PLOTS_DIR    = BASE_DIR / "plots"

SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# Regime-Namen (aus regime_detection.py)
REGIME_NAMEN = {
    0: "Seitwärts",
    1: "Aufwärtstrend",
    2: "Abwärtstrend",
    3: "Hohe Volatilität",
}

# Zielwerte für die Bewertung (gleiche Maßstäbe wie Roadmap)
ZIEL_PROB_MIN    = 0.60   # Durchschnittliche Signal-Wahrscheinlichkeit ≥ 0.60
ZIEL_WIN_RATE    = 0.45   # Win-Rate ≥ 45% (bei 1:1 RRR)
ZIEL_MAX_SL_RATE = 0.60   # SL-Exit-Rate ≤ 60% (unter 60% = OK)


# ============================================================
# 1. Trade-Daten laden
# ============================================================


def trades_laden(
    symbol: str,
    monate: int = 3,
) -> Optional[pd.DataFrame]:
    """Lädt die gespeicherten Trades und filtert auf die letzten N Monate.

    Args:
        symbol: Handelssymbol (z.B. "USDCAD")
        monate: Anzahl Monate rückwirkend (Standard: 3)

    Returns:
        Gefilterter DataFrame oder None wenn Datei fehlt.
    """
    pfad = BACKTEST_DIR / f"{symbol}_trades.csv"
    if not pfad.exists():
        logger.warning(
            f"[{symbol}] Keine Trades-CSV gefunden: {pfad}\n"
            f"  Bitte zuerst backtest.py ausführen."
        )
        return None

    df = pd.read_csv(pfad, index_col=0, parse_dates=True)

    if df.empty:
        logger.warning(f"[{symbol}] Trades-CSV ist leer.")
        return None

    # Zeitzone normalisieren (backtest.py speichert mit UTC-Offset)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Auf letzte N Monate filtern
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=monate)
    df_gefiltert = df[df.index >= cutoff]

    if df_gefiltert.empty:
        logger.warning(
            f"[{symbol}] Keine Trades in den letzten {monate} Monaten. "
            f"Letzter Trade: {df.index[-1].strftime('%Y-%m-%d')}"
        )
        # Wenn keine aktuellen Trades: alle Trades zurückgeben (für älteren Zeitraum)
        logger.info(f"[{symbol}] Verwende alle {len(df)} vorhandenen Trades.")
        return df

    logger.info(
        f"[{symbol}] {len(df_gefiltert)}/{len(df)} Trades "
        f"(letzte {monate} Monate)"
    )
    return df_gefiltert


# ============================================================
# 2. Einzelne Analyse-Funktionen
# ============================================================


def signal_verteilung_analysieren(df: pd.DataFrame) -> dict:
    """Analysiert die Verteilung von Long vs. Short Signalen.

    Ein ausgeglichenes System sollte nicht stark zu einer Seite tendieren
    (z.B. 70% Long wäre verdächtig für ein Long-Bias).

    Args:
        df: Trade-DataFrame

    Returns:
        Dict mit Verteilungs-Metriken.
    """
    n_total = len(df)
    n_long  = (df["richtung"] == "Long").sum()
    n_short = (df["richtung"] == "Short").sum()

    long_pct  = n_long  / n_total * 100 if n_total > 0 else 0.0
    short_pct = n_short / n_total * 100 if n_total > 0 else 0.0

    # Balance-Score: 0.5 = perfekt ausgeglichen, 0.0/1.0 = einseitig
    balance = min(long_pct, short_pct) / 50  # 1.0 = perfekt balanced

    return {
        "n_total":   n_total,
        "n_long":    int(n_long),
        "n_short":   int(n_short),
        "long_pct":  round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "balance":   round(balance, 3),  # 0.0 = einseitig, 1.0 = perfekt
    }


def wahrscheinlichkeit_analysieren(df: pd.DataFrame) -> dict:
    """Analysiert die durchschnittliche Signal-Wahrscheinlichkeit.

    Ein gutes System sollte Trades mit hohem Vertrauen generieren (≥ 0.60).
    Niedrige Ø-Prob deutet auf zu niedrigen Schwellenwert hin.

    Args:
        df: Trade-DataFrame (muss 'prob'-Spalte haben)

    Returns:
        Dict mit Wahrscheinlichkeits-Metriken.
    """
    if "prob" not in df.columns:
        return {"avg_prob": 0.0, "min_prob": 0.0, "median_prob": 0.0}

    prob = df["prob"]
    return {
        "avg_prob":    round(float(prob.mean()), 4),
        "median_prob": round(float(prob.median()), 4),
        "min_prob":    round(float(prob.min()), 4),
        "max_prob":    round(float(prob.max()), 4),
        "n_ueber_065": int((prob >= 0.65).sum()),
        "pct_ueber_065": round((prob >= 0.65).mean() * 100, 1),
    }


def regime_verteilung_analysieren(df: pd.DataFrame) -> dict:
    """Analysiert in welchen Markt-Regimes die Trades eingegangen wurden.

    Ideal: System handelt hauptsächlich in Trend-Phasen (Regime 1, 2)
    und vermeidet Seitwärtsmärkte (Regime 0) und hohe Volatilität (Regime 3).

    Args:
        df: Trade-DataFrame (muss 'market_regime'-Spalte haben)

    Returns:
        Dict mit Regime-Verteilung pro Regime.
    """
    if "market_regime" not in df.columns:
        return {}

    # Fehlende Regime-Werte ignorieren
    df_clean = df[df["market_regime"].notna()].copy()
    df_clean["market_regime"] = df_clean["market_regime"].astype(int)

    if df_clean.empty:
        return {}

    verteilung = {}
    for regime_nr, regime_name in REGIME_NAMEN.items():
        n = (df_clean["market_regime"] == regime_nr).sum()
        pct = n / len(df_clean) * 100 if len(df_clean) > 0 else 0.0
        verteilung[regime_nr] = {
            "name":  regime_name,
            "n":     int(n),
            "pct":   round(pct, 1),
        }
    return verteilung


def exit_typen_analysieren(df: pd.DataFrame) -> dict:
    """Analysiert die Verteilung der Exit-Typen (TP, SL, Horizon).

    Zu viele SL-Exits = Signale treffen häufig nicht → System möglicherweise
    schlecht oder Marktbedingungen haben sich verändert.

    Faustregel bei 1:1 RRR:
        Win-Rate ≥ 50% → profitabel (inkl. Spread-Kosten: ≥ 52–53%)
        SL-Rate ≤ 50% → Trade-Selektion funktioniert

    Args:
        df: Trade-DataFrame (muss 'exit_grund'-Spalte haben)

    Returns:
        Dict mit Exit-Typ-Verteilung und Bewertung.
    """
    if "exit_grund" not in df.columns:
        return {}

    n_total   = len(df)
    n_tp      = (df["exit_grund"] == "tp").sum()
    n_sl      = (df["exit_grund"] == "sl").sum()
    n_horizon = (df["exit_grund"] == "horizon").sum()

    tp_pct  = n_tp      / n_total * 100 if n_total > 0 else 0.0
    sl_pct  = n_sl      / n_total * 100 if n_total > 0 else 0.0
    hor_pct = n_horizon / n_total * 100 if n_total > 0 else 0.0

    return {
        "n_tp":      int(n_tp),
        "n_sl":      int(n_sl),
        "n_horizon": int(n_horizon),
        "tp_pct":    round(tp_pct, 1),
        "sl_pct":    round(sl_pct, 1),
        "hor_pct":   round(hor_pct, 1),
        "sl_ok":     sl_pct <= (ZIEL_MAX_SL_RATE * 100),
    }


def monatlicher_pnl_analysieren(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet den monatlichen P&L-Verlauf.

    Konsistenz ist wichtiger als absolute Performance: Ein System das mal
    +5% und mal -3% macht ist besser als eines das +15% und -13% macht.

    Args:
        df: Trade-DataFrame (muss 'pnl_pct'-Spalte haben und DatetimeIndex)

    Returns:
        DataFrame mit Spalten [monat, n_trades, pnl_pct, win_rate]
    """
    if "pnl_pct" not in df.columns or df.empty:
        return pd.DataFrame()

    # Monatliche Gruppierung
    df_copy = df.copy()
    df_copy["monat"] = df_copy.index.to_period("M")

    monatlich = (
        df_copy.groupby("monat")
        .agg(
            n_trades=("pnl_pct", "count"),
            pnl_pct=("pnl_pct", "sum"),
            win_rate=("gewinn", "mean"),
        )
        .reset_index()
    )
    monatlich["pnl_pct"] *= 100  # In Prozent umrechnen
    monatlich["win_rate"] *= 100

    return monatlich


# ============================================================
# 3. Plot erstellen
# ============================================================


def reality_check_plotten(
    df: pd.DataFrame,
    symbol: str,
    signal_info: dict,
    prob_info: dict,
    exit_info: dict,
    regime_info: dict,
    monatlich_df: pd.DataFrame,
) -> None:
    """Erstellt einen 4-Subplot-Plot für den Reality-Check.

    Subplots:
        1. Signal-Verteilung (Long vs. Short – Balkendiagramm)
        2. Wahrscheinlichkeits-Histogramm der Trades
        3. Monatlicher P&L-Verlauf (Balkendiagramm, grün/rot)
        4. Exit-Typ-Verteilung (TP, SL, Horizon – Kuchendiagramm)

    Args:
        df:          Trade-DataFrame
        symbol:      Handelssymbol
        signal_info: Ergebnis aus signal_verteilung_analysieren()
        prob_info:   Ergebnis aus wahrscheinlichkeit_analysieren()
        exit_info:   Ergebnis aus exit_typen_analysieren()
        regime_info: Ergebnis aus regime_verteilung_analysieren()
        monatlich_df: Ergebnis aus monatlicher_pnl_analysieren()
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Reality-Check: {symbol} – Letzte Trades-Analyse",
        fontsize=14,
        fontweight="bold",
    )
    sns.set_style("whitegrid")

    # ── Subplot 1: Signal-Verteilung (Long vs. Short) ────────
    ax1 = axes[0, 0]
    kategorien = ["Long", "Short"]
    werte = [signal_info["n_long"], signal_info["n_short"]]
    farben = ["#2ecc71", "#e74c3c"]
    bars = ax1.bar(kategorien, werte, color=farben, alpha=0.8, edgecolor="white")

    # Beschriftung auf Balken
    for bar, wert in zip(bars, werte):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{wert}\n({wert/signal_info['n_total']*100:.0f}%%)",
            ha="center", va="bottom", fontsize=11,
        )
    balance_farbe = "#2ecc71" if signal_info["balance"] >= 0.7 else "#e74c3c"
    ax1.set_title(
        f"Signal-Verteilung (Balance={signal_info['balance']:.2f})",
        fontsize=11,
    )
    ax1.set_ylabel("Anzahl Trades")
    ax1.set_ylim(0, max(werte) * 1.2 + 1)
    ax1.text(
        0.5, 0.95,
        "OK: Ausgeglichen" if signal_info["balance"] >= 0.7 else "!! Einseitig",
        transform=ax1.transAxes, ha="center", va="top",
        fontsize=10, color=balance_farbe,
    )

    # ── Subplot 2: Wahrscheinlichkeits-Histogramm ────────────
    ax2 = axes[0, 1]
    if "prob" in df.columns:
        ax2.hist(
            df["prob"],
            bins=20,
            color="#3498db",
            alpha=0.8,
            edgecolor="white",
        )
        ax2.axvline(
            prob_info["avg_prob"],
            color="#e74c3c",
            linewidth=2,
            linestyle="--",
            label=f"Ø={prob_info['avg_prob']:.3f}",
        )
        ax2.axvline(
            ZIEL_PROB_MIN,
            color="#f39c12",
            linewidth=1.5,
            linestyle=":",
            label=f"Ziel≥{ZIEL_PROB_MIN}",
        )
        ax2.legend(fontsize=9)
    ax2.set_title(f"Signal-Wahrscheinlichkeit (Ø={prob_info.get('avg_prob', 0):.3f})", fontsize=11)
    ax2.set_xlabel("Wahrscheinlichkeit")
    ax2.set_ylabel("Häufigkeit")

    # ── Subplot 3: Monatlicher P&L-Verlauf ───────────────────
    ax3 = axes[1, 0]
    if not monatlich_df.empty:
        monate_labels = [str(m) for m in monatlich_df["monat"]]
        pnl_werte = monatlich_df["pnl_pct"].values
        bar_farben = ["#2ecc71" if p >= 0 else "#e74c3c" for p in pnl_werte]
        ax3.bar(
            range(len(monate_labels)),
            pnl_werte,
            color=bar_farben,
            alpha=0.8,
            edgecolor="white",
        )
        ax3.set_xticks(range(len(monate_labels)))
        ax3.set_xticklabels(monate_labels, rotation=45, ha="right", fontsize=8)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("P&L (%)")
    ax3.set_title("Monatlicher P&L-Verlauf", fontsize=11)

    # ── Subplot 4: Exit-Typ-Verteilung (Kuchendiagramm) ─────
    ax4 = axes[1, 1]
    if exit_info:
        labels = [
            f"TP ({exit_info['tp_pct']:.0f}%%)",
            f"SL ({exit_info['sl_pct']:.0f}%%)",
            f"Horizon ({exit_info['hor_pct']:.0f}%%)",
        ]
        werte_exit = [exit_info["n_tp"], exit_info["n_sl"], exit_info["n_horizon"]]
        farben_exit = ["#2ecc71", "#e74c3c", "#95a5a6"]

        # Nur Segmente mit Werten > 0 darstellen
        labels_f = [lb for lb, w in zip(labels, werte_exit) if w > 0]
        werte_f = [w for w in werte_exit if w > 0]
        farben_f = [fc for fc, w in zip(farben_exit, werte_exit) if w > 0]

        if werte_f:
            ax4.pie(
                werte_f,
                labels=labels_f,
                colors=farben_f,
                autopct="%1.0f%%",
                startangle=90,
                pctdistance=0.75,
            )
    sl_bewertung = (
        "SL-Rate OK"
        if exit_info.get("sl_ok", True)
        else "!! SL-Rate hoch!"
    )
    ax4.set_title(f"Exit-Typen ({sl_bewertung})", fontsize=11)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_pfad = PLOTS_DIR / f"{symbol}_reality_check.png"
    plt.savefig(plot_pfad, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] Plot gespeichert: {plot_pfad}")


# ============================================================
# 4. Text-Bericht erstellen
# ============================================================


def bericht_schreiben(  # pylint: disable=too-many-arguments
    symbol: str,
    monate: int,
    signal_info: dict,
    prob_info: dict,
    exit_info: dict,
    regime_info: dict,
    monatlich_df: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Schreibt einen strukturierten Text-Bericht.

    Args:
        symbol:       Handelssymbol
        monate:       Analysierter Zeitraum in Monaten
        signal_info:  Ergebnis aus signal_verteilung_analysieren()
        prob_info:    Ergebnis aus wahrscheinlichkeit_analysieren()
        exit_info:    Ergebnis aus exit_typen_analysieren()
        regime_info:  Ergebnis aus regime_verteilung_analysieren()
        monatlich_df: Ergebnis aus monatlicher_pnl_analysieren()
        df:           Originaler Trade-DataFrame
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bericht_pfad = REPORTS_DIR / f"{symbol}_reality_check.txt"

    jetzt = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Zeitraum des Berichts
    if not df.empty:
        von_datum = df.index[0].strftime("%Y-%m-%d")
        bis_datum = df.index[-1].strftime("%Y-%m-%d")
    else:
        von_datum = bis_datum = "N/A"

    zeilen = [
        "=" * 65,
        f"REALITY-CHECK: {symbol}",
        f"Erstellt: {jetzt} | Zeitraum: {von_datum} – {bis_datum}",
        f"Analysierte Monate: letzte {monate} Monate",
        "=" * 65,
        "",
        "1. SIGNAL-VERTEILUNG (ausgeglichen?)",
        "─" * 65,
        f"   Gesamt Trades:  {signal_info['n_total']}",
        f"   Long:           {signal_info['n_long']} ({signal_info['long_pct']:.1f}%%)",
        f"   Short:          {signal_info['n_short']} ({signal_info['short_pct']:.1f}%%)",
        f"   Balance:        {signal_info['balance']:.2f} (1.0 = perfekt, <0.7 = einseitig)",
        f"   Bewertung:      {'✅ Ausgeglichen' if signal_info['balance'] >= 0.7 else '⚠️  Einseitig – prüfe Long/Short Bias!'}",
        "",
        "2. SIGNAL-WAHRSCHEINLICHKEIT (Vertrauen des Modells)",
        "─" * 65,
        f"   Durchschnitt:   {prob_info.get('avg_prob', 0):.4f} (Ziel: ≥ {ZIEL_PROB_MIN:.2f})",
        f"   Median:         {prob_info.get('median_prob', 0):.4f}",
        f"   Min / Max:      {prob_info.get('min_prob', 0):.4f} / {prob_info.get('max_prob', 0):.4f}",
        f"   Über 0.65:      {prob_info.get('n_ueber_065', 0)} Trades ({prob_info.get('pct_ueber_065', 0):.1f}%%)",
        f"   Bewertung:      {'✅ OK' if prob_info.get('avg_prob', 0) >= ZIEL_PROB_MIN else '⚠️  Niedrig – evtl. Schwelle erhöhen'}",
        "",
        "3. REGIME-VERTEILUNG (richtige Marktphasen?)",
        "─" * 65,
    ]

    if regime_info:
        for nr, info in regime_info.items():
            trend_icon = "✅" if nr in (1, 2) and info["pct"] >= 20 else "  "
            zeilen.append(
                f"   Regime {nr} ({info['name']:16}): "
                f"{info['n']:4d} Trades ({info['pct']:5.1f}%%) {trend_icon}"
            )
    else:
        zeilen.append("   Keine Regime-Daten verfügbar.")

    zeilen += [
        "",
        "4. EXIT-TYPEN (Qualität der Trade-Ausführung)",
        "─" * 65,
        f"   TP-Exits:       {exit_info.get('n_tp', 0)} ({exit_info.get('tp_pct', 0):.1f}%%) – Gewinntrades",
        f"   SL-Exits:       {exit_info.get('n_sl', 0)} ({exit_info.get('sl_pct', 0):.1f}%%) – Verlusttrades",
        f"   Horizon-Exits:  {exit_info.get('n_horizon', 0)} ({exit_info.get('hor_pct', 0):.1f}%%) – Zeitablauf",
        f"   Bewertung:      {'✅ SL-Rate OK' if exit_info.get('sl_ok', True) else '⚠️  Zu viele SL-Exits!'}",
        "",
        "5. MONATLICHER P&L-VERLAUF (Konsistenz)",
        "─" * 65,
    ]

    if not monatlich_df.empty:
        for _, row in monatlich_df.iterrows():
            pfeil = "▲" if row["pnl_pct"] >= 0 else "▼"
            zeilen.append(
                f"   {str(row['monat']):8}:  {pfeil} {row['pnl_pct']:+7.2f}%%  "
                f"({int(row['n_trades'])} Trades, "
                f"Win-Rate: {row['win_rate']:.0f}%%)"
            )
    else:
        zeilen.append("   Keine monatlichen Daten verfügbar.")

    # Gesamt-P&L
    if "pnl_pct" in df.columns:
        gesamt_pnl = df["pnl_pct"].sum() * 100
        win_rate   = df["gewinn"].mean() * 100 if "gewinn" in df.columns else 0.0
        zeilen += [
            "",
            f"   Gesamt-P&L:     {gesamt_pnl:+.2f}%%",
            f"   Gesamt Win-Rate: {win_rate:.1f}%% (Ziel: ≥ {ZIEL_WIN_RATE*100:.0f}%%)",
            f"   Bewertung:      {'✅ OK' if win_rate >= ZIEL_WIN_RATE * 100 else '⚠️  Win-Rate niedrig'}",
        ]

    zeilen += [
        "",
        "=" * 65,
        "FAZIT",
        "─" * 65,
    ]

    # Automatisches Fazit
    probleme = []
    staerken = []

    if signal_info["balance"] >= 0.7:
        staerken.append("Signal-Verteilung ausgeglichen")
    else:
        probleme.append("Einseitige Signal-Verteilung (Long/Short Bias)")

    if prob_info.get("avg_prob", 0) >= ZIEL_PROB_MIN:
        staerken.append(f"Hohe Ø-Wahrscheinlichkeit ({prob_info.get('avg_prob', 0):.3f})")
    else:
        probleme.append(
            f"Niedrige Ø-Wahrscheinlichkeit ({prob_info.get('avg_prob', 0):.3f} < {ZIEL_PROB_MIN})"
        )

    if not exit_info.get("sl_ok", True):
        probleme.append(f"Hohe SL-Rate ({exit_info.get('sl_pct', 0):.0f}%%)")
    else:
        staerken.append(f"SL-Rate in Ordnung ({exit_info.get('sl_pct', 0):.0f}%%)")

    if "pnl_pct" in df.columns:
        win_rate = df["gewinn"].mean() * 100 if "gewinn" in df.columns else 0.0
        if win_rate >= ZIEL_WIN_RATE * 100:
            staerken.append(f"Win-Rate OK ({win_rate:.0f}%%)")
        else:
            probleme.append(f"Win-Rate unter Ziel ({win_rate:.0f}%% < {ZIEL_WIN_RATE*100:.0f}%%)")

    zeilen.append("Stärken:")
    for s in staerken:
        zeilen.append(f"  ✅ {s}")
    if probleme:
        zeilen.append("Probleme:")
        for p in probleme:
            zeilen.append(f"  ⚠️  {p}")
    if not probleme:
        zeilen.append("  Keine signifikanten Probleme gefunden.")

    zeilen += [
        "",
        f"Plot:  plots/{symbol}_reality_check.png",
        "=" * 65,
    ]

    # Datei schreiben
    with open(bericht_pfad, "w", encoding="utf-8") as f:
        f.write("\n".join(zeilen))

    logger.info(f"[{symbol}] Bericht gespeichert: {bericht_pfad}")

    # Auch auf der Konsole ausgeben
    print("\n".join(zeilen))


# ============================================================
# 5. Haupt-Analyse für ein Symbol
# ============================================================


def symbol_reality_check(symbol: str, monate: int = 3) -> bool:
    """Führt den vollständigen Reality-Check für ein Symbol durch.

    Args:
        symbol: Handelssymbol (z.B. "USDCAD")
        monate: Anzahl Monate für den Analysezeitraum

    Returns:
        True wenn die Analyse erfolgreich war, False bei Fehler.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Reality-Check – {symbol} (letzte {monate} Monate)")
    logger.info(f"{'=' * 60}")

    # Trades laden
    df = trades_laden(symbol, monate)
    if df is None or df.empty:
        return False

    # Alle Analysen durchführen
    signal_info  = signal_verteilung_analysieren(df)
    prob_info    = wahrscheinlichkeit_analysieren(df)
    exit_info    = exit_typen_analysieren(df)
    regime_info  = regime_verteilung_analysieren(df)
    monatlich_df = monatlicher_pnl_analysieren(df)

    # Plot erstellen
    reality_check_plotten(
        df, symbol,
        signal_info, prob_info,
        exit_info, regime_info,
        monatlich_df,
    )

    # Text-Bericht schreiben
    bericht_schreiben(
        symbol, monate,
        signal_info, prob_info,
        exit_info, regime_info,
        monatlich_df, df,
    )

    return True


# ============================================================
# 6. Hauptprogramm
# ============================================================


def main() -> None:
    """Hauptprogramm: Reality-Check für ein oder alle Symbole."""

    parser = argparse.ArgumentParser(
        description=(
            "MT5 ML-Trading – Out-of-Sample Reality-Check\n"
            "Analysiert die letzten N Monate der Backtest-Trades."
        )
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        default=["alle"],
        help=(
            "Ein oder mehrere Symbole (Standard: alle). "
            "Beispiel: --symbol USDCAD USDJPY"
        ),
    )
    parser.add_argument(
        "--monate",
        type=int,
        default=3,
        help=(
            "Analysezeitraum in Monaten rückwirkend (Standard: 3). "
            "Beispiel: --monate 6 für die letzten 6 Monate."
        ),
    )

    args = parser.parse_args()

    # Symbole bestimmen
    if len(args.symbol) == 1 and args.symbol[0].lower() == "alle":
        ziel_symbole = SYMBOLE
    else:
        ziel_symbole = []
        for sym in args.symbol:
            if sym.upper() in SYMBOLE:
                ziel_symbole.append(sym.upper())
            else:
                print(f"Unbekanntes Symbol: '{sym}'")
                print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
                return

    start_zeit = datetime.now()
    logger.info("=" * 60)
    logger.info("MT5 ML-Trading – Reality-Check Pipeline")
    logger.info(f"Symbole: {', '.join(ziel_symbole)}")
    logger.info(f"Zeitraum: letzte {args.monate} Monate")
    logger.info("=" * 60)

    n_erfolgreich = 0
    for symbol in ziel_symbole:
        ok = symbol_reality_check(symbol, args.monate)
        if ok:
            n_erfolgreich += 1

    dauer_sek = int((datetime.now() - start_zeit).total_seconds())
    print(f"\n{'='*60}")
    print(
        f"Reality-Check abgeschlossen: "
        f"{n_erfolgreich}/{len(ziel_symbole)} Symbole analysiert"
    )
    print(f"Laufzeit: {dauer_sek}s")
    print("Berichte: reports/SYMBOL_reality_check.txt")
    print("Plots:    plots/SYMBOL_reality_check.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
