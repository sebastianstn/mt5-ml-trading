"""
two_stage_backtest.py – Backtest für Option 1 (HTF H1 -> LTF M5)

Dieses Skript bewertet die neuen Zwei-Stufen-Modelle und vergleicht sie mit
Single-Stage-H1 auf einem gemeinsamen Zeitraum.

Ablauf:
    1) Zwei-Stufen-Signale für M5 erzeugen (HTF-Bias + LTF-Entry)
    2) Trades mit bestehender Backtest-Engine simulieren
    3) KPI-Gates prüfen (Sharpe, Profit Factor, Max Drawdown)
    4) Baseline-Vergleich gegen Single-Stage-H1 (gleiches Zeitfenster)

Läuft auf:
    Linux-Server
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Projektwurzel auf den Python-Pfad setzen, damit lokale Module importierbar sind.
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Bestehende Backtest-Logik über Dateipfad laden (Ordner ist kein Python-Package).
BACKTEST_MODUL_PFAD = BASE_DIR / "backtest" / "backtest.py"
_spec = importlib.util.spec_from_file_location("backtest_mod", BACKTEST_MODUL_PFAD)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Konnte backtest.py nicht laden: {BACKTEST_MODUL_PFAD}")
_backtest_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backtest_mod)

# Benötigte Symbole aus dem geladenen Modul referenzieren.
AUSSCHLUSS_SPALTEN = _backtest_mod.AUSSCHLUSS_SPALTEN
BACKTEST_DIR = _backtest_mod.BACKTEST_DIR
kennzahlen_berechnen = _backtest_mod.kennzahlen_berechnen
symbol_backtest = _backtest_mod.symbol_backtest
trades_simulieren = _backtest_mod.trades_simulieren
RisikoConfig = _backtest_mod.RisikoConfig


def _labeled_pfad(symbol: str, timeframe: str, version: str) -> Path:
    """Ermittelt den Pfad zur gelabelten CSV-Datei.

    Args:
        symbol: Handelssymbol (z.B. "USDCAD").
        timeframe: Zeitrahmen ("H1" oder "M5").
        version: Versionssuffix (z.B. "v4").

    Returns:
        Pfad zur gelabelten Datei.
    """
    if version == "v1":
        return BASE_DIR / "data" / f"{symbol}_{timeframe}_labeled.csv"
    return BASE_DIR / "data" / f"{symbol}_{timeframe}_labeled_{version}.csv"


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extrahiert Modell-Features ohne Rohpreis-/Label-Spalten.

    Args:
        df: Eingabe-DataFrame.

    Returns:
        Feature-DataFrame ohne ausgeschlossene Spalten.
    """
    feature_cols = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    x = df[feature_cols].copy()
    if x.isna().any().any():
        # Sicherheitsnetz: NaN-Werte mit Median je Spalte auffüllen.
        x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return x


def _htf_bias_features(h1_df: pd.DataFrame, htf_model: object) -> pd.DataFrame:
    """Berechnet HTF-Bias-Features und verzögert sie um 1 H1-Kerze.

    Args:
        h1_df: H1-Daten mit Features.
        htf_model: Trainiertes HTF-Bias-Modell.

    Returns:
        Bias-Features auf H1-Index (leakage-sicher durch shift(1)).
    """
    # Exakt dieselben Features verwenden wie beim Training gespeichert.
    # So vermeiden wir Mismatches, wenn sich globale Ausschlusslisten ändern.
    x_h1 = _feature_matrix(h1_df)
    if hasattr(htf_model, "feature_name_") and htf_model.feature_name_:
        model_features = list(htf_model.feature_name_)
        fehlend = [f for f in model_features if f not in x_h1.columns]
        if fehlend:
            raise ValueError(
                "HTF-Backtest: Fehlende Modell-Features in H1-Daten: " f"{fehlend}"
            )
        x_h1 = x_h1[model_features]
    proba = htf_model.predict_proba(x_h1)
    pred_class = np.argmax(proba, axis=1)

    htf_bias = pd.DataFrame(
        {
            "htf_bias_class": pred_class.astype(int),
            "htf_bias_prob_short": proba[:, 0],
            "htf_bias_prob_neutral": proba[:, 1],
            "htf_bias_prob_long": proba[:, 2],
        },
        index=h1_df.index,
    )

    # Kritisch gegen Look-Ahead-Bias: nur abgeschlossene H1-Kerze nutzbar.
    return htf_bias.shift(1)


def _project_bias_to_ltf(
    ltf_df: pd.DataFrame, htf_bias_df: pd.DataFrame
) -> pd.DataFrame:
    """Projiziert H1-Bias auf M5-Zeitstempel via backward asof-merge.

    Args:
        ltf_df: M5-Daten.
        htf_bias_df: Verzögerte H1-Bias-Daten.

    Returns:
        M5-DataFrame inkl. HTF-Bias-Spalten.
    """
    ltf_reset = ltf_df.reset_index().rename(columns={"time": "timestamp"})
    htf_reset = htf_bias_df.reset_index().rename(columns={"time": "timestamp"})

    merged = pd.merge_asof(
        ltf_reset.sort_values("timestamp"),
        htf_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp")
    merged.index.name = "time"
    return merged


def two_stage_signale_generieren(
    symbol: str,
    version: str,
    schwelle: float,
    ltf_timeframe: str = "M5",
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Erzeugt Two-Stage-Signale für ein Symbol auf LTF-Daten.

    Args:
        symbol: Handelssymbol.
        version: Versionssuffix (z.B. v4).
        schwelle: Mindestwahrscheinlichkeit für Long/Short.
        ltf_timeframe: LTF-Zeitrahmen (Standard: M5).

    Returns:
        Tuple aus
        - DataFrame mit `signal` und `prob_signal`
        - Startzeitpunkt des verfügbaren M5-Zeitraums
        - Endzeitpunkt des verfügbaren M5-Zeitraums
    """
    models_dir = BASE_DIR / "models"

    meta_path = (
        models_dir / f"two_stage_{symbol.lower()}_{ltf_timeframe}_{version}.json"
    )
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta-Datei nicht gefunden: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    htf_model_path = models_dir / meta["htf_model"]
    ltf_model_path = models_dir / meta["ltf_model"]
    if not htf_model_path.exists() or not ltf_model_path.exists():
        raise FileNotFoundError(
            f"Two-Stage-Modelle fehlen: {htf_model_path.name}, {ltf_model_path.name}"
        )

    # Gelabelte Daten laden (enthalten OHLC + Features + market_regime).
    h1_df = pd.read_csv(
        _labeled_pfad(symbol, "H1", version),
        index_col="time",
        parse_dates=True,
    ).sort_index()
    ltf_df = pd.read_csv(
        _labeled_pfad(symbol, ltf_timeframe, version),
        index_col="time",
        parse_dates=True,
    ).sort_index()

    # Modelle laden.
    htf_model = joblib.load(htf_model_path)
    ltf_model = joblib.load(ltf_model_path)

    # HTF-Bias berechnen und auf LTF projizieren.
    htf_bias_h1 = _htf_bias_features(h1_df, htf_model)
    ltf_mit_bias = _project_bias_to_ltf(ltf_df, htf_bias_h1)

    # Startphase ohne verfügbaren Bias entfernen.
    bias_cols = [
        "htf_bias_class",
        "htf_bias_prob_short",
        "htf_bias_prob_neutral",
        "htf_bias_prob_long",
    ]
    ltf_mit_bias = ltf_mit_bias.dropna(subset=bias_cols).copy()

    # Featureliste strikt aus Meta verwenden (Trainingskonsistenz).
    ltf_features = meta["ltf_features"]
    x_ltf = ltf_mit_bias.reindex(columns=ltf_features).copy()
    x_ltf = x_ltf.fillna(x_ltf.median(numeric_only=True)).fillna(0.0)
    if "htf_bias_class" in x_ltf.columns:
        x_ltf["htf_bias_class"] = x_ltf["htf_bias_class"].astype(int)

    # Wahrscheinlichkeiten und Klassen berechnen.
    proba = ltf_model.predict_proba(x_ltf)
    raw_pred = np.argmax(proba, axis=1)

    signal = np.zeros(len(x_ltf), dtype=int)
    prob_signal = np.zeros(len(x_ltf), dtype=float)

    # Schwellenwertlogik analog zum bestehenden Backtest.
    for i in range(len(x_ltf)):
        if raw_pred[i] == 2 and proba[i, 2] > schwelle:
            signal[i] = 2
            prob_signal[i] = float(proba[i, 2])
        elif raw_pred[i] == 0 and proba[i, 0] > schwelle:
            signal[i] = -1
            prob_signal[i] = float(proba[i, 0])

    result = ltf_mit_bias.copy()
    result["signal"] = signal
    result["prob_signal"] = prob_signal

    return result, result.index[0], result.index[-1]


def _gates(kennzahlen: dict) -> dict[str, bool]:
    """Prüft KPI-Gates für die Go/No-Go-Entscheidung.

    Args:
        kennzahlen: Kennzahlen-Dict aus `kennzahlen_berechnen`.

    Returns:
        Dict mit boolschen Ergebnissen pro Gate.
    """
    return {
        "Sharpe>0.8": kennzahlen["sharpe_ratio"] > 0.8,
        "PF>1.3": kennzahlen["gewinnfaktor"] > 1.3,
        "MaxDD>-10%": kennzahlen["max_drawdown_pct"] > -10.0,
    }


def main() -> None:
    """Haupteinstieg für Two-Stage-Backtest + Baseline-Vergleich."""
    parser = argparse.ArgumentParser(description="Two-Stage Backtest (HTF H1 / LTF M5)")
    parser.add_argument("--symbol", nargs="+", default=["USDCAD", "USDJPY"])
    parser.add_argument("--version", default="v4")
    parser.add_argument("--ltf_timeframe", default="M5", choices=["M5", "M15"])
    parser.add_argument("--schwelle", type=float, default=0.52)
    parser.add_argument("--regime_filter", type=str, default="0,1,2")
    parser.add_argument("--tp_pct", type=float, default=0.003)
    parser.add_argument("--sl_pct", type=float, default=0.003)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--atr_sl", action="store_true", default=True)
    parser.add_argument("--atr_faktor", type=float, default=1.5)
    parser.add_argument("--spread_faktor", type=float, default=1.0)
    args = parser.parse_args()

    # Zielsymbole normalisieren (nur aktive Paare empfohlen).
    ziel_symbole = [s.upper() for s in args.symbol]

    regime_erlaubt: Optional[list[int]] = None
    if args.regime_filter:
        regime_erlaubt = [
            int(r.strip()) for r in args.regime_filter.split(",") if r.strip()
        ]

    risiko_config = RisikoConfig(
        kapital=0.0,
        risiko_pct=0.01,
        kontrakt_groesse=100_000.0,
        atr_sl=args.atr_sl,
        atr_faktor=args.atr_faktor,
    )

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    print("=" * 90)
    print("TWO-STAGE BACKTEST (HTF H1 / LTF M5) + BASELINE-VERGLEICH")
    print("=" * 90)

    for symbol in ziel_symbole:
        print(f"\n--- {symbol} ---")

        # 1) Two-Stage-Signale erzeugen.
        df_two_stage, von_ts, bis_ts = two_stage_signale_generieren(
            symbol=symbol,
            version=args.version,
            schwelle=args.schwelle,
            ltf_timeframe=args.ltf_timeframe,
        )

        # 2) Trades mit bestehender Engine simulieren.
        trades_ts = trades_simulieren(
            df=df_two_stage,
            symbol=symbol,
            schwelle=args.schwelle,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            regime_erlaubt=regime_erlaubt,
            horizon=args.horizon,
            risiko_config=risiko_config,
            spread_faktor=args.spread_faktor,
            swap_aktiv=False,
            timeframe=args.ltf_timeframe,
        )

        if trades_ts.empty:
            print(f"[WARN] {symbol}: Keine Two-Stage Trades gefunden.")
            continue

        k_ts = kennzahlen_berechnen(
            trades_ts, f"{symbol}_TWO_STAGE_{args.ltf_timeframe}"
        )
        gates_ts = _gates(k_ts)

        # Trades speichern (separater Dateiname für Two-Stage).
        trade_path = (
            BACKTEST_DIR / f"{symbol}_{args.ltf_timeframe}_two_stage_trades.csv"
        )
        trades_ts.to_csv(trade_path)

        # 3) Baseline auf gleichem Zeitraum (fairer Vergleich).
        k_base = symbol_backtest(
            symbol=symbol,
            schwelle=args.schwelle,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            regime_erlaubt=regime_erlaubt,
            version=args.version,
            model_version=None,
            horizon=args.horizon,
            risiko_config=risiko_config,
            zeitraum_von=str(von_ts.date()),
            zeitraum_bis=str(bis_ts.date()),
            spread_faktor=args.spread_faktor,
            swap_aktiv=False,
            timeframe="H1",
        )

        if k_base is None:
            print(f"[WARN] {symbol}: Baseline-H1 konnte nicht berechnet werden.")
            continue

        gates_base = _gates(k_base)

        # 4) Vergleichszeile bauen.
        row = {
            "symbol": symbol,
            "zeitraum_von": str(von_ts.date()),
            "zeitraum_bis": str(bis_ts.date()),
            "two_stage_n_trades": k_ts["n_trades"],
            "two_stage_return_pct": k_ts["gesamtrendite_pct"],
            "two_stage_sharpe": k_ts["sharpe_ratio"],
            "two_stage_pf": k_ts["gewinnfaktor"],
            "two_stage_maxdd_pct": k_ts["max_drawdown_pct"],
            "two_stage_gate_sharpe": gates_ts["Sharpe>0.8"],
            "two_stage_gate_pf": gates_ts["PF>1.3"],
            "two_stage_gate_dd": gates_ts["MaxDD>-10%"],
            "baseline_n_trades": k_base["n_trades"],
            "baseline_return_pct": k_base["gesamtrendite_pct"],
            "baseline_sharpe": k_base["sharpe_ratio"],
            "baseline_pf": k_base["gewinnfaktor"],
            "baseline_maxdd_pct": k_base["max_drawdown_pct"],
            "baseline_gate_sharpe": gates_base["Sharpe>0.8"],
            "baseline_gate_pf": gates_base["PF>1.3"],
            "baseline_gate_dd": gates_base["MaxDD>-10%"],
            "delta_return_pct": round(
                k_ts["gesamtrendite_pct"] - k_base["gesamtrendite_pct"], 2
            ),
            "delta_sharpe": round(k_ts["sharpe_ratio"] - k_base["sharpe_ratio"], 3),
            "delta_pf": round(k_ts["gewinnfaktor"] - k_base["gewinnfaktor"], 3),
            "delta_maxdd_pct": round(
                k_ts["max_drawdown_pct"] - k_base["max_drawdown_pct"], 2
            ),
        }
        rows.append(row)

        print(
            f"Two-Stage: Trades={k_ts['n_trades']} | Rendite={k_ts['gesamtrendite_pct']:+.2f}% | "
            f"Sharpe={k_ts['sharpe_ratio']:.3f} | PF={k_ts['gewinnfaktor']:.3f} | DD={k_ts['max_drawdown_pct']:.2f}%"
        )
        print(
            f"Baseline : Trades={k_base['n_trades']} | Rendite={k_base['gesamtrendite_pct']:+.2f}% | "
            f"Sharpe={k_base['sharpe_ratio']:.3f} | PF={k_base['gewinnfaktor']:.3f} | DD={k_base['max_drawdown_pct']:.2f}%"
        )

    if not rows:
        print("\n[FEHLER] Keine Vergleichsergebnisse erzeugt.")
        sys.exit(1)

    df_summary = pd.DataFrame(rows)
    summary_path = BACKTEST_DIR / "two_stage_backtest_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("ZUSAMMENFASSUNG GESPEICHERT:")
    print(f"  {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
