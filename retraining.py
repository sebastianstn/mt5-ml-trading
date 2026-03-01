"""
retraining.py – Automatisches monatliches Retraining des ML-Modells

Zweck:
    Prüft ob das LightGBM-Modell für ein Symbol neu trainiert werden soll
    und deployed das neue Modell nur wenn es besser (oder gleich gut) ist.

Ablauf:
    1. Trigger prüfen: Rolling Sharpe < Schwelle? (oder --erzwingen)
    2. Neues Modell trainieren (LightGBM + Optuna 30 Trials)
    3. F1-Score auf Validierungs-Set auswerten
    4. Mit F1 des bisherigen Modells vergleichen (JSON-Historie)
    5. Deployment: nur wenn F1_neu >= F1_alt - Toleranz (1%)
    6. Versionierung: v1.pkl → v2.pkl → v3.pkl usw.
    7. F1-Historie aktualisieren (JSON)

Läuft auf: Linux-Server

Verwendung:
    cd /mnt/1T-Data/XGBoost-LightGBM
    source venv/bin/activate
    python retraining.py --symbol USDCAD           # Prüft + trainiert wenn nötig
    python retraining.py --symbol alle             # Alle 7 Symbole
    python retraining.py --symbol alle --erzwingen # Immer trainieren (monatlicher Lauf)
    python retraining.py --symbol EURUSD --sharpe_limit 0.5

Eingabe:  data/SYMBOL_H1_labeled.csv
          backtest/backtest_zusammenfassung.csv (für Sharpe-Prüfung)
          models/SYMBOL_f1_history.json         (bisherige F1-Werte)
Ausgabe:  models/lgbm_SYMBOL_vN.pkl             (versioniertes Modell)
          models/SYMBOL_f1_history.json         (aktualisiert)
"""

# Standard-Bibliotheken
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Datenverarbeitung
import pandas as pd

# ML
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score

# Modell speichern
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Kein Optuna-Spam

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retraining.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Pfade und Konstanten
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
BACKTEST_DIR = BASE_DIR / "backtest"

# Alle 7 Forex-Hauptpaare
SYMBOLE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

# Gleiche Ausschluss-Spalten wie in train_model.py und backtest.py
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

# Zeitliche Aufteilung (MUSS mit train_model.py übereinstimmen!)
TRAIN_BIS = "2021-12-31"  # Training: bis Ende 2021
VAL_BIS = "2022-12-31"  # Validierung: 2022 (für F1-Vergleich)
# Test: 2023+ → NIEMALS für Retraining-Entscheidungen verwenden!

# Standard-Retraining-Parameter
OPTUNA_TRIALS = 30  # Weniger als initiales Training (50), für monatliche Läufe
SHARPE_GRENZWERT = 0.5  # Trigger: Sharpe unter diesem Wert → Retraining empfohlen
F1_TOLERANZ = 0.01  # Toleranz: neues Modell wird deployed wenn F1 >= alt - 1%

# Wichtiges Schema-Fix:
# Retraining-Modelle nutzen EIGENES Versionsschema (rt1, rt2, ...),
# damit keine Verwechslung mit Labeling-/Datenversionen (v1, v2, v3) entsteht.
RETRAINING_VERSION_PREFIX = "rt"


# ============================================================
# 1. F1-Historie verwalten
# ============================================================


def f1_history_pfad(symbol: str) -> Path:
    """Gibt den Pfad zur F1-Historien-JSON zurück.

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        Path zur JSON-Datei
    """
    return MODEL_DIR / f"{symbol.upper()}_f1_history.json"


def f1_history_laden(symbol: str) -> dict:
    """Lädt die F1-Historie für ein Symbol aus der JSON-Datei.

    Wenn keine Geschichte vorhanden, wird ein leeres Dict zurückgegeben.

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        Dict mit Versionsnummern als Keys und F1-Werten als Values.
        Beispiel: {"v1": 0.493, "v2": 0.501}
    """
    pfad = f1_history_pfad(symbol)
    if not pfad.exists():
        # Noch keine Geschichte → leere Historie zurückgeben
        logger.info("[%s] Keine F1-Historie gefunden – starte frisch.", symbol)
        return {}
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def f1_history_speichern(symbol: str, version: str, f1: float) -> None:
    """Speichert einen neuen F1-Eintrag in der JSON-Historie.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versionsnummer (z.B. "v2")
        f1:      F1-Score des neuen Modells auf dem Validierungsset
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pfad = f1_history_pfad(symbol)
    # Bestehende Historie laden (oder leeres Dict)
    historie = f1_history_laden(symbol)
    # Neuen Eintrag hinzufügen
    historie[version] = round(f1, 6)
    # Zeitstempel für Nachvollziehbarkeit hinzufügen
    historie[f"{version}_datum"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(historie, f, indent=2, ensure_ascii=False)
    logger.info("[%s] F1-Historie gespeichert: %s", symbol, pfad)


def aktuellen_f1_laden(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Gibt den F1-Score und die Version des neuesten gespeicherten Modells zurück.

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        Tuple (f1_score, version) oder (None, None) wenn keine Historie vorhanden.
    """
    historie = f1_history_laden(symbol)
    if not historie:
        return None, None
    # Nur echte Versions-Keys (keine Datum-Keys) filtern
    versions = {k: v for k, v in historie.items() if not k.endswith("_datum")}
    if not versions:
        return None, None

    # Neueste Version = höchste Ziffer am Ende (unterstützt v3 UND rt3)
    def _version_sort_key(version: str) -> int:
        match = re.search(r"(\d+)$", version)
        return int(match.group(1)) if match else -1

    neueste_version = max(versions.keys(), key=_version_sort_key)
    return float(versions[neueste_version]), neueste_version


# ============================================================
# 2. Versionsnummer verwalten
# ============================================================


def naechste_versionsnummer(
    symbol: str, prefix: str = RETRAINING_VERSION_PREFIX
) -> str:
    """Berechnet die nächste Versionsnummer für das Modell.

    Sucht alle vorhandenen PKL-Dateien für dieses Symbol und gibt
    die nächste Versionsnummer zurück.

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")
        prefix: Versionsprefix (Standard: "rt" → rt1, rt2, ...)

    Returns:
        Neue Versionsnummer als String, z.B. "rt2" oder "rt3"
    """
    sym_lower = symbol.lower()
    # Alle vorhandenen PKL-Dateien für dieses Symbol finden
    vorhandene = list(MODEL_DIR.glob(f"lgbm_{sym_lower}_{prefix}*.pkl"))
    if not vorhandene:
        # Noch kein Modell vorhanden → erste Version
        return f"{prefix}1"
    # Höchste Versionsnummer extrahieren
    versionen = []
    for pfad in vorhandene:
        # Dateiname: lgbm_eurusd_rt1.pkl → "rt1"
        match = re.search(rf"_{re.escape(prefix)}(\d+)$", pfad.stem)
        if match:
            versionen.append(int(match.group(1)))
    if not versionen:
        return f"{prefix}2"
    return f"{prefix}{max(versionen) + 1}"


def modell_pfad(symbol: str, version: str) -> Path:
    """Gibt den Pfad zum LightGBM-Modell zurück.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versionsnummer (z.B. "v1")

    Returns:
        Path zum PKL-File
    """
    return MODEL_DIR / f"lgbm_{symbol.lower()}_{version}.pkl"


# ============================================================
# 3. Retraining-Trigger prüfen
# ============================================================


def trigger_pruefen(symbol: str, sharpe_limit: float = SHARPE_GRENZWERT) -> bool:
    """Prüft ob ein Retraining für dieses Symbol sinnvoll ist.

    Kriterium: Sharpe Ratio aus der letzten Backtest-Zusammenfassung
    liegt unter dem Grenzwert → Modell-Qualität lässt nach.

    Args:
        symbol:      Handelssymbol (z.B. "EURUSD")
        sharpe_limit: Schwellenwert unterhalb dessen Retraining empfohlen wird

    Returns:
        True wenn Retraining empfohlen, False wenn Modell noch gut.
    """
    zusammenfassung_pfad = BACKTEST_DIR / "backtest_zusammenfassung.csv"

    if not zusammenfassung_pfad.exists():
        # Kein Backtest vorhanden → Retraining vorsichtshalber empfehlen
        logger.warning(
            "[%s] Keine backtest_zusammenfassung.csv gefunden – "
            "Retraining wird empfohlen.",
            symbol,
        )
        return True

    try:
        df = pd.read_csv(zusammenfassung_pfad)
        # Symbol-Zeile suchen (case-insensitive)
        zeile = df[df["symbol"].str.upper() == symbol.upper()]
        if zeile.empty:
            logger.warning(
                "[%s] Kein Eintrag in backtest_zusammenfassung.csv – "
                "Retraining wird empfohlen.",
                symbol,
            )
            return True

        sharpe = float(zeile["sharpe_ratio"].iloc[0])
        if sharpe < sharpe_limit:
            logger.info(
                "[%s] Trigger ausgelöst: Sharpe=%.3f < Limit=%.2f → Retraining empfohlen",
                symbol,
                sharpe,
                sharpe_limit,
            )
            return True
        logger.info(
            "[%s] Kein Trigger: Sharpe=%.3f >= Limit=%.2f → Modell noch OK",
            symbol,
            sharpe,
            sharpe_limit,
        )
        return False

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[%s] Fehler beim Lesen der Zusammenfassung: %s", symbol, e)
        return True  # Im Zweifel trainieren


# ============================================================
# 4. Daten vorbereiten
# ============================================================


def daten_laden_und_aufteilen(
    symbol: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Lädt die gelabelten Daten und teilt sie in Train/Val auf.

    Wichtig: Test-Set (2023+) wird NICHT geladen – das bleibt heilig!
    Retraining verwendet nur Training- und Validierungsdaten.

    Args:
        symbol: Handelssymbol (z.B. "EURUSD")

    Returns:
        Tuple (X_train, y_train, X_val, y_val)

    Raises:
        FileNotFoundError: Wenn die Datendatei fehlt.
        ValueError: Wenn die Daten keine gültigen Spalten haben.
    """
    pfad = DATA_DIR / f"{symbol.upper()}_H1_labeled.csv"
    if not pfad.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {pfad}\n"
            f"Bitte zuerst features/labeling.py ausführen."
        )

    logger.info("[%s] Lade Daten: %s", symbol, pfad)
    df = pd.read_csv(pfad, index_col=0, parse_dates=True)

    # Label-Spalte prüfen
    if "label" not in df.columns:
        raise ValueError(f"[{symbol}] Spalte 'label' fehlt in {pfad}")

    # Features auswählen (gleiche Logik wie train_model.py)
    feature_spalten = [c for c in df.columns if c not in AUSSCHLUSS_SPALTEN]
    logger.info(
        "[%s] %s Features | %s Kerzen gesamt", symbol, len(feature_spalten), len(df)
    )

    # Zeitliche Aufteilung (KEINE Shuffle! Zeitreihen-Konvention)
    train_maske = df.index <= TRAIN_BIS
    val_maske = (df.index > TRAIN_BIS) & (df.index <= VAL_BIS)
    # test_maske = df.index > VAL_BIS  → NIEMALS verwenden!

    X = df[feature_spalten]
    y = df["label"]

    X_train, y_train = X[train_maske], y[train_maske]
    X_val, y_val = X[val_maske], y[val_maske]

    logger.info("[%s] Train: %s | Val: %s Kerzen", symbol, len(X_train), len(X_val))

    # NaN-Werte entfernen (Rolling-Features erzeugen NaN am Anfang)
    val_valid = ~(X_val.isnull().any(axis=1) | y_val.isnull())
    train_valid = ~(X_train.isnull().any(axis=1) | y_train.isnull())

    return (
        X_train[train_valid],
        y_train[train_valid],
        X_val[val_valid],
        y_val[val_valid],
    )


# ============================================================
# 5. LightGBM mit Optuna trainieren
# ============================================================


def neues_modell_trainieren(
    symbol: str,
    version: str,
    trials: int = OPTUNA_TRIALS,
) -> float:
    """Trainiert ein neues LightGBM-Modell mit Optuna-Optimierung.

    Das Modell wird NICHT gespeichert – nur F1 und temporäres Modell
    werden zurückgegeben. Das Deployment entscheidet modelle_vergleichen().

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Neue Versionsnummer (z.B. "v2")
        trials:  Anzahl Optuna-Trials (Standard: 30)

    Returns:
        F1-Macro-Score auf dem Validierungs-Set (Wert zwischen 0 und 1)
    """
    logger.info("[%s] Starte Retraining → Version %s", symbol, version)
    start = datetime.now()

    # Daten laden und aufteilen
    X_train, y_train, X_val, y_val = daten_laden_und_aufteilen(symbol)

    # ── Optuna Objective ─────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        """Optuna-Zielfunktion: maximiert F1-Macro auf Val-Set."""
        params = {
            "objective": "multiclass",
            "num_class": 3,  # Short=0, Neutral=1, Long=2
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "random_state": 42,
        }
        modell = lgb.LGBMClassifier(**params)
        modell.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        y_pred = modell.predict(X_val)
        return float(f1_score(y_val, y_pred, average="macro"))

    # ── Optuna-Optimierung starten ───────────────────────────
    logger.info("[%s] Optuna Retraining: %s Trials ...", symbol, trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    bestes_f1 = study.best_value
    beste_params = study.best_params
    dauer_sek = int((datetime.now() - start).total_seconds())

    logger.info(
        "[%s] Optuna abgeschlossen: F1=%.4f | Trials=%s | Dauer=%sm %ss",
        symbol,
        bestes_f1,
        trials,
        dauer_sek // 60,
        dauer_sek % 60,
    )

    # ── Bestes Modell final trainieren und temporär speichern ─
    final_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "random_state": 42,
        **beste_params,
    }
    finales_modell = lgb.LGBMClassifier(**final_params)
    finales_modell.fit(X_train, y_train)

    # Temporär speichern (Deployment-Entscheidung kommt danach)
    temp_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{version}_temp.pkl"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(finales_modell, temp_pfad)
    logger.info("[%s] Temp-Modell gespeichert: %s", symbol, temp_pfad)

    return bestes_f1


# ============================================================
# 6. Modelle vergleichen und deployen
# ============================================================


def modelle_vergleichen(
    f1_neu: float,
    f1_alt: Optional[float],
    toleranz: float = F1_TOLERANZ,
) -> bool:
    """Entscheidet ob das neue Modell deployed werden soll.

    Logik:
        - Kein altes Modell vorhanden → immer deployen
        - F1_neu >= F1_alt - Toleranz → deployen (Verbesserung oder gleich gut)
        - F1_neu < F1_alt - Toleranz → NICHT deployen (schlechter)

    Args:
        f1_neu:  F1-Score des neuen Modells
        f1_alt:  F1-Score des bisherigen Modells (None = kein altes Modell)
        toleranz: Toleranz-Puffer in F1-Punkten (Standard: 0.01 = 1%)

    Returns:
        True wenn das neue Modell deployed werden soll.
    """
    if f1_alt is None:
        # Erstes Modell für dieses Symbol → immer deployen
        logger.info("  Kein altes Modell vorhanden → Deployment empfohlen")
        return True

    grenzwert = f1_alt - toleranz
    if f1_neu >= grenzwert:
        logger.info(
            "  F1: %.4f >= %.4f (Alt=%.4f - Toleranz=%s) → Deployment empfohlen",
            f1_neu,
            grenzwert,
            f1_alt,
            toleranz,
        )
        return True

    logger.info(
        "  F1: %.4f < %.4f (Alt=%.4f - Toleranz=%s) → Kein Deployment",
        f1_neu,
        grenzwert,
        f1_alt,
        toleranz,
    )
    return False


def modell_deployen(symbol: str, version: str, f1: float) -> None:
    """Aktiviert das neue Modell durch Umbenennung des Temp-Files.

    Verschiebt das temporäre Modell auf den offiziellen Pfad und
    aktualisiert die F1-Historie.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Neue Versionsnummer (z.B. "v2")
        f1:      F1-Score des neuen Modells
    """
    temp_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{version}_temp.pkl"
    ziel_pfad = modell_pfad(symbol, version)

    if not temp_pfad.exists():
        logger.error("[%s] Temp-Modell nicht gefunden: %s", symbol, temp_pfad)
        return

    # Umbenennen (Deployment)
    temp_pfad.rename(ziel_pfad)
    logger.info("[%s] ✅ Modell deployed: %s", symbol, ziel_pfad)

    # F1-Historie speichern
    f1_history_speichern(symbol, version, f1)


def temp_modell_loeschen(symbol: str, version: str) -> None:
    """Löscht das temporäre Modell wenn kein Deployment stattfindet.

    Args:
        symbol:  Handelssymbol (z.B. "EURUSD")
        version: Versionsnummer des temporären Modells
    """
    temp_pfad = MODEL_DIR / f"lgbm_{symbol.lower()}_{version}_temp.pkl"
    if temp_pfad.exists():
        temp_pfad.unlink()
        logger.info("[%s] Temp-Modell gelöscht: %s", symbol, temp_pfad)


# ============================================================
# 7. Haupt-Retraining-Logik für ein Symbol
# ============================================================


def symbol_retraining(
    symbol: str,
    erzwingen: bool = False,
    sharpe_limit: float = SHARPE_GRENZWERT,
    trials: int = OPTUNA_TRIALS,
) -> dict:
    """Führt den vollständigen Retraining-Prozess für ein Symbol durch.

    Args:
        symbol:       Handelssymbol (z.B. "EURUSD")
        erzwingen:    True = immer trainieren (ignoriert Trigger-Prüfung)
        sharpe_limit: Sharpe-Schwelle für Trigger-Prüfung (Standard: 0.5)
        trials:       Anzahl Optuna-Trials (Standard: 30)

    Returns:
        Dict mit Ergebnis-Infos (deployed, f1_neu, f1_alt, version, grund)
    """
    logger.info("\n%s", "=" * 60)
    logger.info("  Retraining – %s", symbol)
    logger.info("%s", "=" * 60)

    ergebnis = {
        "symbol": symbol,
        "trainiert": False,
        "deployed": False,
        "f1_neu": None,
        "f1_alt": None,
        "version": None,
        "grund": "",
    }

    # ── Schritt 1: Trigger prüfen ───────────────────────────
    if not erzwingen:
        braucht_retraining = trigger_pruefen(symbol, sharpe_limit)
        if not braucht_retraining:
            ergebnis["grund"] = "Kein Trigger (Sharpe OK)"
            logger.info("[%s] Kein Retraining nötig.", symbol)
            return ergebnis
    else:
        logger.info("[%s] --erzwingen gesetzt → überspringe Trigger-Prüfung", symbol)

    # ── Schritt 2: Alte F1 laden ────────────────────────────
    f1_alt, version_alt = aktuellen_f1_laden(symbol)
    if f1_alt is not None:
        logger.info(
            "[%s] Altes Modell: Version=%s | F1=%.4f", symbol, version_alt, f1_alt
        )
    else:
        logger.info("[%s] Kein altes F1 in Historie → erstes Retraining", symbol)
    ergebnis["f1_alt"] = f1_alt

    # ── Schritt 3: Nächste Retraining-Versionsnummer bestimmen ─────────
    neue_version = naechste_versionsnummer(symbol)
    logger.info("[%s] Neue Versionsnummer: %s", symbol, neue_version)
    ergebnis["version"] = neue_version

    # ── Schritt 4: Neues Modell trainieren ──────────────────
    try:
        f1_neu = neues_modell_trainieren(symbol, neue_version, trials)
        ergebnis["trainiert"] = True
        ergebnis["f1_neu"] = f1_neu
    except FileNotFoundError as e:
        logger.error("[%s] Daten fehlen: %s", symbol, e)
        ergebnis["grund"] = "Fehler: Datei nicht gefunden"
        return ergebnis
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("[%s] Trainingsfehler: %s", symbol, e, exc_info=True)
        ergebnis["grund"] = f"Fehler: {e}"
        return ergebnis

    # ── Schritt 5: Vergleich und Deployment-Entscheidung ────
    logger.info("[%s] Vergleiche Modelle:", symbol)
    soll_deployen = modelle_vergleichen(f1_neu, f1_alt)

    if soll_deployen:
        modell_deployen(symbol, neue_version, f1_neu)
        ergebnis["deployed"] = True
        # f1_alt kann None sein (erstes Retraining) → separat formatieren
        f1_alt_str = f"{f1_alt:.4f}" if f1_alt is not None else "N/A"
        ergebnis["grund"] = (
            f"Deployed {neue_version}: F1={f1_neu:.4f} (Alt: {f1_alt_str})"
        )
        logger.info("[%s] ✅ DEPLOYED: %s (F1=%.4f)", symbol, neue_version, f1_neu)
    else:
        temp_modell_loeschen(symbol, neue_version)
        ergebnis["grund"] = (
            f"Kein Deployment: F1_neu={f1_neu:.4f} < F1_alt={f1_alt:.4f} "
            f"- Toleranz={F1_TOLERANZ}"
        )
        logger.info("[%s] ❌ NICHT deployed (altes Modell bleibt aktiv)", symbol)

    return ergebnis


# ============================================================
# 8. Hauptprogramm
# ============================================================


def main() -> None:
    """Hauptprogramm: Retraining für ein oder alle Symbole."""

    parser = argparse.ArgumentParser(
        description=(
            "MT5 ML-Trading – Monatliches Retraining mit Qualitätsprüfung\n"
            "Deployment nur wenn neues Modell nicht schlechter als altes."
        )
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help=(
            "Symbol für Retraining (Standard: EURUSD) oder 'alle'. "
            "Beispiel: --symbol USDCAD"
        ),
    )
    parser.add_argument(
        "--erzwingen",
        action="store_true",
        default=False,
        help=(
            "Retraining immer durchführen – ignoriert den Sharpe-Trigger. "
            "Nützlich für monatliche Pflicht-Läufe."
        ),
    )
    parser.add_argument(
        "--sharpe_limit",
        type=float,
        default=SHARPE_GRENZWERT,
        help=(
            f"Sharpe-Schwelle für automatischen Trigger (Standard: {SHARPE_GRENZWERT}). "
            f"Wenn Sharpe < Wert → Retraining empfohlen."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=OPTUNA_TRIALS,
        help=(
            f"Anzahl Optuna-Trials (Standard: {OPTUNA_TRIALS}). "
            f"Mehr Trials = bessere Hyperparameter, aber länger."
        ),
    )
    parser.add_argument(
        "--toleranz",
        type=float,
        default=F1_TOLERANZ,
        help=(
            f"F1-Toleranz für Deployment (Standard: {F1_TOLERANZ} = 1%%). "
            f"Neues Modell wird deployed wenn F1_neu >= F1_alt - Toleranz."
        ),
    )

    args = parser.parse_args()

    # Symbole bestimmen
    if args.symbol.lower() == "alle":
        ziel_symbole = SYMBOLE
    elif args.symbol.upper() in SYMBOLE:
        ziel_symbole = [args.symbol.upper()]
    else:
        print(f"Unbekanntes Symbol: '{args.symbol}'")
        print(f"Verfügbar: {', '.join(SYMBOLE)} oder 'alle'")
        return

    start_zeit = datetime.now()
    logger.info("%s", "=" * 60)
    logger.info("MT5 ML-Trading – Retraining-Pipeline")
    logger.info("Symbole:       %s", ", ".join(ziel_symbole))
    logger.info("Erzwingen:     %s", "Ja" if args.erzwingen else "Nein")
    logger.info("Sharpe-Limit:  %.2f", args.sharpe_limit)
    logger.info("Optuna-Trials: %s", args.trials)
    logger.info(
        "Version-Schema: %sN (entkoppelt von Datenversion vN)",
        RETRAINING_VERSION_PREFIX,
    )
    logger.info("F1-Toleranz:   %.2f (%s%%)", args.toleranz, f"{args.toleranz*100:.0f}")
    logger.info("HINWEIS: Test-Set (2023+) wird NIEMALS für Retraining verwendet!")
    logger.info("%s", "=" * 60)

    # Retraining für alle Ziel-Symbole
    alle_ergebnisse = []
    for symbol in ziel_symbole:
        ergebnis = symbol_retraining(
            symbol=symbol,
            erzwingen=args.erzwingen,
            sharpe_limit=args.sharpe_limit,
            trials=args.trials,
        )
        alle_ergebnisse.append(ergebnis)

    # Zusammenfassung ausgeben
    dauer_sek = int((datetime.now() - start_zeit).total_seconds())
    print("\n" + "=" * 65)
    print("RETRAINING ABGESCHLOSSEN – Zusammenfassung")
    print("=" * 65)
    print(
        f"{'Symbol':8} {'Trainiert':11} {'Deployed':10} "
        f"{'F1 Alt':8} {'F1 Neu':8} Ergebnis"
    )
    print("─" * 65)

    n_deployed = 0
    n_trainiert = 0
    for e in alle_ergebnisse:
        f1_alt_str = f"{e['f1_alt']:.4f}" if e["f1_alt"] is not None else "  N/A "
        f1_neu_str = f"{e['f1_neu']:.4f}" if e["f1_neu"] is not None else "  N/A "
        deployed_icon = "✅" if e["deployed"] else "─"
        trainiert_icon = "✅" if e["trainiert"] else "─"
        if e["deployed"]:
            n_deployed += 1
        if e["trainiert"]:
            n_trainiert += 1
        print(
            f"  {e['symbol']:8} {trainiert_icon:10} {deployed_icon:9} "
            f"{f1_alt_str:8} {f1_neu_str:8} {e['grund']}"
        )

    print("─" * 65)
    print(f"Trainiert: {n_trainiert}/{len(alle_ergebnisse)}")
    print(f"Deployed:  {n_deployed}/{len(alle_ergebnisse)}")
    print(f"Laufzeit:  {dauer_sek // 60}m {dauer_sek % 60}s")
    print("=" * 65)

    # Nächste Schritte
    if n_deployed > 0:
        print("\nNächster Schritt: Backtest mit dem neuen Modell ausführen:")
        for e in alle_ergebnisse:
            if e["deployed"]:
                print(
                    f"  python backtest/backtest.py "
                    f"--symbol {e['symbol']} "
                    f"--version v1 --model_version {e['version']} "
                    f"--schwelle 0.60 --regime_filter 1,2"
                )
    else:
        print("\nKeine neuen Modelle deployed – alte Modelle bleiben aktiv.")


if __name__ == "__main__":
    main()
