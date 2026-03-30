"""
phase7_monitoring_guard.py – L1/L2/L3 Guardrail-Check für Phase 7.

Zweck:
    - L1: GO-Wochen-Streak gegen Ziel (12 Wochen) prüfen
    - L2: Retraining-Fälligkeit (monatlich) bewerten
    - L3: Drift-Hinweis über Prob-Verteilungs-Shift aus Live-Signalen melden

Hinweis:
    Dieses Skript ist ein operativer Guard und ersetzt keine tiefe Modellanalyse.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

GO_TARGET_WEEKS = 12
RETRAIN_DAYS = 30
DRIFT_ALERT_THRESHOLD = 0.15


def _latest_go_streak(history: pd.DataFrame) -> int:
    if "overall_status" not in history.columns:
        return 0
    streak = 0
    for status in history["overall_status"].astype(str).iloc[::-1]:
        if status.upper() == "GO":
            streak += 1
        else:
            break
    return streak


def _latest_week_ts(history: pd.DataFrame) -> Optional[pd.Timestamp]:
    for col in ["report_date", "date", "created_at"]:
        if col in history.columns:
            dt = pd.to_datetime(history[col], errors="coerce")
            if dt.notna().any():
                return dt.max()
    return None


def _latest_model_mtime(symbol: str) -> Optional[datetime]:
    candidates = sorted(
        MODELS_DIR.glob(f"lgbm_{symbol.lower()}_*.pkl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        return None
    return datetime.fromtimestamp(candidates[-1].stat().st_mtime)


def _drift_score_from_signals(symbol: str) -> Optional[float]:
    path = LOG_DIR / f"{symbol}_signals.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "prob" not in df.columns or "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "prob"]).sort_values("time")
    if len(df) < 300:
        return None

    recent = df.tail(150)["prob"].astype(float).to_numpy()
    base = df.iloc[-450:-150]["prob"].astype(float).to_numpy()
    if len(base) < 150:
        return None

    # Einfache Drift-Näherung: Mittelwert-Shift + STD-Normierung
    mean_shift = float(abs(np.mean(recent) - np.mean(base)))
    std_base = float(np.std(base)) if float(np.std(base)) > 1e-9 else 1e-9
    return mean_shift / std_base


def main() -> None:
    history_path = REPORTS_DIR / "weekly_kpi_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Fehlt: {history_path}")

    history = pd.read_csv(history_path)

    go_streak = _latest_go_streak(history)
    last_week = _latest_week_ts(history)

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "go_streak_weeks": go_streak,
        "go_target_weeks": GO_TARGET_WEEKS,
        "l1_pass": go_streak >= GO_TARGET_WEEKS,
        "last_kpi_week": str(last_week) if last_week is not None else "n/a",
        "symbols": {},
    }

    for symbol in ["USDCAD", "USDJPY"]:
        model_mtime = _latest_model_mtime(symbol)
        retrain_due = True
        days_since_model = None
        if model_mtime is not None:
            days_since_model = (datetime.now() - model_mtime).days
            retrain_due = days_since_model >= RETRAIN_DAYS

        drift_score = _drift_score_from_signals(symbol)
        drift_alert = drift_score is not None and drift_score >= DRIFT_ALERT_THRESHOLD

        report["symbols"][symbol] = {
            "last_model_update": (
                model_mtime.strftime("%Y-%m-%d %H:%M:%S") if model_mtime else "n/a"
            ),
            "days_since_model": days_since_model,
            "l2_retrain_due": retrain_due,
            "drift_score": drift_score,
            "l3_drift_alert": drift_alert,
        }

    out = REPORTS_DIR / "phase7_monitoring_guard_latest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Guard gespeichert: %s", out)
    logger.info("L1 GO-Streak: %s/%s", go_streak, GO_TARGET_WEEKS)


if __name__ == "__main__":
    main()
