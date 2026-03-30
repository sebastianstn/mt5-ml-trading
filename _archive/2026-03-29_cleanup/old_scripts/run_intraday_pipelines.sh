#!/usr/bin/env bash
set -euo pipefail

# Intraday-Pipeline Runner für M15, M30, M60 (Linux-Server)
# Voraussetzung: Rohdaten wurden vom Windows-Laptop nach data/SYMBOL_<TF>.csv kopiert.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

SYMBOLS=("USDCAD" "USDJPY")
TIMEFRAMES=("M15" "M30" "M60")
VERSION="v1"
TRIALS=50
SCHWELLE=0.60
REGIME_FILTER="1,2"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FEHLER] Python-Interpreter nicht gefunden: ${PYTHON_BIN}"
  echo "Bitte zuerst die virtuelle Umgebung einrichten."
  exit 1
fi

for tf in "${TIMEFRAMES[@]}"; do
  for sym in "${SYMBOLS[@]}"; do
    if [[ ! -f "${ROOT_DIR}/data/${sym}_${tf}.csv" ]]; then
      echo "[FEHLER] Fehlende Rohdaten: data/${sym}_${tf}.csv"
      echo "Bitte zuerst auf Windows exportieren und per scp übertragen."
      exit 1
    fi
  done
done

for tf in "${TIMEFRAMES[@]}"; do
  echo ""
  echo "=============================================================="
  echo "Starte Pipeline für Timeframe: ${tf}"
  echo "=============================================================="

  echo "[1/6] Feature Engineering (${tf})"
  for sym in "${SYMBOLS[@]}"; do
    "${PYTHON_BIN}" "${ROOT_DIR}/features/feature_engineering.py" --symbol "${sym}" --timeframe "${tf}"
  done

  echo "[2/6] Labeling (${tf})"
  for sym in "${SYMBOLS[@]}"; do
    "${PYTHON_BIN}" "${ROOT_DIR}/features/labeling.py" --symbol "${sym}" --version "${VERSION}" --timeframe "${tf}"
  done

  echo "[3/6] Training (${tf}, Optuna ${TRIALS} Trials)"
  for sym in "${SYMBOLS[@]}"; do
    "${PYTHON_BIN}" "${ROOT_DIR}/train_model.py" --symbol "${sym}" --version "${VERSION}" --timeframe "${tf}" --trials "${TRIALS}"
  done

  echo "[4/6] Walk-Forward (${tf})"
  for sym in "${SYMBOLS[@]}"; do
    "${PYTHON_BIN}" "${ROOT_DIR}/walk_forward.py" --symbol "${sym}" --version "${VERSION}" --timeframe "${tf}"
  done

  echo "[5/6] Backtest (${tf})"
  for sym in "${SYMBOLS[@]}"; do
    "${PYTHON_BIN}" "${ROOT_DIR}/backtest/backtest.py" \
      --symbol "${sym}" \
      --version "${VERSION}" \
      --timeframe "${tf}" \
      --schwelle "${SCHWELLE}" \
      --regime_filter "${REGIME_FILTER}"
  done

  echo "[6/6] KPI-Report (${tf})"
  "${PYTHON_BIN}" "${ROOT_DIR}/reports/weekly_kpi_report.py" --timeframe "${tf}" --tage 7

done

echo ""
echo "Fertig. Wichtige Artefakte pro Timeframe:"
echo "- backtest/USDCAD_<TF>_trades.csv"
echo "- backtest/USDJPY_<TF>_trades.csv"
echo "- reports/weekly_kpi_report_<TF>.md"
