#!/usr/bin/env bash
set -euo pipefail

# M15-Pipeline (Linux-Server)
# Voraussetzung: M15-Rohdaten wurden zuvor auf dem Windows-Laptop mit data_loader.py erzeugt
# und nach data/SYMBOL_M15.csv auf den Linux-Server kopiert.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

SYMBOLS=("USDCAD" "USDJPY")
VERSION="v1"
TRIALS=50
SCHWELLE=0.60
REGIME_FILTER="1,2"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FEHLER] Python-Interpreter nicht gefunden: ${PYTHON_BIN}"
  echo "Bitte zuerst die virtuelle Umgebung einrichten."
  exit 1
fi

for sym in "${SYMBOLS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/data/${sym}_M15.csv" ]]; then
    echo "[FEHLER] Fehlende M15-Rohdaten: data/${sym}_M15.csv"
    echo "Bitte auf Windows ausführen: data_loader.py --timeframe M15"
    exit 1
  fi
done

echo "[1/6] Feature Engineering (M15)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/features/feature_engineering.py" --symbol "${sym}" --timeframe M15
done

echo "[2/6] Labeling (M15)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/features/labeling.py" --symbol "${sym}" --version "${VERSION}" --timeframe M15
done

echo "[3/6] Training (M15, Optuna ${TRIALS} Trials)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/train_model.py" --symbol "${sym}" --version "${VERSION}" --timeframe M15 --trials "${TRIALS}"
done

echo "[4/6] Walk-Forward (M15)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/walk_forward.py" --symbol "${sym}" --version "${VERSION}" --timeframe M15
done

echo "[5/6] Backtest (M15)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/backtest/backtest.py" \
    --symbol "${sym}" \
    --version "${VERSION}" \
    --timeframe M15 \
    --schwelle "${SCHWELLE}" \
    --regime_filter "${REGIME_FILTER}"
done

echo "[6/6] KPI-Report (M15)"
"${PYTHON_BIN}" "${ROOT_DIR}/reports/weekly_kpi_report.py" --timeframe M15 --tage 7

echo "Fertig. Wichtige Artefakte:"
echo "- backtest/USDCAD_M15_trades.csv"
echo "- backtest/USDJPY_M15_trades.csv"
echo "- reports/weekly_kpi_report_M15.md"
