#!/usr/bin/env bash
set -euo pipefail

# M5-Datenkette für Option 1 (HTF=H1 / LTF=M5)
# Schritte (Linux-Server):
#   1) Prüft Rohdaten data/SYMBOL_M5.csv
#   2) Berechnet M5-Features
#   3) Labelt M5-Daten im ATR-Modus (v4)
#
# Hinweis:
# - MT5-Rohdaten müssen vorher auf Windows mit data_loader.py exportiert
#   und nach data/ auf den Linux-Server kopiert werden.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

SYMBOLS=("USDCAD" "USDJPY")
VERSION="v4"
TIMEFRAME="M5"
HORIZON=5
ATR_FAKTOR=1.5

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FEHLER] Python-Interpreter nicht gefunden: ${PYTHON_BIN}"
  echo "Bitte zuerst virtuelle Umgebung einrichten."
  exit 1
fi

echo "[1/4] Prüfe M5-Rohdaten"
for sym in "${SYMBOLS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/data/${sym}_${TIMEFRAME}.csv" ]]; then
    echo "[FEHLER] Fehlende Rohdaten: data/${sym}_${TIMEFRAME}.csv"
    echo "Bitte auf Windows ausführen: data_loader.py --symbol ${sym} --timeframe ${TIMEFRAME} --bars 30000"
    exit 1
  fi
done

echo "[2/4] Feature Engineering (${TIMEFRAME})"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/features/feature_engineering.py" \
    --symbol "${sym}" \
    --timeframe "${TIMEFRAME}"
done

echo "[3/4] Labeling (${TIMEFRAME}, ${VERSION}, Modus=ATR)"
for sym in "${SYMBOLS[@]}"; do
  "${PYTHON_BIN}" "${ROOT_DIR}/features/labeling.py" \
    --symbol "${sym}" \
    --timeframe "${TIMEFRAME}" \
    --version "${VERSION}" \
    --modus atr \
    --atr_faktor "${ATR_FAKTOR}" \
    --horizon "${HORIZON}"
done

echo "[4/4] Prüfe erzeugte labeled-Dateien"
for sym in "${SYMBOLS[@]}"; do
  FILE="${ROOT_DIR}/data/${sym}_${TIMEFRAME}_labeled_${VERSION}.csv"
  if [[ ! -f "${FILE}" ]]; then
    echo "[FEHLER] Erwartete Datei fehlt: ${FILE}"
    exit 1
  fi
  echo "✓ ${sym}_${TIMEFRAME}_labeled_${VERSION}.csv"
done

echo ""
echo "Fertig. Nächster Schritt:"
echo "  bash run_two_stage_pipeline.sh"
echo ""
echo "Erwartete Two-Stage-Artefakte in models/:"
echo "  - lgbm_htf_bias_<symbol>_H1_${VERSION}.pkl"
echo "  - lgbm_ltf_entry_<symbol>_${TIMEFRAME}_${VERSION}.pkl"
echo "  - two_stage_<symbol>_${TIMEFRAME}_${VERSION}.json"
