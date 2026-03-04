#!/usr/bin/env bash
set -euo pipefail

# Zwei-Stufen-Pipeline (Option 1)
# Stufe 1: HTF-Bias auf H1
# Stufe 2: LTF-Entry auf M5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

SYMBOLS=("USDCAD" "USDJPY")
VERSION="v4"
LTF_TIMEFRAME="M5"

if [[ "${VERSION}" == "v1" ]]; then
  H1_SUFFIX=""
  LTF_SUFFIX=""
else
  H1_SUFFIX="_${VERSION}"
  LTF_SUFFIX="_${VERSION}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FEHLER] Python-Interpreter nicht gefunden: ${PYTHON_BIN}"
  echo "Bitte zuerst virtuelle Umgebung auf dem Linux-Server einrichten."
  exit 1
fi

echo "[1/2] Prüfe Eingabedaten"
for sym in "${SYMBOLS[@]}"; do
  if [[ ! -f "${ROOT_DIR}/data/${sym}_H1_labeled${H1_SUFFIX}.csv" ]]; then
    echo "[FEHLER] Fehlende Datei: data/${sym}_H1_labeled${H1_SUFFIX}.csv"
    exit 1
  fi
  if [[ ! -f "${ROOT_DIR}/data/${sym}_${LTF_TIMEFRAME}_labeled${LTF_SUFFIX}.csv" ]]; then
    echo "[FEHLER] Fehlende Datei: data/${sym}_${LTF_TIMEFRAME}_labeled${LTF_SUFFIX}.csv"
    echo "Bitte zuerst Feature-Engineering + Labeling für ${LTF_TIMEFRAME} ausführen."
    echo "Tipp: bash run_m5_pipeline.sh"
    exit 1
  fi
done

echo "[2/2] Trainiere Zwei-Stufen-Modelle"
for sym in "${SYMBOLS[@]}"; do
  echo "-> ${sym} | HTF=H1 | LTF=${LTF_TIMEFRAME} | Version=${VERSION}"
  "${PYTHON_BIN}" "${ROOT_DIR}/train_two_stage.py" \
    --symbol "${sym}" \
    --ltf_timeframe "${LTF_TIMEFRAME}" \
    --version "${VERSION}"
done

echo "Fertig. Gespeicherte Artefakte in models/:"
echo "- lgbm_htf_bias_<symbol>_H1_<version>.pkl"
echo "- lgbm_ltf_entry_<symbol>_<ltf>_<version>.pkl"
echo "- two_stage_<symbol>_<ltf>_<version>.json"
