#!/usr/bin/env bash
set -euo pipefail

# Phase-7 Auto-Check Runner (Linux-Server)
# Führt zyklisch Sync-Verifikation + Daily-Dashboard aus.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

INTERVAL_MINUTES="${1:-30}"
LOG_DIR_ARG="${2:-logs/paper_test128}"
SYMBOLS_ARG="${3:-USDCAD,USDJPY}"
TIMEFRAME_ARG="${4:-M5_TWO_STAGE}"
MAX_AGE_MINUTES="${5:-10}"

REPORTS_DIR="${ROOT_DIR}/reports"
CHECK_LOG="${REPORTS_DIR}/phase7_autocheck.log"
LATEST_STATUS="${REPORTS_DIR}/phase7_autocheck_latest.txt"
LOCK_FILE="${REPORTS_DIR}/.phase7_autocheck.lock"

mkdir -p "${REPORTS_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FEHLER] Python-Interpreter nicht gefunden: ${PYTHON_BIN}"
  exit 1
fi

if ! [[ "${INTERVAL_MINUTES}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_MINUTES}" -lt 1 ]]; then
  echo "[FEHLER] Intervall muss eine ganze Zahl >= 1 sein (aktuell: ${INTERVAL_MINUTES})"
  exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[INFO] Auto-Check läuft bereits (Lock aktiv: ${LOCK_FILE})."
  exit 0
fi

run_once() {
  local ts verify_rc dashboard_rc overall_rc
  ts="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  echo "" | tee -a "${CHECK_LOG}"
  echo "================================================================" | tee -a "${CHECK_LOG}"
  echo "[${ts}] PHASE7 AUTO-CHECK" | tee -a "${CHECK_LOG}"
  echo "LogDir=${LOG_DIR_ARG} | Symbols=${SYMBOLS_ARG} | MaxAge=${MAX_AGE_MINUTES}min | TF=${TIMEFRAME_ARG}" | tee -a "${CHECK_LOG}"
  echo "================================================================" | tee -a "${CHECK_LOG}"

  set +e
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/verify_live_log_sync.py" \
    --log_dir "${LOG_DIR_ARG}" \
    --symbols "${SYMBOLS_ARG}" \
    --max_age_minutes "${MAX_AGE_MINUTES}" \
    --check_watchdog 2>&1 | tee -a "${CHECK_LOG}"
  verify_rc=${PIPESTATUS[0]}

  "${PYTHON_BIN}" "${ROOT_DIR}/reports/daily_phase7_dashboard.py" \
    --log_dir "${LOG_DIR_ARG}" \
    --hours 24 \
    --timeframe "${TIMEFRAME_ARG}" 2>&1 | tee -a "${CHECK_LOG}"
  dashboard_rc=${PIPESTATUS[0]}
  set -e

  overall_rc=0
  if [[ "${verify_rc}" -ne 0 || "${dashboard_rc}" -ne 0 ]]; then
    overall_rc=1
  fi

  {
    echo "timestamp_utc=${ts}"
    echo "verify_exit_code=${verify_rc}"
    echo "dashboard_exit_code=${dashboard_rc}"
    echo "overall_exit_code=${overall_rc}"
  } > "${LATEST_STATUS}"

  if [[ "${overall_rc}" -eq 0 ]]; then
    echo "[${ts}] Ergebnis: OK" | tee -a "${CHECK_LOG}"
  else
    echo "[${ts}] Ergebnis: NICHT_OK (verify=${verify_rc}, dashboard=${dashboard_rc})" | tee -a "${CHECK_LOG}"
  fi
}

echo "[INFO] Starte Phase-7 Auto-Check: alle ${INTERVAL_MINUTES} Minute(n)." | tee -a "${CHECK_LOG}"
echo "[INFO] Stoppen mit: pkill -f run_phase7_autocheck.sh" | tee -a "${CHECK_LOG}"

while true; do
  run_once
  sleep "$((INTERVAL_MINUTES * 60))"
done
