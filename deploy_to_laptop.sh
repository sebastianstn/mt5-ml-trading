#!/bin/bash
# =============================================================================
# deploy_to_laptop.sh – Modelle + Live-Trading-Skript auf Windows Laptop kopieren
#
# Ausführen auf dem Linux-Server:
#     bash deploy_to_laptop.sh
#
# Voraussetzung: SSH-Verbindung zum Windows-Laptop muss funktionieren.
#     Auf Windows: OpenSSH-Server installieren (Einstellungen → Apps → Optionale Features)
#     Test: ssh BENUTZER@LAPTOP_IP "echo OK"
# =============================================================================

set -e  # Bei Fehler sofort stoppen

# ------------------------------------------------------------
# Konfiguration – HIER ANPASSEN
# ------------------------------------------------------------
LAPTOP_BENUTZER="sebastian setnescu"          # Windows-Benutzername (whoami: acemagic\sebastian setnescu)
LAPTOP_IP="192.168.1.19"           # IP-Adresse des Laptops (z.B. 192.168.1.x)
LAPTOP_ZIELORDNER="C:/Users/sebastian setnescu/mt5_trading"  # Absoluter Pfad (für PowerShell mkdir)
LAPTOP_ZIELORDNER_SFTP="mt5_trading"          # Relativer Pfad für sftp (startet im Home-Dir des Users)

# ------------------------------------------------------------
# SSH ControlMaster – einmal Passwort, dann Tunnel wiederverwenden
# ------------------------------------------------------------
SSH_CONTROL_DIR=$(mktemp -d)
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/ssh-%r@%h:%p"
SSH_OPTS="-o ControlPath=${SSH_CONTROL_PATH}"

# Aufräum-Funktion: SSH-Tunnel schließen wenn Skript beendet wird
cleanup() {
    ssh -o ControlPath="${SSH_CONTROL_PATH}" -O exit "${LAPTOP_BENUTZER}@${LAPTOP_IP}" 2>/dev/null || true
    rm -rf "${SSH_CONTROL_DIR}"
}
trap cleanup EXIT

# Linux-Server Pfade
SERVER_BASIS="/mnt/1Tb-Data/XGBoost-LightGBM"
MODELL_ORDNER="${SERVER_BASIS}/models"
LIVE_SKRIPT="${SERVER_BASIS}/live/live_trader.py"
MT5_DASHBOARD_SKRIPT="${SERVER_BASIS}/live/mt5/LiveSignalDashboard.mq5"
REQUIREMENTS="${SERVER_BASIS}/requirements-laptop.txt"

# ------------------------------------------------------------
# Hilfsfunktion: Datei via SFTP übertragen (nur Fehler anzeigen)
# ------------------------------------------------------------
sftp_put() {
    local local_path="$1"
    local remote_path="$2"
    local tmp_out

    tmp_out=$(mktemp)

    # ControlPath nutzt den bestehenden SSH-Tunnel → kein erneutes Passwort nötig
    if sftp -q -o ControlPath="${SSH_CONTROL_PATH}" -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" >"${tmp_out}" 2>&1 <<SFTP_END
put "${local_path}" "${remote_path}"
SFTP_END
    then
        rm -f "${tmp_out}"
        return 0
    fi

    echo "        ❌ SFTP-Transfer fehlgeschlagen: ${local_path} -> ${remote_path}"
    sed 's/^/        /' "${tmp_out}"
    rm -f "${tmp_out}"
    return 1
}

# Welche Modelle übertragen? (nur v4 – v5 wurde gestoppt)
MODELLE=(
    "lgbm_usdcad_v4.pkl"                    # USDCAD H1 – Single-Stage Fallback (v4)
    "lgbm_htf_bias_usdcad_H1_v4.pkl"        # USDCAD HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdcad_M5_v4.pkl"       # USDCAD LTF-Entry (Two-Stage)
    "two_stage_usdcad_M5_v4.json"            # USDCAD Two-Stage Metadaten (Feature-Listen)
    "lgbm_usdjpy_v4.pkl"                    # USDJPY H1 – Single-Stage Fallback (v4)
    "lgbm_htf_bias_usdjpy_H1_v4.pkl"        # USDJPY HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdjpy_M5_v4.pkl"       # USDJPY LTF-Entry (Two-Stage)
    "two_stage_usdjpy_M5_v4.json"            # USDJPY Two-Stage Metadaten (Feature-Listen)
)

# Optionale Modelle (nur Info bei Nichtvorhandensein, kein Warnsignal)
OPTIONALE_MODELLE=(
    "lgbm_usdchf_H4_v1.pkl"                 # USDCHF H4 – optionaler 3. Kandidat
)

# ------------------------------------------------------------
# Verbindungstest
# ------------------------------------------------------------
echo "=================================================="
echo "  MT5 ML-Trading Deploy-Skript"
echo "  Ziel: ${LAPTOP_BENUTZER}@${LAPTOP_IP}"
echo "=================================================="
echo ""

echo "[ 1/4 ] Teste SSH-Verbindung zum Laptop (Passwort wird nur einmal abgefragt)..."
# ControlMaster-Tunnel aufbauen: bleibt für alle weiteren SSH/SFTP-Befehle offen
if ssh -o ConnectTimeout=10 -o ControlMaster=yes -o ControlPath="${SSH_CONTROL_PATH}" -o ControlPersist=300 "${LAPTOP_BENUTZER}@${LAPTOP_IP}" "echo 'SSH OK'"; then
    echo "        ✅ SSH-Verbindung erfolgreich (Tunnel aktiv)"
else
    echo "        ❌ SSH-Verbindung fehlgeschlagen!"
    echo ""
    echo "  Lösung:"
    echo "  1. OpenSSH-Server auf Windows aktivieren:"
    echo "     Einstellungen → System → Optionale Features → OpenSSH-Server"
    echo "  2. Windows Firewall: Port 22 freigeben"
    echo "  3. IP-Adresse prüfen: ipconfig auf Windows"
    echo "  4. Verbindungstest: ssh ${LAPTOP_BENUTZER}@${LAPTOP_IP}"
    exit 1
fi

# ------------------------------------------------------------
# Zielordner auf Laptop erstellen
# ------------------------------------------------------------
echo ""
echo "[ 2/4 ] Erstelle Ordnerstruktur auf Laptop..."
# Windows braucht PowerShell statt mkdir -p (cmd.exe kennt -p nicht)
ssh ${SSH_OPTS} "${LAPTOP_BENUTZER}@${LAPTOP_IP}" "powershell -Command \"New-Item -ItemType Directory -Force -Path '${LAPTOP_ZIELORDNER}/live','${LAPTOP_ZIELORDNER}/live/mt5','${LAPTOP_ZIELORDNER}/models','${LAPTOP_ZIELORDNER}/logs','${LAPTOP_ZIELORDNER}/logs/paper_test128','${LAPTOP_ZIELORDNER}/scripts','${LAPTOP_ZIELORDNER}/tests' | Out-Null; Write-Output 'Ordner erstellt'\""
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/live/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/live/mt5/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/models/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/logs/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/logs/paper_test128/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/scripts/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/tests/"

# ------------------------------------------------------------
# Modelle übertragen
# ------------------------------------------------------------
echo ""
echo "[ 3/4 ] Übertrage Modelle..."
for MODELL in "${MODELLE[@]}"; do
    PFAD="${MODELL_ORDNER}/${MODELL}"
    if [ -f "${PFAD}" ]; then
        GROESSE=$(du -sh "${PFAD}" | cut -f1)
        echo "        Übertrage ${MODELL} (${GROESSE})..."
        # sftp statt scp: SFTP-Protokoll umgeht Shell-Quoting-Probleme bei Windows-Pfaden mit Leerzeichen
        # Relativer SFTP-Pfad: Windows OpenSSH startet im User-Home C:\Users\sebastian setnescu\
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/models/${MODELL}"
        echo "        ✅ ${MODELL}"
    else
        echo "        ⚠️  ${MODELL} nicht gefunden – übersprungen"
    fi
done

# Optionale Modelle nur bei Verfügbarkeit übertragen (kein Warnsignal)
for MODELL in "${OPTIONALE_MODELLE[@]}"; do
    PFAD="${MODELL_ORDNER}/${MODELL}"
    if [ -f "${PFAD}" ]; then
        GROESSE=$(du -sh "${PFAD}" | cut -f1)
        echo "        Übertrage optionales Modell ${MODELL} (${GROESSE})..."
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/models/${MODELL}"
        echo "        ✅ optional: ${MODELL}"
    else
        echo "        ℹ️  Optionales Modell ${MODELL} nicht vorhanden – übersprungen"
    fi
done

# ------------------------------------------------------------
# Live-Skript + Requirements übertragen
# ------------------------------------------------------------
echo ""
echo "[ 4/4 ] Übertrage Skripte..."

# Alle Live-Trading-Module übertragen (nach Refactoring in Schritt 4)
LIVE_MODULE=(
    "live_trader.py"
    "config.py"
    "indicators.py"
    "feature_builder.py"
    "external_api.py"
    "signal_engine.py"
    "mt5_connector.py"
    "trade_logger.py"
    "paper_trading.py"
    "risk_manager.py"
    "db_manager.py"
    "__init__.py"
    "two_stage_signal.py"
)
for MODUL in "${LIVE_MODULE[@]}"; do
    PFAD="${SERVER_BASIS}/live/${MODUL}"
    if [ -f "${PFAD}" ]; then
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/live/${MODUL}"
        echo "        ✅ live/${MODUL}"
    else
        echo "        ⚠️  live/${MODUL} nicht gefunden – übersprungen"
    fi
done

# MT5 Dashboard-Indicator (MQL5)
if [ -f "${MT5_DASHBOARD_SKRIPT}" ]; then
    sftp_put "${MT5_DASHBOARD_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/LiveSignalDashboard.mq5"
    echo "        ✅ live/mt5/LiveSignalDashboard.mq5"
else
    echo "        ⚠️  LiveSignalDashboard.mq5 nicht gefunden – übersprungen"
fi

# MT5 Sync-Skripte (PowerShell, für Scheduled Tasks)
MT5_SYNC_SKRIPT="${SERVER_BASIS}/live/mt5/sync_live_logs_to_mt5_common.ps1"
MT5_INSTALL_TASK="${SERVER_BASIS}/live/mt5/install_sync_task.ps1"
if [ -f "${MT5_SYNC_SKRIPT}" ]; then
    sftp_put "${MT5_SYNC_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/sync_live_logs_to_mt5_common.ps1"
    echo "        ✅ live/mt5/sync_live_logs_to_mt5_common.ps1"
fi
if [ -f "${MT5_INSTALL_TASK}" ]; then
    sftp_put "${MT5_INSTALL_TASK}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/install_sync_task.ps1"
    echo "        ✅ live/mt5/install_sync_task.ps1"
fi

# Tests übertragen (Windows-kompatible Tests für live/ Module)
TEST_DATEIEN=("conftest.py" "test_live_trader_mt5_sync.py")
for TEST in "${TEST_DATEIEN[@]}"; do
    PFAD="${SERVER_BASIS}/tests/${TEST}"
    if [ -f "${PFAD}" ]; then
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/tests/${TEST}"
        echo "        ✅ tests/${TEST}"
    fi
done

sftp_put "${REQUIREMENTS}" "${LAPTOP_ZIELORDNER_SFTP}/requirements-laptop.txt"
echo "        ✅ requirements-laptop.txt"

# Batch-Skript zum Starten (Demo-Live mit PnL-Tracking)
sftp_put "${SERVER_BASIS}/start_paper_trading.bat" "${LAPTOP_ZIELORDNER_SFTP}/start_paper_trading.bat"
echo "        ✅ start_paper_trading.bat (Demo-Live, PnL-Tracking aktiv)"

# Batch-Skript für neue Top-Testphase (Paper, Two-Stage v4)
TOPCONFIG_START_SKRIPT="${SERVER_BASIS}/start_testphase_topconfig.bat"
if [ -f "${TOPCONFIG_START_SKRIPT}" ]; then
    sftp_put "${TOPCONFIG_START_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/start_testphase_topconfig.bat"
    echo "        ✅ start_testphase_topconfig.bat (Top-Konfiguration, Paper)"
fi

# Batch-Skript für Test 128 (Paper, best-balanced aus 50er-Feintuning)
TEST128_START_SKRIPT="${SERVER_BASIS}/start_paper_trading_test128.bat"
if [ -f "${TEST128_START_SKRIPT}" ]; then
    sftp_put "${TEST128_START_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/start_paper_trading_test128.bat"
    echo "        ✅ start_paper_trading_test128.bat (Test 128, Paper)"
fi

# Batch-Skript für Test-128-Log-Sync (Laptop -> Linux-Server)
TEST128_SYNC_SKRIPT="${SERVER_BASIS}/register_test128_log_sync_to_server.bat"
if [ -f "${TEST128_SYNC_SKRIPT}" ]; then
    sftp_put "${TEST128_SYNC_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/register_test128_log_sync_to_server.bat"
    echo "        ✅ register_test128_log_sync_to_server.bat (Test 128 Logs -> Server)"
fi

# Batch-Skript zum sauberen Stoppen aller Trader
STOP_ALL_SKRIPT="${SERVER_BASIS}/stop_all_traders.bat"
if [ -f "${STOP_ALL_SKRIPT}" ]; then
    sftp_put "${STOP_ALL_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/stop_all_traders.bat"
    echo "        ✅ stop_all_traders.bat"
fi

# Windows Sync-Task Skripte für Logs (Laptop -> Linux)
WIN_SYNC_SKRIPT="${SERVER_BASIS}/scripts/windows_sync_live_logs.ps1"
WIN_TASK_REGISTER_SKRIPT="${SERVER_BASIS}/scripts/windows_register_live_log_sync_task.ps1"
WIN_TASK_TEMPLATE="${SERVER_BASIS}/scripts/windows_task_live_log_sync.xml.template"
WIN_WATCHDOG_SKRIPT="${SERVER_BASIS}/scripts/windows_live_log_watchdog.ps1"

if [ -f "${WIN_SYNC_SKRIPT}" ]; then
    sftp_put "${WIN_SYNC_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_sync_live_logs.ps1"
    echo "        ✅ scripts/windows_sync_live_logs.ps1"
fi
if [ -f "${WIN_WATCHDOG_SKRIPT}" ]; then
    sftp_put "${WIN_WATCHDOG_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_live_log_watchdog.ps1"
    echo "        ✅ scripts/windows_live_log_watchdog.ps1"
fi
if [ -f "${WIN_TASK_REGISTER_SKRIPT}" ]; then
    sftp_put "${WIN_TASK_REGISTER_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_register_live_log_sync_task.ps1"
    echo "        ✅ scripts/windows_register_live_log_sync_task.ps1"
fi
if [ -f "${WIN_TASK_TEMPLATE}" ]; then
    sftp_put "${WIN_TASK_TEMPLATE}" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_task_live_log_sync.xml.template"
    echo "        ✅ scripts/windows_task_live_log_sync.xml.template"
fi

# ------------------------------------------------------------
# Abschlussmeldung
# ------------------------------------------------------------
echo ""
echo "=================================================="
echo "  ✅ Deploy abgeschlossen!"
echo "=================================================="
echo ""
echo "  Nächste Schritte auf dem Windows-Laptop:"
echo ""
echo "  1. PowerShell als Administrator öffnen"
echo "  2. In Projektordner wechseln:"
echo "       cd ${LAPTOP_ZIELORDNER}"
echo ""
echo "  3. Virtuelle Umgebung erstellen + aktivieren:"
echo "       python -m venv venv"
echo "       venv\\Scripts\\activate"
echo ""
echo "  4. Abhängigkeiten installieren:"
echo "       pip install -r requirements-laptop.txt"
echo ""
echo "  5. MT5 Terminal öffnen und angemeldet lassen"
echo ""
echo "  6. Empfohlener Standard-Start: Test 128 (Paper)"
echo "       Doppelklick auf: start_paper_trading_test128.bat"
echo ""
echo "     Startet USDCAD + USDJPY im Test-128-Setup"
echo "     Two-Stage v4 | Paper-Modus | Log-Ordner: logs/paper_test128"
echo "     MT5-Zugangsdaten sind in start_paper_trading_test128.bat bereits hinterlegt"
echo ""
echo "  7. Direkt danach: Test-128-Logs automatisch zum Linux-Server syncen"
echo "       Doppelklick auf: register_test128_log_sync_to_server.bat"
echo "       (nutzt automatisch scripts/windows_live_log_watchdog.ps1)"
echo ""
echo "  8. Optional / Alternativen"
echo ""
echo "       Option A) Alle Trader sauber stoppen (vor Neustart empfohlen):"
echo "                 Doppelklick auf: stop_all_traders.bat"
echo ""
echo "       Option B) Neue Top-Testphase (Paper) starten:"
echo "                 Doppelklick auf: start_testphase_topconfig.bat"
echo ""
echo "       Option C) Demo-Live-Trading starten (PnL-Tracking aktiv):"
echo "                 Doppelklick auf: start_paper_trading.bat"
echo "                 Startet USDCAD v4 + USDJPY v4 (Two-Stage, Regime 1+2)"
echo "                 Demo-Konto → kein echtes Geld, aber echte Orders mit PnL"
echo ""
echo "       Option D) Manuell in zwei separaten PowerShell-Fenstern:"
cat <<'EOF'
                                 Fenster 1 (USDCAD v4):
                                     python live\live_trader.py `
                                         --symbol USDCAD `
                                         --version v4 `
                                         --paper_trading 0 `
                                         --schwelle 0.55 `
                                         --short_schwelle 0.45 `
                                         --decision_mapping long_prob `
                                         --regime_filter 1,2 `
                                         --atr_sl 1 `
                                         --atr_faktor 1.5 `
                                         --lot 0.01 `
                                         --two_stage_enable 1 `
                                         --two_stage_ltf_timeframe M5 `
                                         --two_stage_version v4

                                 Fenster 2 (USDJPY v4):
                                     python live\live_trader.py `
                                         --symbol USDJPY `
                                         --version v4 `
                                         --paper_trading 0 `
                                         --schwelle 0.55 `
                                         --short_schwelle 0.45 `
                                         --decision_mapping long_prob `
                                         --regime_filter 1,2 `
                                         --atr_sl 1 `
                                         --atr_faktor 1.5 `
                                         --lot 0.01 `
                                         --two_stage_enable 1 `
                                         --two_stage_ltf_timeframe M5 `
                                         --two_stage_version v4
EOF
echo ""
echo "  ⚠️  Aktuelle Einstellung (2026-03-10):"
echo "       - Empfohlener Startpfad: Test 128 + Log-Sync"
echo "       - Test 128: Paper-Modus (paper_trading=1), Log-Ordner logs/paper_test128"
echo "       - Alternativ vorhanden: Demo-Live mit start_paper_trading.bat"
echo "       - MT5-Zugangsdaten sind in start_paper_trading_test128.bat bereits hinterlegt"
echo ""
echo "  Modelle übertragen:"
for MODELL in "${MODELLE[@]}"; do
    if [ -f "${MODELL_ORDNER}/${MODELL}" ]; then
        echo "       ✅ models/${MODELL}"
    fi
done
echo ""
