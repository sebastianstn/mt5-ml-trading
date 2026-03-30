#!/bin/bash
# =============================================================================
# deploy_to_laptop.sh – Aktive Dateien auf den Windows-Laptop deployen
#
# Ausführen auf dem Linux-Server:
#     bash deploy_to_laptop.sh
#
# Was wird deployed?
#   - v4-Modelle (USDCAD + USDJPY, Two-Stage H1-Bias + M15-Entry)
#   - Alle live/ Python-Module (Trader, Signal-Engine, Konnektoren)
#   - MQ5-Dateien (Dashboard-Indikator + PythonSignalExecutor EA)
#   - Aktive BAT-Dateien (Topconfig-Start, Stop-All, Log-Sync)
#   - Windows-Sync-Skripte (Log-Transfer Laptop → Server)
#   - requirements-laptop.txt
#
# Was wird NICHT deployed (nicht mehr aktiv):
#   - start_paper_trading.bat (alte Config, anderer Decision-Mapping)
#   - start_paper_trading_test128.bat (Test 128 war NO-GO)
#   - Optionale Modelle (z.B. USDCHF H4 v1)
#   - Tests (nicht für Laufzeit benötigt)
#
# Voraussetzung: SSH-Verbindung zum Windows-Laptop muss funktionieren.
#     Auf Windows: OpenSSH-Server installieren (Einstellungen → Apps → Optionale Features)
#     Test: ssh BENUTZER@LAPTOP_IP "echo OK"
#
# Stand: 27.03.2026
# =============================================================================

set -e  # Bei Fehler sofort stoppen

# Deploy bewusst NICHT als root ausfuehren.
# Sonst werden root-spezifische SSH-Keys/known_hosts verwendet,
# was oft zu Login-Fehlern auf dem Windows-Laptop fuehrt.
if [ "${EUID}" -eq 0 ]; then
    echo ""
    echo "[FEHLER] deploy_to_laptop.sh darf nicht mit sudo/root gestartet werden."
    echo ""
    echo "Bitte so ausfuehren:"
    echo "  bash deploy_to_laptop.sh"
    echo ""
    echo "Warum: root nutzt einen anderen SSH-Kontext (known_hosts/Keys) als dein User."
    exit 1
fi

# ------------------------------------------------------------
# Konfiguration – HIER ANPASSEN
# ------------------------------------------------------------
LAPTOP_SSH_BENUTZER="sebas"                   # SSH-Loginname auf Windows (meist lokaler Benutzername)
LAPTOP_IP="192.168.1.13"                      # IP-Adresse des Windows-Laptops
LAPTOP_SSH_PORT="22"                          # SSH-Port (Standard 22)
LAPTOP_ZIELORDNER_WIN="C:\\Users\\sebas\\mt5_trading"  # Absoluter Zielpfad auf Windows
LAPTOP_ZIELORDNER_SFTP="/C:/Users/sebas/mt5_trading"        # Absoluter Windows-Pfad für SFTP

# ------------------------------------------------------------
# SSH ControlMaster – einmal Passwort, dann Tunnel wiederverwenden
# ------------------------------------------------------------
SSH_CONTROL_DIR=$(mktemp -d)
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/ssh-%r@%h:%p"
SSH_OPTS="-o ControlPath=${SSH_CONTROL_PATH}"

# Aufräum-Funktion: SSH-Tunnel schließen wenn Skript beendet wird
cleanup() {
    ssh -p "${LAPTOP_SSH_PORT}" -o ControlPath="${SSH_CONTROL_PATH}" -O exit "${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}" 2>/dev/null || true
    rm -rf "${SSH_CONTROL_DIR}"
}
trap cleanup EXIT

# Linux-Server Pfade
SERVER_BASIS="/mnt/1Tb-Data/XGBoost-LightGBM"
MODELL_ORDNER="${SERVER_BASIS}/models"
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
    if sftp -q -P "${LAPTOP_SSH_PORT}" -o ControlPath="${SSH_CONTROL_PATH}" -b - "${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}" >"${tmp_out}" 2>&1 <<SFTP_END
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

# ------------------------------------------------------------
# Hilfsfunktion: Windows-Zielordner via PowerShell anlegen
# ------------------------------------------------------------
erstelle_windows_ordner() {
    ssh -p "${LAPTOP_SSH_PORT}" ${SSH_OPTS} "${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}" \
    "cmd /c if not exist \"${LAPTOP_ZIELORDNER_WIN}\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\" && if not exist \"${LAPTOP_ZIELORDNER_WIN}\\live\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\\live\" && if not exist \"${LAPTOP_ZIELORDNER_WIN}\\live\\mt5\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\\live\\mt5\" && if not exist \"${LAPTOP_ZIELORDNER_WIN}\\models\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\\models\" && if not exist \"${LAPTOP_ZIELORDNER_WIN}\\logs\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\\logs\" && if not exist \"${LAPTOP_ZIELORDNER_WIN}\\scripts\" mkdir \"${LAPTOP_ZIELORDNER_WIN}\\scripts\""
}

# ------------------------------------------------------------
# Zusatzdiagnose: Netzwerkstatus vom Linux-Server aus ausgeben
# ------------------------------------------------------------
zeige_netzdiagnose() {
    local route_output
    local neigh_output
    local nc_output

    echo ""
    echo "  Zusatzdiagnose vom Linux-Server:"

    route_output=$(ip route get "${LAPTOP_IP}" 2>/dev/null | head -n 1 || true)
    if [ -n "${route_output}" ]; then
        echo "  - Route: ${route_output}"
    else
        echo "  - Route: keine Route ermittelbar"
    fi

    neigh_output=$(ip neigh show "${LAPTOP_IP}" 2>/dev/null || true)
    if [ -n "${neigh_output}" ]; then
        echo "  - ARP/Neighbor-Eintrag vorhanden:"
        printf '%s\n' "${neigh_output}" | sed 's/^/    /'
    else
        echo "  - ARP/Neighbor-Eintrag: keiner vorhanden"
    fi

    if ping -c 1 -W 1 "${LAPTOP_IP}" >/dev/null 2>&1; then
        echo "  - Ping: Antwort erhalten"
    else
        echo "  - Ping: keine Antwort"
    fi

    if command -v nc >/dev/null 2>&1; then
        nc_output=$(timeout 4 nc -vz -w 2 "${LAPTOP_IP}" "${LAPTOP_SSH_PORT}" 2>&1 || true)
        if [ -n "${nc_output}" ]; then
            echo "  - Port ${LAPTOP_SSH_PORT}:"
            printf '%s\n' "${nc_output}" | sed 's/^/    /'
        fi
    fi
}

# ------------------------------------------------------------
# Modelle: Nur v4 Two-Stage (USDCAD + USDJPY) – das aktive Setup
# ------------------------------------------------------------
MODELLE=(
    "lgbm_usdcad_v4.pkl"                    # USDCAD H1 – Single-Stage Fallback
    "lgbm_htf_bias_usdcad_H1_v4.pkl"        # USDCAD HTF-Bias (Two-Stage Stufe 1)
    "lgbm_ltf_entry_usdcad_M15_v4.pkl"      # USDCAD LTF-Entry M15 (Two-Stage Stufe 2)
    "two_stage_usdcad_M15_v4.json"           # USDCAD Two-Stage Metadaten (Feature-Listen)
    "lgbm_usdjpy_v4.pkl"                     # USDJPY H1 – Single-Stage Fallback
    "lgbm_htf_bias_usdjpy_H1_v4.pkl"        # USDJPY HTF-Bias (Two-Stage Stufe 1)
    "lgbm_ltf_entry_usdjpy_M15_v4.pkl"      # USDJPY LTF-Entry M15 (Two-Stage Stufe 2)
    "two_stage_usdjpy_M15_v4.json"           # USDJPY Two-Stage Metadaten (Feature-Listen)
)

# ------------------------------------------------------------
# Verbindungstest
# ------------------------------------------------------------
echo "=================================================="
echo "  MT5 ML-Trading Deploy-Skript (Stand: 29.03.2026)"
echo "  Ziel: ${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}"
echo "  SSH-Port: ${LAPTOP_SSH_PORT}"
echo "  Aktives Setup: Two-Stage v4 (H1-Bias + M15-Entry)"
echo "  Symbole: USDCAD + USDJPY (Paper-Modus)"
echo "=================================================="
echo ""

# Kurzer Plausibilitäts-Hinweis für SSH-Usernamen
if [[ "${LAPTOP_SSH_BENUTZER}" == *" "* ]]; then
    echo "[HINWEIS] LAPTOP_SSH_BENUTZER enthält Leerzeichen ('${LAPTOP_SSH_BENUTZER}')."
    echo "          Bei Windows-OpenSSH ist der Loginname meist ohne Leerzeichen (z.B. 'sebastian')."
    echo ""
fi

echo "[ 1/4 ] Teste SSH-Verbindung zum Laptop (Passwort wird nur einmal abgefragt)..."

# Vorab: Port 22 erreichbar?
if timeout 3 bash -lc "cat < /dev/null > /dev/tcp/${LAPTOP_IP}/${LAPTOP_SSH_PORT}" 2>/dev/null; then
    echo "        ✅ Port ${LAPTOP_SSH_PORT} erreichbar"
else
    echo "        ❌ Port ${LAPTOP_SSH_PORT} auf ${LAPTOP_IP} nicht erreichbar (Netz/Firewall/SSH-Dienst)."
    zeige_netzdiagnose
    echo ""
    echo "  Interpretation:"
    echo "  - Falls ein ARP/Neighbor-Eintrag sichtbar ist, ist unter ${LAPTOP_IP} ein Gerät im LAN erreichbar."
    echo "  - Wenn Port ${LAPTOP_SSH_PORT} trotzdem in einen Timeout läuft, blockiert meist Firewall/sshd oder die IP gehört nicht zum erwarteten Laptop."
    echo ""
    echo "  Schnellchecks auf dem Windows-Laptop (PowerShell als Administrator):"
    echo ""
    echo "  1. OpenSSH-Server installieren (einmalig):"
    echo "     Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'"
    echo "     Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
    echo ""
    echo "  2. SSH-Dienst starten und aktivieren:"
    echo "     Start-Service sshd"
    echo "     Set-Service -Name sshd -StartupType Automatic"
    echo ""
    echo "  3. Status prüfen:"
    echo "     Get-Service sshd"
    echo "     Test-NetConnection -ComputerName localhost -Port 22"
    echo ""
    echo "  4. Firewall-Regel prüfen/setzen:"
    echo "     New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22"
    echo ""
    echo "  Netzwerkcheck:"
    echo "  - hostname"
    echo "  - Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.254*' }"
    echo "  - Beide Geräte im gleichen LAN/WLAN"
    echo "  - Vom Server testen: ping ${LAPTOP_IP}"
    exit 1
fi

# ControlMaster-Tunnel aufbauen: bleibt für alle weiteren SSH/SFTP-Befehle offen
if ssh -p "${LAPTOP_SSH_PORT}" -o ConnectTimeout=10 -o ControlMaster=yes -o ControlPath="${SSH_CONTROL_PATH}" -o ControlPersist=300 "${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}" "echo 'SSH OK'"; then
    echo "        ✅ SSH-Verbindung erfolgreich (Tunnel aktiv)"
else
    echo "        ❌ SSH-Verbindung fehlgeschlagen!"
    echo ""
    echo "  Interpretation:"
    echo "  - Port ${LAPTOP_SSH_PORT} ist erreichbar, aber die Anmeldung wurde vom Windows-Laptop abgewiesen."
    echo "  - Ursache ist jetzt typischerweise Benutzername/Passwort oder fehlende Public-Key-Freigabe."
    echo ""
    echo "  Lösung:"
    echo "  1. OpenSSH-Server auf Windows installieren und starten:"
    echo "     Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
    echo "     Start-Service sshd"
    echo "     Set-Service -Name sshd -StartupType Automatic"
    echo "  2. Firewall prüfen: Test-NetConnection -ComputerName localhost -Port 22"
    echo "  3. IP-Adresse prüfen: Get-NetIPAddress -AddressFamily IPv4"
    echo "  4. Benutzername prüfen: whoami"
    echo "  5. Verbindung interaktiv am Server testen: ssh -p ${LAPTOP_SSH_PORT} ${LAPTOP_SSH_BENUTZER}@${LAPTOP_IP}"
    echo "  6. Falls Passwort-Login scheitert: Windows-Kontopasswort prüfen oder SSH-Key in %USERPROFILE%\\.ssh\\authorized_keys hinterlegen"
    exit 1
fi

# ------------------------------------------------------------
# Zielordner auf Laptop erstellen (nur aktiv benötigte Ordner)
# ------------------------------------------------------------
echo ""
echo "[ 2/4 ] Erstelle Ordnerstruktur auf Windows-Laptop..."
erstelle_windows_ordner
echo "        ✅ Ordner: live/, live/mt5/, models/, logs/, scripts/"

# ------------------------------------------------------------
# Modelle übertragen (nur v4 Two-Stage, USDCAD + USDJPY)
# ------------------------------------------------------------
echo ""
echo "[ 3/4 ] Übertrage Modelle..."
for MODELL in "${MODELLE[@]}"; do
    PFAD="${MODELL_ORDNER}/${MODELL}"
    if [ -f "${PFAD}" ]; then
        GROESSE=$(du -sh "${PFAD}" | cut -f1)
        echo "        Übertrage ${MODELL} (${GROESSE})..."
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/models/${MODELL}"
        echo "        ✅ ${MODELL}"
    else
        echo "        ⚠️  ${MODELL} nicht gefunden – FEHLT!"
    fi
done

# ------------------------------------------------------------
# Skripte übertragen
# ------------------------------------------------------------
echo ""
echo "[ 4/4 ] Übertrage Skripte..."

# --- Live-Trading Python-Module (alle für den Trader erforderlich) ---
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
    "two_stage_signal.py"
    "__init__.py"
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

# --- MQ5-Dateien (Dashboard-Indikator + EA) ---
MQ5_DATEIEN=(
    "LiveSignalDashboard.mq5"
    "PythonSignalExecutor.mq5"
)
for MQ5 in "${MQ5_DATEIEN[@]}"; do
    PFAD="${SERVER_BASIS}/live/mt5/${MQ5}"
    if [ -f "${PFAD}" ]; then
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/${MQ5}"
        echo "        ✅ live/mt5/${MQ5}"
    else
        echo "        ⚠️  live/mt5/${MQ5} nicht gefunden – übersprungen"
    fi
done

# --- MT5 Common-Sync Skripte (CSV-Signal → MT5 Common/Files) ---
MT5_SYNC_DATEIEN=(
    "sync_live_logs_to_mt5_common.ps1"
    "install_sync_task.ps1"
)
for SYNC in "${MT5_SYNC_DATEIEN[@]}"; do
    PFAD="${SERVER_BASIS}/live/mt5/${SYNC}"
    if [ -f "${PFAD}" ]; then
        sftp_put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/${SYNC}"
        echo "        ✅ live/mt5/${SYNC}"
    fi
done

# --- Windows-Start- und Stop-Dateien ---
sftp_put "${SERVER_BASIS}/start_testphase_topconfig_H1_M15.bat" "${LAPTOP_ZIELORDNER_SFTP}/start_testphase_topconfig_H1_M15.bat"
echo "        ✅ start_testphase_topconfig_H1_M15.bat (AKTIVE Startdatei)"

sftp_put "${SERVER_BASIS}/stop_all_traders.bat" "${LAPTOP_ZIELORDNER_SFTP}/stop_all_traders.bat"
echo "        ✅ stop_all_traders.bat"

if [ -f "${SERVER_BASIS}/register_test128_log_sync_to_server.bat" ]; then
    sftp_put "${SERVER_BASIS}/register_test128_log_sync_to_server.bat" "${LAPTOP_ZIELORDNER_SFTP}/register_test128_log_sync_to_server.bat"
    echo "        ✅ register_test128_log_sync_to_server.bat"
else
    echo "        ⚠️  register_test128_log_sync_to_server.bat nicht gefunden – übersprungen"
fi

sftp_put "${SERVER_BASIS}/setup_windows.bat" "${LAPTOP_ZIELORDNER_SFTP}/setup_windows.bat"
echo "        ✅ setup_windows.bat"

if [ -f "${SERVER_BASIS}/check_logs_now.bat" ]; then
    sftp_put "${SERVER_BASIS}/check_logs_now.bat" "${LAPTOP_ZIELORDNER_SFTP}/check_logs_now.bat"
    echo "        ✅ check_logs_now.bat"
else
    echo "        ⚠️  check_logs_now.bat nicht gefunden – übersprungen"
fi

if [ -f "${SERVER_BASIS}/register_check_logs_auto.bat" ]; then
    sftp_put "${SERVER_BASIS}/register_check_logs_auto.bat" "${LAPTOP_ZIELORDNER_SFTP}/register_check_logs_auto.bat"
    echo "        ✅ register_check_logs_auto.bat"
else
    echo "        ⚠️  register_check_logs_auto.bat nicht gefunden – übersprungen"
fi

if [ -f "${SERVER_BASIS}/register_check_logs_rotation_auto.bat" ]; then
    sftp_put "${SERVER_BASIS}/register_check_logs_rotation_auto.bat" "${LAPTOP_ZIELORDNER_SFTP}/register_check_logs_rotation_auto.bat"
    echo "        ✅ register_check_logs_rotation_auto.bat"
else
    echo "        ⚠️  register_check_logs_rotation_auto.bat nicht gefunden – übersprungen"
fi

# --- Windows-Sync-Skripte (Laptop -> Linux-Server) ---
if [ -f "${SERVER_BASIS}/scripts/windows_sync_live_logs.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_sync_live_logs.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_sync_live_logs.ps1"
    echo "        ✅ scripts/windows_sync_live_logs.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_register_live_log_sync_task.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_register_live_log_sync_task.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_register_live_log_sync_task.ps1"
    echo "        ✅ scripts/windows_register_live_log_sync_task.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_task_live_log_sync.xml.template" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_task_live_log_sync.xml.template" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_task_live_log_sync.xml.template"
    echo "        ✅ scripts/windows_task_live_log_sync.xml.template"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_live_log_watchdog.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_live_log_watchdog.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_live_log_watchdog.ps1"
    echo "        ✅ scripts/windows_live_log_watchdog.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_run_check_logs_once.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_run_check_logs_once.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_run_check_logs_once.ps1"
    echo "        ✅ scripts/windows_run_check_logs_once.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_register_check_logs_task_v2.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_register_check_logs_task_v2.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_register_check_logs_task_v2.ps1"
    echo "        ✅ scripts/windows_register_check_logs_task_v2.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_rotate_check_logs_daily.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_rotate_check_logs_daily.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_rotate_check_logs_daily.ps1"
    echo "        ✅ scripts/windows_rotate_check_logs_daily.ps1"
fi

if [ -f "${SERVER_BASIS}/scripts/windows_register_check_logs_rotation_task.ps1" ]; then
    sftp_put "${SERVER_BASIS}/scripts/windows_register_check_logs_rotation_task.ps1" "${LAPTOP_ZIELORDNER_SFTP}/scripts/windows_register_check_logs_rotation_task.ps1"
    echo "        ✅ scripts/windows_register_check_logs_rotation_task.ps1"
fi

# --- requirements-laptop.txt ---
sftp_put "${REQUIREMENTS}" "${LAPTOP_ZIELORDNER_SFTP}/requirements-laptop.txt"
echo "        ✅ requirements-laptop.txt"

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
echo "  0. Falls erstmalig – OpenSSH auf Windows aktivieren:"
echo "       Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
echo "       Start-Service sshd"
echo "       Set-Service -Name sshd -StartupType Automatic"
echo ""
echo "  1. Projektordner und venv prüfen:"
echo "       cd /d ${LAPTOP_ZIELORDNER_WIN}"
echo "       .\\.venv\\Scripts\\activate"
echo "       (falls .venv fehlt: .\\setup_windows.bat)"
echo ""
echo "  2. Optional: Log-Sync-Task registrieren:"
echo "       .\\register_test128_log_sync_to_server.bat"
echo ""
echo "  3. Trader starten (Paper-Modus, Two-Stage v4):"
echo "       .\\start_testphase_topconfig_H1_M15.bat"
echo ""
echo "  4. Alle Trader stoppen:"
echo "       .\\stop_all_traders.bat"
echo ""
echo "  Deployed:"
for MODELL in "${MODELLE[@]}"; do
    if [ -f "${MODELL_ORDNER}/${MODELL}" ]; then
        echo "       ✅ models/${MODELL}"
    fi
done
echo ""
