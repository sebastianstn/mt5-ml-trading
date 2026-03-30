#!/bin/bash
# =============================================================================
# linux_log_sync_to_server.sh – Logs vom Linux Mint Laptop zum Linux-Server
#
# Ersetzt: Windows Task Scheduler + windows_sync_live_logs.ps1
# Läuft auf: Linux Mint Laptop
#
# Verwendung:
#   bash linux_log_sync_to_server.sh          → einmaliger Sync
#   bash linux_log_sync_to_server.sh install  → Crontab einrichten (alle 5 Min)
#   bash linux_log_sync_to_server.sh remove   → Crontab-Eintrag entfernen
#
# Voraussetzung:
#   - SSH-Key-Auth zum Server eingerichtet (kein Passwort nötig für Cron)
#     ssh-keygen -t ed25519
#     ssh-copy-id sebastian@192.168.1.35
# =============================================================================

# --- Konfiguration ---
LOCAL_LOG_DIR="$HOME/mt5_trading/logs"
SERVER_USER="sebastian"
SERVER_IP="192.168.1.35"
SERVER_LOG_DIR="/mnt/1Tb-Data/XGBoost-LightGBM/logs"
LOCK_FILE="/tmp/mt5_log_sync.lock"

# --- Einmaliger Sync ---
do_sync() {
    # Lock verhindert parallele Ausführung
    if [ -f "$LOCK_FILE" ]; then
        echo "[INFO] Sync läuft bereits (Lock-Datei vorhanden). Abbruch."
        exit 0
    fi
    touch "$LOCK_FILE"
    trap "rm -f '$LOCK_FILE'" EXIT

    if [ ! -d "$LOCAL_LOG_DIR" ]; then
        echo "[WARNUNG] Log-Ordner nicht gefunden: $LOCAL_LOG_DIR"
        exit 0
    fi

    # rsync: nur CSV + DB + Log-Dateien übertragen, keine unnötigen Dateien
    rsync -az --include="*.csv" --include="*.log" --include="*.db" --include="*.json" \
        --exclude="*" \
        "$LOCAL_LOG_DIR/" \
        "${SERVER_USER}@${SERVER_IP}:${SERVER_LOG_DIR}/"

    echo "[OK] $(date '+%Y-%m-%d %H:%M:%S') – Sync abgeschlossen"
}

# --- Crontab einrichten ---
install_cron() {
    SCRIPT_PATH="$(realpath "$0")"
    CRON_LINE="*/5 * * * * $SCRIPT_PATH >> /tmp/mt5_log_sync.log 2>&1"

    # Prüfen ob Eintrag schon existiert
    if crontab -l 2>/dev/null | grep -qF "$SCRIPT_PATH"; then
        echo "[INFO] Crontab-Eintrag existiert bereits."
    else
        # Eintrag hinzufügen
        (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
        echo "[OK] Crontab eingerichtet: alle 5 Minuten"
        echo "     $CRON_LINE"
    fi

    echo ""
    echo "Aktueller Crontab:"
    crontab -l
}

# --- Crontab entfernen ---
remove_cron() {
    SCRIPT_PATH="$(realpath "$0")"
    crontab -l 2>/dev/null | grep -vF "$SCRIPT_PATH" | crontab -
    echo "[OK] Crontab-Eintrag entfernt."
}

# --- Hauptprogramm ---
case "${1:-sync}" in
    install) install_cron ;;
    remove)  remove_cron  ;;
    *)       do_sync      ;;
esac
