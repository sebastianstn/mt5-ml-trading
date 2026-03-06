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

    if sftp -q -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" >"${tmp_out}" 2>&1 <<SFTP_END
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

# Welche Modelle übertragen? (v4 Kontrollgruppe + v5 Kandidat für Shadow-Compare)
MODELLE=(
    "lgbm_usdcad_v4.pkl"                    # USDCAD H1 – Single-Stage Fallback (v4)
    "lgbm_usdcad_v5.pkl"                    # USDCAD H1 – Single-Stage Fallback (v5)
    "lgbm_htf_bias_usdcad_H1_v4.pkl"        # USDCAD HTF-Bias (Two-Stage)
    "lgbm_htf_bias_usdcad_H1_v5.pkl"        # USDCAD HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdcad_M5_v4.pkl"       # USDCAD LTF-Entry (Two-Stage)
    "lgbm_ltf_entry_usdcad_M5_v5.pkl"       # USDCAD LTF-Entry (Two-Stage)
    "two_stage_usdcad_M5_v4.json"            # USDCAD Two-Stage Metadaten (Feature-Listen)
    "two_stage_usdcad_M5_v5.json"            # USDCAD Two-Stage Metadaten (Feature-Listen)
    "lgbm_usdjpy_v4.pkl"                    # USDJPY H1 – Single-Stage Fallback (v4)
    "lgbm_usdjpy_v5.pkl"                    # USDJPY H1 – Single-Stage Fallback (v5)
    "lgbm_htf_bias_usdjpy_H1_v4.pkl"        # USDJPY HTF-Bias (Two-Stage)
    "lgbm_htf_bias_usdjpy_H1_v5.pkl"        # USDJPY HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdjpy_M5_v4.pkl"       # USDJPY LTF-Entry (Two-Stage)
    "lgbm_ltf_entry_usdjpy_M5_v5.pkl"       # USDJPY LTF-Entry (Two-Stage)
    "two_stage_usdjpy_M5_v4.json"            # USDJPY Two-Stage Metadaten (Feature-Listen)
    "two_stage_usdjpy_M5_v5.json"            # USDJPY Two-Stage Metadaten (Feature-Listen)
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

echo "[ 1/4 ] Teste SSH-Verbindung zum Laptop..."
if ssh -o ConnectTimeout=5 "${LAPTOP_BENUTZER}@${LAPTOP_IP}" "echo 'SSH OK'" 2>/dev/null; then
    echo "        ✅ SSH-Verbindung erfolgreich"
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
ssh "${LAPTOP_BENUTZER}@${LAPTOP_IP}" "powershell -Command \"New-Item -ItemType Directory -Force -Path '${LAPTOP_ZIELORDNER}/live','${LAPTOP_ZIELORDNER}/live/mt5','${LAPTOP_ZIELORDNER}/models','${LAPTOP_ZIELORDNER}/logs' | Out-Null; Write-Output 'Ordner erstellt'\""
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/live/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/live/mt5/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/models/"
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/logs/"

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
sftp_put "${LIVE_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/live_trader.py"
echo "        ✅ live/live_trader.py"

# MT5 Dashboard-Indicator (MQL5)
if [ -f "${MT5_DASHBOARD_SKRIPT}" ]; then
    sftp_put "${MT5_DASHBOARD_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/mt5/LiveSignalDashboard.mq5"
    echo "        ✅ live/mt5/LiveSignalDashboard.mq5"
else
    echo "        ⚠️  LiveSignalDashboard.mq5 nicht gefunden – übersprungen"
fi

# Two-Stage-Signal-Modul (für Shadow-Mode)
TWO_STAGE_SKRIPT="${SERVER_BASIS}/live/two_stage_signal.py"
if [ -f "${TWO_STAGE_SKRIPT}" ]; then
    sftp_put "${TWO_STAGE_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/two_stage_signal.py"
    echo "        ✅ live/two_stage_signal.py (Two-Stage Shadow-Mode)"
fi

sftp_put "${REQUIREMENTS}" "${LAPTOP_ZIELORDNER_SFTP}/requirements-laptop.txt"
echo "        ✅ requirements-laptop.txt"

# Batch-Skript für automatischen Start beider Trader
sftp_put "${SERVER_BASIS}/start_both_traders.bat" "${LAPTOP_ZIELORDNER_SFTP}/start_both_traders.bat"
echo "        ✅ start_both_traders.bat"

# Batch-Skript für kontrollierten Shadow-Compare
SHADOW_START_SKRIPT="${SERVER_BASIS}/start_shadow_compare.bat"
if [ -f "${SHADOW_START_SKRIPT}" ]; then
    sftp_put "${SHADOW_START_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/start_shadow_compare.bat"
    echo "        ✅ start_shadow_compare.bat"
fi

# Batch-Skript für aggressiven Demo-Micro-Reactive Modus
DEMO_MICRO_SKRIPT="${SERVER_BASIS}/start_demo_micro_reactive.bat"
if [ -f "${DEMO_MICRO_SKRIPT}" ]; then
    sftp_put "${DEMO_MICRO_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/start_demo_micro_reactive.bat"
    echo "        ✅ start_demo_micro_reactive.bat"
fi

# Batch-Skript für Demo-Turbo-Max (maximale Aktivität)
DEMO_TURBO_SKRIPT="${SERVER_BASIS}/start_demo_turbo_max.bat"
if [ -f "${DEMO_TURBO_SKRIPT}" ]; then
    sftp_put "${DEMO_TURBO_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/start_demo_turbo_max.bat"
    echo "        ✅ start_demo_turbo_max.bat"
fi

# Batch-Skript zum sauberen Stoppen aller Trader
STOP_ALL_SKRIPT="${SERVER_BASIS}/stop_all_traders.bat"
if [ -f "${STOP_ALL_SKRIPT}" ]; then
    sftp_put "${STOP_ALL_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/stop_all_traders.bat"
    echo "        ✅ stop_all_traders.bat"
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
echo "  6. Paper-Trading starten:"
echo "       Option A) Baseline-Betrieb (beide v4):"
echo "                 Doppelklick auf: start_both_traders.bat"
echo ""
echo "       Option B) Shadow-Compare (empfohlen aktuell):"
echo "                 Doppelklick auf: start_shadow_compare.bat"
echo ""
echo "       Option C) Demo-Micro-Reactive (mehr Aktivität/mehr Trades):"
echo "                 Doppelklick auf: start_demo_micro_reactive.bat"
echo ""
echo "       Option D) Demo-Turbo-Max (maximale Aktivität):"
echo "                 Doppelklick auf: start_demo_turbo_max.bat"
echo ""
echo "       Option E) Alle Trader sauber stoppen (vor Neustart empfohlen):"
echo "                 Doppelklick auf: stop_all_traders.bat"
echo ""
echo "       Option F) Manuell in zwei separaten PowerShell-Fenstern:"
cat <<'EOF'
                                 Fenster 1 (USDCAD v4):
                                     python live\live_trader.py `
                                         --symbol USDCAD `
                                         --version v4 `
                                         --schwelle 0.55 `
                                         --short_schwelle 0.45 `
                                         --decision_mapping long_prob `
                                         --regime_filter 0,1,2 `
                                         --atr_sl 1 `
                                         --atr_faktor 1.5 `
                                         --lot 0.01 `
                                         --two_stage_enable 1 `
                                         --two_stage_ltf_timeframe M5 `
                                         --two_stage_version v4

                                 Fenster 2 (USDJPY v5):
                                     python live\live_trader.py `
                                         --symbol USDJPY `
                                         --version v5 `
                                         --schwelle 0.55 `
                                         --short_schwelle 0.45 `
                                         --decision_mapping long_prob `
                                         --regime_filter 0,1,2 `
                                         --atr_sl 1 `
                                         --atr_faktor 1.5 `
                                         --lot 0.01 `
                                         --two_stage_enable 1 `
                                         --two_stage_ltf_timeframe M5 `
                                         --two_stage_version v5
EOF
echo ""
echo "  ⚠️  Aktuelle Einstellung (2026-03-05):"
echo "       - Shadow-Compare: USDCAD v4 (Kontrolle) vs USDJPY v5 (Kandidat)"
echo "       - Schwelle: Long>=0.55 / Short<=0.45, Regime: 0,1,2"
echo "       - Betriebsmodus: PAPER_ONLY"
echo ""
echo "  Modelle übertragen:"
for MODELL in "${MODELLE[@]}"; do
    if [ -f "${MODELL_ORDNER}/${MODELL}" ]; then
        echo "       ✅ models/${MODELL}"
    fi
done
echo ""
