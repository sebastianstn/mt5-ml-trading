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
SERVER_BASIS="/mnt/1T-Data/XGBoost-LightGBM"
MODELL_ORDNER="${SERVER_BASIS}/models"
LIVE_SKRIPT="${SERVER_BASIS}/live/live_trader.py"
REQUIREMENTS="${SERVER_BASIS}/requirements-laptop.txt"

# Welche Modelle übertragen? (Two-Stage für USDCAD + USDJPY, H4 für USDCHF)
MODELLE=(
    "lgbm_usdcad_v4.pkl"                    # USDCAD H1 – Single-Stage Fallback (v4)
    "lgbm_htf_bias_usdcad_H1_v4.pkl"        # USDCAD HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdcad_M5_v4.pkl"       # USDCAD LTF-Entry (Two-Stage)
    "two_stage_usdcad_M5_v4.json"            # USDCAD Two-Stage Metadaten (Feature-Listen)
    "lgbm_usdjpy_v4.pkl"                    # USDJPY H1 – Single-Stage Fallback (v4)
    "lgbm_htf_bias_usdjpy_H1_v4.pkl"        # USDJPY HTF-Bias (Two-Stage)
    "lgbm_ltf_entry_usdjpy_M5_v4.pkl"       # USDJPY LTF-Entry (Two-Stage)
    "two_stage_usdjpy_M5_v4.json"            # USDJPY Two-Stage Metadaten (Feature-Listen)
    "lgbm_usdchf_H4_v1.pkl"                 # USDCHF H4 – Optionaler 3. Kandidat
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
ssh "${LAPTOP_BENUTZER}@${LAPTOP_IP}" "powershell -Command \"New-Item -ItemType Directory -Force -Path '${LAPTOP_ZIELORDNER}/live','${LAPTOP_ZIELORDNER}/models','${LAPTOP_ZIELORDNER}/logs' | Out-Null; Write-Output 'Ordner erstellt'\""
echo "        ✅ Ordner: ${LAPTOP_ZIELORDNER}/live/"
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
        sftp -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" <<SFTP_END
put "${PFAD}" "${LAPTOP_ZIELORDNER_SFTP}/models/${MODELL}"
SFTP_END
        echo "        ✅ ${MODELL}"
    else
        echo "        ⚠️  ${MODELL} nicht gefunden – übersprungen"
    fi
done

# ------------------------------------------------------------
# Live-Skript + Requirements übertragen
# ------------------------------------------------------------
echo ""
echo "[ 4/4 ] Übertrage Skripte..."
sftp -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" <<SFTP_END
put "${LIVE_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/live_trader.py"
SFTP_END
echo "        ✅ live/live_trader.py"

# Two-Stage-Signal-Modul (für Shadow-Mode)
TWO_STAGE_SKRIPT="${SERVER_BASIS}/live/two_stage_signal.py"
if [ -f "${TWO_STAGE_SKRIPT}" ]; then
    sftp -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" <<SFTP_END
put "${TWO_STAGE_SKRIPT}" "${LAPTOP_ZIELORDNER_SFTP}/live/two_stage_signal.py"
SFTP_END
    echo "        ✅ live/two_stage_signal.py (Two-Stage Shadow-Mode)"
fi

sftp -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" <<SFTP_END
put "${REQUIREMENTS}" "${LAPTOP_ZIELORDNER_SFTP}/requirements-laptop.txt"
SFTP_END
echo "        ✅ requirements-laptop.txt"

# Batch-Skript für automatischen Start beider Trader
sftp -b - "${LAPTOP_BENUTZER}@${LAPTOP_IP}" <<SFTP_END
put "${SERVER_BASIS}/start_both_traders.bat" "${LAPTOP_ZIELORDNER_SFTP}/start_both_traders.bat"
SFTP_END
echo "        ✅ start_both_traders.bat"

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
echo "       Option A) Beide Trader automatisch starten (empfohlen):"
echo "                 Doppelklick auf: start_both_traders.bat"
echo ""
echo "       Option B) Manuell in zwei separaten PowerShell-Fenstern:"
echo "                 python live\\live_trader.py --symbol USDCAD --version v4 --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4"
echo "                 python live\\live_trader.py --symbol USDJPY --version v4 --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4"
echo ""
echo "  ⚠️  Aktuelle Einstellung: Test-Phase (Option 1)"
echo "       - Schwelle: 52% (gesenkt für mehr Aktivität)"
echo "       - Regime: Alle (0,1,2) erlaubt"
echo ""
echo "  Modelle übertragen:"
for MODELL in "${MODELLE[@]}"; do
    if [ -f "${MODELL_ORDNER}/${MODELL}" ]; then
        echo "       ✅ models/${MODELL}"
    fi
done
echo ""
