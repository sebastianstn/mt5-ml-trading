#!/bin/bash
# =============================================================================
# start_paper_trading_linux.sh – Paper-Trading starten auf Linux Mint
#
# Ersetzt: start_testphase_topconfig_H1_M15.bat (Windows)
# Läuft auf: Linux Mint (MT5 via Wine)
#
# Konfiguration (Stand: 19.03.2026):
#   - Beide Symbole: USDCAD + USDJPY, Two-Stage v4 (H1-Bias + M15-Entry)
#   - Schwelle: 50% | Mapping=long_prob | TP=0.6% | SL=0.3% | ATR-SL=2.0x
#   - Cooldown: 3 Bars | Regime=0,1,2,3 | Quelle=HMM
#   - Kill-Switch: Drawdown > 15% → automatischer Stopp
#   - Startkapital: 10.000 (für Kill-Switch-Berechnung)
#
# Voraussetzung:
#   - MT5 läuft via Wine (wine mt5setup.exe → MT5 starten)
#   - Python venv existiert unter ~/mt5_trading/venv/
#   - Modelle v4 nach ~/mt5_trading/models/ deployed
#
# Zum Stoppen:
#   bash stop_all_traders.sh
# =============================================================================

set -e

# --- Pfade ---
BASE_DIR="$HOME/mt5_trading"
PYTHON_EXE="$BASE_DIR/venv/bin/python"
TRADER_SCRIPT="$BASE_DIR/live/live_trader.py"
RPYC_SCRIPT="$BASE_DIR/scripts/start_mt5_rpyc_server.sh"

# --- MT5-Zugangsdaten ---
MT5_SERVER="SwissquoteLtd-Server"
MT5_LOGIN="6202835"
MT5_PASSWORD="*0YsQqAk"

# --- RPyC-Bridge-Server prüfen/starten ---
RPYC_PORT="${MT5_RPYC_PORT:-18812}"
if ss -tlnp 2>/dev/null | grep -q ":${RPYC_PORT} "; then
    echo "[OK] RPyC-Bridge-Server läuft bereits auf Port ${RPYC_PORT}"
else
    echo "[INFO] RPyC-Bridge-Server nicht gefunden – starte automatisch..."
    if [ -f "$RPYC_SCRIPT" ]; then
        gnome-terminal \
            --title "MT5-RPyC-Bridge" \
            -- bash -c "bash '$RPYC_SCRIPT'; echo 'RPyC-Server beendet.'; sleep 30" &
        echo "[INFO] Warte 8 Sekunden auf RPyC-Server-Start..."
        sleep 8
        if ss -tlnp 2>/dev/null | grep -q ":${RPYC_PORT} "; then
            echo "[OK] RPyC-Bridge-Server gestartet auf Port ${RPYC_PORT}"
        else
            echo "[FEHLER] RPyC-Server konnte nicht starten!"
            echo "         Bitte manuell starten: bash scripts/start_mt5_rpyc_server.sh"
            echo "         Voraussetzung: MT5 muss via Wine laufen + Wine-Python installiert"
            exit 1
        fi
    else
        echo "[FEHLER] RPyC-Skript nicht gefunden: $RPYC_SCRIPT"
        echo "         Bitte vom Server deployen: bash deploy_to_laptop.sh"
        exit 1
    fi
fi

# --- Voraussetzungen prüfen ---
if [ ! -d "$BASE_DIR" ]; then
    echo "[FEHLER] Projektordner nicht gefunden: $BASE_DIR"
    exit 1
fi

if [ ! -f "$PYTHON_EXE" ]; then
    echo "[WARNUNG] venv-Python nicht gefunden: $PYTHON_EXE"
    echo "[INFO] Fallback auf System-Python..."
    PYTHON_EXE="python3"
fi

if [ ! -f "$TRADER_SCRIPT" ]; then
    echo "[FEHLER] Script nicht gefunden: $TRADER_SCRIPT"
    exit 1
fi

cd "$BASE_DIR"

echo "========================================================"
echo "  MT5 Testphase - TOP-Konfiguration (Paper)"
echo "  USDCAD + USDJPY | Two-Stage v4 (H1-Bias + M15-Entry)"
echo "  Schwelle=50% | Mapping=long_prob | TP=0.6% | SL=0.3% | ATR-SL=2.0x"
echo "  Cooldown=3 Bars | Regime=0,1,2,3 | Quelle=HMM"
echo "  Kill-Switch=15% DD | Startkapital=10.000"
echo "  Server: $MT5_SERVER | Login: $MT5_LOGIN"
echo "  Python: $PYTHON_EXE"
echo "  Log-Ordner: $BASE_DIR/logs"
echo "========================================================"

# Gemeinsame Argumente für beide Symbole
COMMON_ARGS=(
    --version v4
    --paper_trading 1
    --schwelle 0.50
    --short_schwelle 0.50
    --decision_mapping long_prob
    --regime_source market_regime_hmm
    --regime_filter 0,1,2,3
    --atr_sl 1
    --atr_faktor 2.0
    --lot 0.01
    --tp_pct 0.006
    --sl_pct 0.003
    --two_stage_enable 1
    --two_stage_ltf_timeframe M15
    --two_stage_version v4
    --two_stage_kongruenz 1
    --two_stage_allow_neutral_htf 1
    --two_stage_cooldown_bars 3
    --startup_observation_bars 5
    --heartbeat_log 1
    --kill_switch_dd 0.15
    --kapital_start 10000
    --mt5_server "$MT5_SERVER"
    --mt5_login "$MT5_LOGIN"
    --mt5_password "$MT5_PASSWORD"
)

# --- Fenster 1: USDCAD ---
echo "[INFO] Starte USDCAD..."
gnome-terminal \
    --title "MT5-Testphase-USDCAD-v4" \
    -- bash -c "
        cd '$BASE_DIR'
        '$PYTHON_EXE' '$TRADER_SCRIPT' --symbol USDCAD ${COMMON_ARGS[*]}
        echo '[INFO] USDCAD Trader beendet. Fenster schließt in 10s...'
        sleep 10
    " &

# 5 Sekunden warten damit MT5-Verbindung stabil ist
sleep 5

# --- Fenster 2: USDJPY ---
echo "[INFO] Starte USDJPY..."
gnome-terminal \
    --title "MT5-Testphase-USDJPY-v4" \
    -- bash -c "
        cd '$BASE_DIR'
        '$PYTHON_EXE' '$TRADER_SCRIPT' --symbol USDJPY ${COMMON_ARGS[*]}
        echo '[INFO] USDJPY Trader beendet. Fenster schließt in 10s...'
        sleep 10
    " &

echo ""
echo "[OK] Beide Trader gestartet."
echo ""
echo "  Fenster 1: USDCAD v4 (Two-Stage H1-M15, Paper)"
echo "  Fenster 2: USDJPY v4 (Two-Stage H1-M15, Paper)"
echo ""
echo "  Logs: $BASE_DIR/logs"
echo "  Zum Stoppen: bash stop_all_traders.sh"
echo ""
