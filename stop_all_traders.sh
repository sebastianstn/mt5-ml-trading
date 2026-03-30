#!/bin/bash
# =============================================================================
# stop_all_traders.sh – Laufende MT5-Trader stoppen
#
# Ersetzt: stop_all_traders.bat (Windows)
# Läuft auf: Linux Mint
#
# Verhalten:
#   - Beendet alle live_trader.py Prozesse sauber via SIGTERM
#   - Wartet 5 Sekunden, dann SIGKILL falls nötig
# =============================================================================

echo "================================================"
echo "  MT5 ML-Trading - Stoppe Trader Prozesse"
echo "================================================"
echo ""

# Alle live_trader.py PIDs finden
PIDS=$(pgrep -f "live_trader.py" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "[INFO] Keine live_trader.py Prozesse gefunden."
else
    echo "[INFO] Gefundene Prozesse:"
    for PID in $PIDS; do
        CMD=$(ps -p "$PID" -o cmd= 2>/dev/null | cut -c1-80)
        echo "  PID=$PID | $CMD"
    done
    echo ""

    # SIGTERM (sauberes Beenden, löst KeyboardInterrupt in Python aus)
    echo "[INFO] Sende SIGTERM..."
    kill -TERM $PIDS 2>/dev/null && echo "[OK] SIGTERM gesendet."

    # 5 Sekunden warten
    sleep 5

    # Prüfen ob noch Prozesse laufen
    NOCH_AKTIV=$(pgrep -f "live_trader.py" 2>/dev/null)
    if [ -n "$NOCH_AKTIV" ]; then
        echo "[WARNUNG] Prozesse antworten nicht – sende SIGKILL..."
        kill -KILL $NOCH_AKTIV 2>/dev/null
        sleep 1
    fi
fi

# Abschlusskontrolle
VERBLEIBEND=$(pgrep -f "live_trader.py" 2>/dev/null | wc -l)
echo ""
if [ "$VERBLEIBEND" -eq 0 ]; then
    echo "✅ Fertig. Alle live_trader.py Prozesse beendet."
else
    echo "[WARNUNG] $VERBLEIBEND Prozess(e) noch aktiv. Bitte manuell prüfen:"
    echo "  ps aux | grep live_trader"
fi
echo ""
