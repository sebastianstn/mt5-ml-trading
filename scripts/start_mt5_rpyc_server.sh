#!/bin/bash
# =============================================================================
# start_mt5_rpyc_server.sh – RPyC-Bridge-Server für mt5linux starten
#
# Läuft auf: Linux Mint Laptop
#
# Was passiert hier?
#   Wine-Python startet einen RPyC-Server (Port 18812), der die MetaTrader5
#   Python-Bibliothek freigibt. Das native Linux-Python (live_trader.py)
#   verbindet sich über mt5linux zu diesem Server.
#
# Voraussetzung:
#   - Wine + Wine-Python installiert (bash scripts/setup_wine_mt5.sh)
#   - MetaTrader5 + rpyc in Wine-Python installiert
#   - MT5-Terminal läuft via Wine
#
# Verwendung:
#   bash scripts/start_mt5_rpyc_server.sh        → im Vordergrund
#   bash scripts/start_mt5_rpyc_server.sh &       → im Hintergrund
#   bash scripts/start_mt5_rpyc_server.sh stop    → Server stoppen
#
# Konfiguration über Umgebungsvariablen:
#   MT5_RPYC_HOST (Standard: localhost)
#   MT5_RPYC_PORT (Standard: 18812)
# =============================================================================

RPYC_HOST="${MT5_RPYC_HOST:-localhost}"
RPYC_PORT="${MT5_RPYC_PORT:-18812}"

# --- Stop-Befehl ---
if [ "$1" = "stop" ]; then
    PIDS=$(pgrep -f "rpyc_mt5_server" 2>/dev/null)
    if [ -z "$PIDS" ]; then
        echo "[INFO] Kein RPyC-Server-Prozess gefunden."
    else
        kill -TERM $PIDS 2>/dev/null
        echo "[OK] RPyC-Server gestoppt (PIDs: $PIDS)"
    fi
    exit 0
fi

echo "================================================"
echo "  MT5 RPyC-Bridge-Server (mt5linux)"
echo "  Host: $RPYC_HOST | Port: $RPYC_PORT"
echo "  Beenden: Ctrl+C"
echo "================================================"
echo ""

# --- Wine-Python prüfen ---
WINE_PYTHON=""
# Mögliche Wine-Python-Pfade prüfen
for CANDIDATE in \
    "$HOME/.wine/drive_c/users/$USER/AppData/Local/Programs/Python/Python312/python.exe" \
    "$HOME/.wine/drive_c/users/$USER/AppData/Local/Programs/Python/Python311/python.exe" \
    "$HOME/.wine/drive_c/Python312/python.exe" \
    "$HOME/.wine/drive_c/Python311/python.exe" \
    "$HOME/.wine/drive_c/Python/python.exe"; do
    if [ -f "$CANDIDATE" ]; then
        WINE_PYTHON="$CANDIDATE"
        break
    fi
done

if [ -z "$WINE_PYTHON" ]; then
    # Fallback: wine python direkt versuchen
    if wine python --version >/dev/null 2>&1; then
        WINE_PYTHON="python"
        echo "[INFO] Wine-Python über PATH gefunden: $(wine python --version 2>&1)"
    else
        echo "[FEHLER] Wine-Python nicht gefunden!"
        echo ""
        echo "  Lösung: bash scripts/setup_wine_mt5.sh ausführen"
        echo "  Oder manuell: Python-Installer für Windows in Wine installieren"
        exit 1
    fi
else
    echo "[INFO] Wine-Python gefunden: $WINE_PYTHON"
fi

# --- Port-Check ---
if ss -tlnp 2>/dev/null | grep -q ":${RPYC_PORT} "; then
    echo "[INFO] Port ${RPYC_PORT} ist bereits belegt – RPyC-Server läuft vermutlich schon."
    echo "       Zum Stoppen: bash scripts/start_mt5_rpyc_server.sh stop"
    exit 0
fi

# --- RPyC-Server in Wine-Python starten ---
echo "[INFO] Starte RPyC-Server auf ${RPYC_HOST}:${RPYC_PORT}..."
echo ""

wine "$WINE_PYTHON" -c "
import sys
print(f'Wine-Python {sys.version}')

# MetaTrader5 vorab importieren (prüft ob MT5 verfügbar ist)
try:
    import MetaTrader5 as mt5
    print(f'MetaTrader5 v{mt5.__version__} geladen')
except ImportError:
    print('FEHLER: MetaTrader5 nicht in Wine-Python installiert!')
    print('Lösung: wine python -m pip install MetaTrader5')
    sys.exit(1)

# RPyC-Server starten
try:
    import rpyc
    from rpyc.utils.server import ThreadedServer
    from rpyc.core.service import SlaveService
except ImportError:
    print('FEHLER: rpyc nicht in Wine-Python installiert!')
    print('Lösung: wine python -m pip install rpyc')
    sys.exit(1)

config = {
    'allow_public_attrs': True,
    'allow_all_attrs': True,
    'allow_pickle': False,
}

server = ThreadedServer(
    SlaveService,
    hostname='${RPYC_HOST}',
    port=${RPYC_PORT},
    protocol_config=config,
)
print(f'RPyC MT5 Bridge-Server aktiv auf ${RPYC_HOST}:${RPYC_PORT}')
print('Warte auf Verbindungen von live_trader.py ...')
print('Zum Beenden: Ctrl+C')
server.start()
"
