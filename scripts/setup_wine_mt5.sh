#!/bin/bash
# =============================================================================
# setup_wine_mt5.sh – Einmalige Einrichtung: Wine + MT5 + Python auf Linux Mint
#
# Läuft auf: Linux Mint Laptop (einmalig ausführen!)
#
# Was wird installiert?
#   1. Wine (32-bit + 64-bit)
#   2. MetaTrader 5 (via Wine)
#   3. Python 3.12 für Wine (Windows-Python)
#   4. MetaTrader5 + rpyc in Wine-Python
#   5. Native Python-Umgebung mit mt5linux + Trading-Dependencies
#
# Verwendung:
#   cd ~/mt5_trading
#   bash scripts/setup_wine_mt5.sh
#
# Nach dem Setup:
#   1. MT5 starten:  wine "$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe" &
#   2. RPyC-Server:  bash scripts/start_mt5_rpyc_server.sh &
#   3. Trader:        bash start_paper_trading_linux.sh
# =============================================================================

set -e

echo "============================================================"
echo "  MT5 ML-Trading: Wine + MT5 Setup für Linux Mint"
echo "  Einmalige Einrichtung"
echo "============================================================"
echo ""

# ---- Hilfsfunktion ----
check_ok() {
    if [ $? -eq 0 ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ $1 – FEHLER!"
        return 1
    fi
}

# ============================================================
# Schritt 1: Wine installieren
# ============================================================
echo "[1/6] Wine installieren..."
echo ""

if wine --version >/dev/null 2>&1; then
    echo "  ✅ Wine bereits installiert: $(wine --version)"
else
    echo "  Wine wird installiert (32-bit + 64-bit)..."
    sudo dpkg --add-architecture i386
    sudo apt update
    sudo apt install -y wine64 wine32
    check_ok "Wine installiert"
fi

# Wine-Prefix initialisieren (falls noch nicht vorhanden)
if [ ! -d "$HOME/.wine" ]; then
    echo "  Wine-Prefix wird initialisiert..."
    WINEARCH=win64 wineboot --init
    sleep 5
    echo "  ✅ Wine-Prefix erstellt"
else
    echo "  ✅ Wine-Prefix vorhanden: $HOME/.wine"
fi

# ============================================================
# Schritt 2: MetaTrader 5 installieren
# ============================================================
echo ""
echo "[2/6] MetaTrader 5 installieren..."
echo ""

MT5_EXE="$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
if [ -f "$MT5_EXE" ]; then
    echo "  ✅ MT5 bereits installiert: $MT5_EXE"
else
    echo "  MT5-Installer wird heruntergeladen..."
    cd /tmp
    wget -q --show-progress https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
    echo ""
    echo "  ⚡ MT5-Installer startet – bitte durchklicken und installieren."
    echo "     Standard-Pfad beibehalten!"
    echo ""
    wine /tmp/mt5setup.exe
    echo ""

    if [ -f "$MT5_EXE" ]; then
        echo "  ✅ MT5 erfolgreich installiert"
    else
        echo "  ⚠️  MT5 nicht im erwarteten Pfad gefunden."
        echo "     Falls in einem anderen Ordner installiert → Pfad in start_paper_trading_linux.sh anpassen."
    fi
fi

# ============================================================
# Schritt 3: Python für Wine installieren
# ============================================================
echo ""
echo "[3/6] Python 3.12 für Wine installieren..."
echo ""

# Prüfen ob Wine-Python schon da ist
WINE_PYTHON_OK=false
if wine python --version >/dev/null 2>&1; then
    WINE_PY_VER=$(wine python --version 2>&1 | grep -oP 'Python \K[0-9.]+')
    echo "  ✅ Wine-Python bereits installiert: $WINE_PY_VER"
    WINE_PYTHON_OK=true
fi

if [ "$WINE_PYTHON_OK" = false ]; then
    PYTHON_VER="3.12.3"
    PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VER}/python-${PYTHON_VER}-amd64.exe"

    cd /tmp
    if [ ! -f "python-${PYTHON_VER}-amd64.exe" ]; then
        echo "  Python ${PYTHON_VER} Installer wird heruntergeladen..."
        wget -q --show-progress "$PYTHON_URL"
    fi

    echo "  Python ${PYTHON_VER} wird in Wine installiert (Silent-Modus)..."
    wine "python-${PYTHON_VER}-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1

    # Warten bis Installation fertig
    echo "  Warte auf Abschluss der Installation..."
    sleep 15

    if wine python --version >/dev/null 2>&1; then
        echo "  ✅ Wine-Python installiert: $(wine python --version 2>&1)"
    else
        echo "  ❌ Wine-Python nicht erreichbar nach Installation!"
        echo "     Versuche: wine python --version"
        echo "     Falls Fehler: Wine-Prefix mit 'wineboot -u' updaten"
        exit 1
    fi
fi

# ============================================================
# Schritt 4: MetaTrader5 + rpyc in Wine-Python installieren
# ============================================================
echo ""
echo "[4/6] MetaTrader5 + rpyc in Wine-Python installieren..."
echo ""

wine python -m pip install --upgrade pip 2>/dev/null
# WICHTIG: numpy<2.0 erzwingen – Wine 9.0 fehlt ucrtbase.dll.crealf (von numpy 2.x benötigt)
wine python -m pip install "numpy>=1.26,<2.0" 2>/dev/null
# WICHTIG: rpyc==5.2.3 erzwingen – mt5linux 1.0.3 braucht rpyc 5.x, nicht 6.x
wine python -m pip install MetaTrader5 "rpyc==5.2.3" 2>/dev/null
check_ok "MetaTrader5 + rpyc in Wine-Python"

# Schnelltest
echo "  Teste Wine-Python Imports..."
wine python -c "import MetaTrader5; import rpyc; print('OK: MT5 + rpyc importierbar')" 2>/dev/null
check_ok "Wine-Python Import-Test"

# ============================================================
# Schritt 5: Native Python-Umgebung erstellen
# ============================================================
echo ""
echo "[5/6] Native Python-Umgebung erstellen..."
echo ""

cd "$HOME/mt5_trading"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✅ venv erstellt"
else
    echo "  ✅ venv vorhanden"
fi

# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements-laptop.txt 2>&1 | tail -5
check_ok "requirements-laptop.txt installiert"

# pandas_ta separat ohne numba (numba ist optional, nur Beschleunigung)
pip install pandas_ta --no-deps 2>&1 | tail -3
check_ok "pandas_ta installiert (ohne numba)"

# ============================================================
# Schritt 6: SSH-Key für Log-Sync einrichten
# ============================================================
echo ""
echo "[6/6] SSH-Key für Log-Sync zum Server vorbereiten..."
echo ""

if [ -f "$HOME/.ssh/id_ed25519" ]; then
    echo "  ✅ SSH-Key vorhanden"
else
    echo "  SSH-Key wird generiert..."
    ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -q
    echo "  ✅ SSH-Key generiert"
    echo ""
    echo "  ⚡ Jetzt Key zum Server kopieren (einmalig, Passwort nötig):"
    echo "     ssh-copy-id sebastian@192.168.1.35"
fi

# ============================================================
# Abschluss
# ============================================================
echo ""
echo "============================================================"
echo "  ✅ Setup abgeschlossen!"
echo "============================================================"
echo ""
echo "  Architektur:"
echo "    Linux-Python (live_trader.py)"
echo "         ↕ mt5linux / RPyC (Port 18812)"
echo "    Wine-Python (MetaTrader5-Bibliothek)"
echo "         ↕"
echo "    MT5-Terminal (Wine)"
echo ""
echo "  Nächste Schritte:"
echo ""
echo "  1. MT5 starten:"
echo "     wine \"$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe\" &"
echo "     → Bei Swissquote-Demo-Konto anmelden"
echo ""
echo "  2. RPyC-Bridge-Server starten (neues Terminal):"
echo "     cd ~/mt5_trading && bash scripts/start_mt5_rpyc_server.sh"
echo ""
echo "  3. Paper-Trading starten (neues Terminal):"
echo "     cd ~/mt5_trading && bash start_paper_trading_linux.sh"
echo ""
echo "  4. Log-Sync zum Server einrichten (einmalig):"
echo "     bash scripts/linux_log_sync_to_server.sh install"
echo ""
echo "  5. Alle Trader stoppen:"
echo "     bash stop_all_traders.sh"
echo ""
