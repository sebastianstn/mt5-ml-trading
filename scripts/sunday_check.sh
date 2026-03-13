#!/bin/bash
# sunday_check.sh – Wöchentliche Sonntagsroutine
# Läuft auf: Linux Server
# Aufruf:    bash scripts/sunday_check.sh
#
# Führt alle wöchentlichen KPI-Checks automatisch aus und zeigt
# eine fertige Zusammenfassung für das Wochen-Protokoll in der Roadmap.

set -e

# Farben für Terminal-Output
ROT='\033[0;31m'
GRUEN='\033[0;32m'
GELB='\033[1;33m'
BLAU='\033[0;34m'
FETT='\033[1m'
RESET='\033[0m'

BASE_DIR="/mnt/1Tb-Data/XGBoost-LightGBM"
VENV="$BASE_DIR/.venv/bin/python"
DATUM=$(date +"%d.%m.%Y")
KW=$(date +"%V")

echo ""
echo -e "${FETT}${BLAU}============================================================${RESET}"
echo -e "${FETT}${BLAU}  MT5 Trading – Sonntagsroutine | $DATUM (KW $KW)${RESET}"
echo -e "${FETT}${BLAU}============================================================${RESET}"
echo ""

# ── Schritt 1: Letzte Sync-Zeit prüfen ──────────────────────────────────────
echo -e "${FETT}[1/4] Server-Sync prüfen${RESET}"
echo "──────────────────────────────────────────"

USDCAD_SIGNALS="$BASE_DIR/logs/paper_test128/USDCAD_signals.csv"
USDJPY_SIGNALS="$BASE_DIR/logs/paper_test128/USDJPY_signals.csv"

if [ -f "$USDCAD_SIGNALS" ]; then
    LETZTE_USDCAD=$(tail -1 "$USDCAD_SIGNALS" | cut -d',' -f1)
    echo -e "  USDCAD Signals: ${GRUEN}✓${RESET} Letzter Eintrag: $LETZTE_USDCAD"
else
    echo -e "  USDCAD Signals: ${ROT}✗ Datei nicht gefunden!${RESET}"
fi

if [ -f "$USDJPY_SIGNALS" ]; then
    LETZTE_USDJPY=$(tail -1 "$USDJPY_SIGNALS" | cut -d',' -f1)
    echo -e "  USDJPY Signals: ${GRUEN}✓${RESET} Letzter Eintrag: $LETZTE_USDJPY"
else
    echo -e "  USDJPY Signals: ${ROT}✗ Datei nicht gefunden!${RESET}"
fi

# Abgeschlossene Trades zählen
USDCAD_CLOSES="$BASE_DIR/logs/paper_test128/USDCAD_closes.csv"
USDJPY_CLOSES="$BASE_DIR/logs/paper_test128/USDJPY_closes.csv"

USDCAD_N=0
USDJPY_N=0

if [ -f "$USDCAD_CLOSES" ]; then
    # Zeilen zählen minus Header
    USDCAD_N=$(( $(wc -l < "$USDCAD_CLOSES") - 1 ))
fi
if [ -f "$USDJPY_CLOSES" ]; then
    USDJPY_N=$(( $(wc -l < "$USDJPY_CLOSES") - 1 ))
fi

TOTAL_TRADES=$(( USDCAD_N + USDJPY_N ))
echo ""
echo -e "  Abgeschlossene Trades: ${FETT}USDCAD=$USDCAD_N | USDJPY=$USDJPY_N | Gesamt=$TOTAL_TRADES${RESET}"

if [ "$TOTAL_TRADES" -lt 30 ]; then
    echo -e "  ${GELB}⚠  Noch $((30 - TOTAL_TRADES)) Trades bis zur statistischen Signifikanz (min. 30)${RESET}"
fi

echo ""

# ── Schritt 2: KPI-Report generieren ────────────────────────────────────────
echo -e "${FETT}[2/4] KPI-Report generieren${RESET}"
echo "──────────────────────────────────────────"

cd "$BASE_DIR"

if [ -f "$VENV" ]; then
    PYTHON="$VENV"
else
    PYTHON="python3"
    echo -e "  ${GELB}Hinweis: venv nicht gefunden, nutze System-Python${RESET}"
fi

echo ""
$PYTHON reports/weekly_kpi_report.py --tage 7 --log_dir logs/paper_test128
echo ""

# ── Schritt 3: Closes dieser Woche anzeigen ─────────────────────────────────
echo -e "${FETT}[3/4] Trades dieser Woche${RESET}"
echo "──────────────────────────────────────────"

WOCHE_START=$(date -d "last monday" +"%Y-%m-%d" 2>/dev/null || date -v-Mon +"%Y-%m-%d")

echo "  Abgeschlossene Trades seit $WOCHE_START:"
echo ""

for DATEI in "$USDCAD_CLOSES" "$USDJPY_CLOSES"; do
    if [ -f "$DATEI" ]; then
        SYMBOL=$(basename "$DATEI" | cut -d'_' -f1)
        # Letzte 5 Einträge der Closes anzeigen
        RECENT=$(tail -5 "$DATEI" 2>/dev/null | grep -v "^time" | grep -v "^$")
        if [ -n "$RECENT" ]; then
            echo "  $SYMBOL:"
            echo "$RECENT" | while IFS=',' read -r ts sym richt sig prob reg rname paper modus entry sl tp htf ltf exit pips money grund dauer ticket; do
                FARBE=$GRUEN
                if (( $(echo "$money < 0" | bc -l 2>/dev/null || echo 0) )); then
                    FARBE=$ROT
                fi
                printf "    %-20s %-8s %+8s USD  %s\n" "$ts" "$richt" "$money" "$grund"
            done
        else
            echo "  $SYMBOL: Keine Trades diese Woche"
        fi
    fi
done

echo ""

# ── Schritt 4: Erinnerungen ─────────────────────────────────────────────────
echo -e "${FETT}[4/4] Checkliste${RESET}"
echo "──────────────────────────────────────────"
echo ""
echo -e "  ${FETT}Laptop-Checks (manuell auf Windows):${RESET}"
echo "  ☐  Beide PowerShell-Fenster noch offen? (USDCAD + USDJPY)"
echo "  ☐  MT5 Terminal verbunden? (grünes Symbol unten rechts)"
echo "  ☐  Laptop NICHT im Schlafmodus?"
echo "  ☐  Dashboard zeigt CONNECTED?"
echo ""
echo -e "  ${FETT}Ist heute der 1. Sonntag im Monat?${RESET}"
echo "  ☐  Falls ja: Frische Daten laden + Retraining prüfen:"
echo "      python data_loader.py  (auf Windows Laptop)"
echo "      python retraining.py --symbol USDCAD --sharpe_limit 0.5  (auf Server)"
echo "      python retraining.py --symbol USDJPY --sharpe_limit 0.5  (auf Server)"
echo ""
echo -e "  ${FETT}Roadmap.md aktualisieren:${RESET}"
echo "  → Wochen-Protokoll eintragen (Trades, Rendite, GO/NO-GO)"
echo "  → Datei: Roadmap.md, Abschnitt '📊 Wochen-Protokoll'"
echo ""
echo -e "${FETT}${BLAU}============================================================${RESET}"
echo -e "${FETT}  Nächste Sonntagsroutine: $(date -d 'next sunday' '+%d.%m.%Y' 2>/dev/null || echo 'nächsten Sonntag')${RESET}"
echo -e "${FETT}${BLAU}============================================================${RESET}"
echo ""
