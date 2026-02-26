#!/bin/bash
# run_pipeline_v2_v3.sh – Komplette Pipeline für Option A (v2) und Option B (v3)
#
# Was dieses Skript tut:
#   1. Wartet auf laufendes v2-Training (PID aus train_v2.log)
#   2. Walk-Forward-Analyse v2 (Stabilitäts-Check)
#   3. Backtest v2 (TP=SL=0.3%, Horizon=10)
#   4. Training v3 (TP=SL=0.15%, Horizon=5)
#   5. Walk-Forward-Analyse v3
#   6. Backtest v3 (TP=SL=0.15%, Horizon=5)
#   7. Vergleichs-Zusammenfassung
#
# Ausführen:
#   source venv/bin/activate
#   bash run_pipeline_v2_v3.sh | tee pipeline.log
#
# Läuft auf: Linux-Server

set -e  # Skript beenden bei Fehler

BASE_DIR="/mnt/1T-Data/XGBoost-LightGBM"
cd "$BASE_DIR"
source venv/bin/activate

TIMESTAMP() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "Pipeline v2 + v3 – gestartet: $(TIMESTAMP)"
echo "============================================================"

# ============================================================
# Schritt 1: Warte auf v2-Training (läuft bereits im Hintergrund)
# ============================================================
V2_PID=$(ps aux | grep "train_model.py --symbol alle --version v2" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$V2_PID" ]; then
    echo ""
    echo "⏳ Warte auf v2-Training (PID: $V2_PID) ..."
    wait "$V2_PID" 2>/dev/null || true  # Fehler ignorieren (Prozess könnte schon fertig sein)
    echo "✓ v2-Training abgeschlossen: $(TIMESTAMP)"
else
    echo "✓ v2-Training bereits fertig (kein laufender Prozess)"
fi

# Prüfen ob v2-Modelle vorhanden sind
echo ""
echo "Prüfe v2-Modelle ..."
MISSING_V2=0
for SYM in EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD; do
    if [ ! -f "models/lgbm_${SYM,,}_v2.pkl" ]; then
        echo "  ✗ Fehlt: models/lgbm_${SYM,,}_v2.pkl"
        MISSING_V2=1
    else
        echo "  ✓ Vorhanden: models/lgbm_${SYM,,}_v2.pkl"
    fi
done

if [ "$MISSING_V2" -eq 1 ]; then
    echo ""
    echo "⚠️  Einige v2-Modelle fehlen! Training neu starten ..."
    python train_model.py --symbol alle --version v2 --trials 50
fi

# ============================================================
# Schritt 2: Walk-Forward-Analyse v2
# ============================================================
echo ""
echo "============================================================"
echo "Walk-Forward-Analyse v2 – Start: $(TIMESTAMP)"
echo "============================================================"
python walk_forward.py --symbol alle --version v2 | tee walk_forward_v2.log
echo "✓ Walk-Forward v2 abgeschlossen: $(TIMESTAMP)"

# ============================================================
# Schritt 3: Backtest v2 (Horizon=10, TP=SL=0.3%)
# ============================================================
echo ""
echo "============================================================"
echo "Backtest v2 (Horizon=10, TP=SL=0.3%) – Start: $(TIMESTAMP)"
echo "============================================================"
python backtest/backtest.py --symbol alle --version v2 --horizon 10 --schwelle 0.55 | tee backtest_v2.log
echo "✓ Backtest v2 abgeschlossen: $(TIMESTAMP)"

# ============================================================
# Schritt 4: Training v3 (TP=SL=0.15%, Horizon=5)
# ============================================================
echo ""
echo "============================================================"
echo "Training v3 (TP=SL=0.15%, Horizon=5) – Start: $(TIMESTAMP)"
echo "============================================================"
python train_model.py --symbol alle --version v3 --trials 50 | tee train_v3.log
echo "✓ Training v3 abgeschlossen: $(TIMESTAMP)"

# ============================================================
# Schritt 5: Walk-Forward-Analyse v3
# ============================================================
echo ""
echo "============================================================"
echo "Walk-Forward-Analyse v3 – Start: $(TIMESTAMP)"
echo "============================================================"
python walk_forward.py --symbol alle --version v3 | tee walk_forward_v3.log
echo "✓ Walk-Forward v3 abgeschlossen: $(TIMESTAMP)"

# ============================================================
# Schritt 6: Backtest v3 (Horizon=5, TP=SL=0.15%)
# ============================================================
echo ""
echo "============================================================"
echo "Backtest v3 (Horizon=5, TP=SL=0.15%) – Start: $(TIMESTAMP)"
echo "============================================================"
python backtest/backtest.py --symbol alle --version v3 --horizon 5 --tp_pct 0.0015 --sl_pct 0.0015 --schwelle 0.55 | tee backtest_v3.log
echo "✓ Backtest v3 abgeschlossen: $(TIMESTAMP)"

# ============================================================
# Schritt 7: Vergleichs-Zusammenfassung
# ============================================================
echo ""
echo "============================================================"
echo "VERGLEICH – v1 (Original) vs v2 (Horizon=10) vs v3 (TP=0.15%)"
echo "============================================================"

echo ""
echo "v1 (TP=SL=0.30%, Horizon=5):"
if [ -f "backtest/backtest_zusammenfassung.csv" ]; then
    python -c "
import pandas as pd
df = pd.read_csv('backtest/backtest_zusammenfassung.csv')
print(df[['symbol','n_trades','gesamtrendite_pct','win_rate_pct','gewinnfaktor','sharpe_ratio']].to_string(index=False))
"
fi

echo ""
echo "v2 (TP=SL=0.30%, Horizon=10):"
if [ -f "backtest/backtest_zusammenfassung_v2.csv" ]; then
    python -c "
import pandas as pd
df = pd.read_csv('backtest/backtest_zusammenfassung_v2.csv')
print(df[['symbol','n_trades','gesamtrendite_pct','win_rate_pct','gewinnfaktor','sharpe_ratio']].to_string(index=False))
"
fi

echo ""
echo "v3 (TP=SL=0.15%, Horizon=5):"
if [ -f "backtest/backtest_zusammenfassung_v3.csv" ]; then
    python -c "
import pandas as pd
df = pd.read_csv('backtest/backtest_zusammenfassung_v3.csv')
print(df[['symbol','n_trades','gesamtrendite_pct','win_rate_pct','gewinnfaktor','sharpe_ratio']].to_string(index=False))
"
fi

echo ""
echo "============================================================"
echo "Pipeline ABGESCHLOSSEN: $(TIMESTAMP)"
echo "============================================================"
echo "Plots:  plots/SYMBOL_backtest_equity.png"
echo "Trades: backtest/SYMBOL_trades.csv"
echo "Logs:   train_v2.log, train_v3.log, backtest_v2.log, backtest_v3.log"
echo "============================================================"
