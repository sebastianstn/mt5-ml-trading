ARCHIV – MT5 ML-Trading-System
================================
Erstellt: 2026-03-04
Grund: Projekt-Cleanup (Phase 7 – Mittlere Stufe)

INHALT
======

2026-03-04_models_old/
  46 alte Modell-Dateien (.pkl):
  - Nicht-aktive Symbole: AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCHF (alle Versionen)
  - USDCAD/USDJPY alte Versionen: v1, v2, v3, H4_v1

  Verbleibend in models/:
  - USDCAD: v4 (lgbm, xgb, ensemble) + M15/M30/M60_v1 (je lgbm+xgb) = 9
  - USDJPY: v4 (lgbm, xgb, ensemble) + M15/M30/M60_v1 (je lgbm+xgb) = 9
  → Gesamt: 18 aktive Modelle

2026-03-04_data_old/
  28 gelabelte CSV-Dateien:
  - Nicht-aktive Symbole: AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCHF (H1_labeled + Versionen)
  - USDCAD/USDJPY alte Versionen: H1_labeled (v1), _v2, _v3, H4_labeled

  Verbleibend in data/:
  - USDCAD: H1_labeled_v4, M15/M30/M60_labeled = 4
  - USDJPY: H1_labeled_v4, M15/M30/M60_labeled = 4
  → Gesamt: 8 aktive CSVs

2026-03-04_scripts_old/
  2 veraltete Runner-Scripts:
  - run_pipeline_v2_v3.sh (verwendete altes venv, alle 7 Symbole)
  - scripts/install_pre_commit_hook.sh (nicht in Verwendung)

  Verbleibend im Root + scripts/:
  - deploy_to_laptop.sh
  - run_intraday_pipelines.sh (M15/M30/M60, USDCAD+USDJPY)
  - run_m15_pipeline.sh (M15, USDCAD+USDJPY)
  - run_two_stage_pipeline.sh (HTF/LTF, Test-Phase)
  - start_both_traders.bat (Windows)
  → Gesamt: 5 aktive Scripts

2026-03-04_venv_old/
  Duplikat der virtuellen Umgebung (venv/).
  Aktiv ist: .venv/

2026-03-04_logs_old/
  (leer – Logs wurden nach logs/train/ verschoben, nicht ins Archiv)

WIEDERHERSTELLUNG
=================
Falls ein Archiv-Artefakt wieder benötigt wird:
  cp _archive/2026-03-04_<kategorie>/<dateiname> <ziel>/

LÖSCHUNG
========
Empfohlung: Archiv mindestens 3 Monate behalten.
Nach erfolgreicher Phase-7-Eskalation (12 GO-Wochen), frühestens ab 2026-06-01 löschbar.

DOKUMENTATION
=============
Details zum Cleanup-Plan: Conversation Summary, 2026-03-04
