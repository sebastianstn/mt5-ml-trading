@echo off
REM =============================================================================
REM start_paper_trading.bat – Paper-Trading mit optimierten Parametern
REM
REM Ausfuehren auf Windows 11 Laptop:
REM     Doppelklick auf diese Datei ODER in PowerShell: .\start_paper_trading.bat
REM
REM Konfiguration (Stand: 08.03.2026):
REM     - Beide Symbole auf v4 (stabil)
REM     - Two-Stage: H1-Bias + M5-Entry
REM     - Schwelle: lockerer (Long >= 45%%, Short <= 55%%)
REM     - Regime-Filter: alle Regime erlaubt (0,1,2,3)
REM     - ATR-SL 1.5x mit Mindest-SL (4x Spread)
REM     - RRR 3:1 (TP=0.9%%, SL=0.3%%)
REM     - Neutraler H1-Bias darf M5-Entries zulassen
REM     - Kongruenzfilter deaktiviert fuer mehr Feedback im Paper-Lauf
REM     - Nach Start: 5 Kerzen nur beobachten, erst dann Trades erlauben
REM     - Demo-Live-Modus (paper_trading=0 auf Demo-Konto fuer PnL-Tracking)
REM
REM Voraussetzung:
REM     - MT5 Terminal laeuft und ist angemeldet
REM     - Virtuelle Umgebung existiert (venv/)
REM     - Modelle v4 nach models/ deployed
REM     - Umgebungsvariablen: MT5_SERVER, MT5_LOGIN, MT5_PASSWORD
REM       (setx MT5_SERVER "..." / setx MT5_LOGIN "..." / setx MT5_PASSWORD "...")
REM =============================================================================

setlocal EnableDelayedExpansion

REM ========================================================
REM Fest eingebettete MT5-Zugangsdaten
REM ========================================================
set "MT5_ACCOUNT_NAME=Sebastian Setnescu"
set "MT5_ACCOUNT_TYPE=Demo Allocations LTD"
set "MT5_SERVER=SwissquoteLtd-Server"
set "MT5_LOGIN=6202835"
set "MT5_PASSWORD=*0YsQqAk"
set "MT5_INVESTOR=ZcEpDl@8"

echo ================================================
echo   MT5 ML-Trading - Demo-Live (PnL-Tracking aktiv)
echo   USDCAD v4 + USDJPY v4
echo ================================================
echo.

echo [INFO] Verwendete Session-Variablen:
echo   Konto=%MT5_ACCOUNT_NAME% ^| %MT5_ACCOUNT_TYPE%
echo   MT5_SERVER=%MT5_SERVER%
echo   MT5_LOGIN=%MT5_LOGIN%
echo   MT5_PASSWORD=***
echo.

cd /d "C:\Users\Sebastian Setnescu\mt5_trading"

REM Fenster 1: USDCAD v4 (Two-Stage H1+M5, gelockert)
start "MT5-Demo-USDCAD-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --paper_trading 0 --schwelle 0.45 --short_schwelle 0.55 --decision_mapping long_prob --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY v4 (Two-Stage H1+M5, gelockert)
start "MT5-Demo-USDJPY-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v4 --paper_trading 0 --schwelle 0.45 --short_schwelle 0.55 --decision_mapping long_prob --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo [OK] Demo-Live-Trading gestartet (PnL-Tracking aktiv)!
echo.
echo   Fenster 1: USDCAD v4 (Two-Stage, alle Regime, lockerer)
echo   Fenster 2: USDJPY v4 (Two-Stage, alle Regime, lockerer)
echo.
echo   Zum Stoppen: Ctrl+C in jedem Fenster druecken.
echo.
pause
endlocal
exit /b 0
