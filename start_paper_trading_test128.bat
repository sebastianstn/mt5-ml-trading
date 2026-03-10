@echo off
REM =============================================================================
REM start_paper_trading_test128.bat – Paper-Trading für Test 128
REM
REM Läuft auf: Windows 11 Laptop (NICHT auf dem Linux-Server)
REM
REM Backtest-Sieger aus 50er Feintuning:
REM   Test 128 = ZoneC s0.59 h18
REM
REM 1:1 übernehmbare Parameter in live_trader.py:
REM   - Schwelle: gelockert auf 0.45 fuer mehr Live-Feedback
REM   - Regime-Filter: 0,1,2,3
REM   - ATR-SL: 1.5x
REM   - TP/SL: 0.6%% / 0.3%%
REM   - Cooldown: 3 Bars
REM   - Nach Start: 5 Kerzen nur beobachten
REM   - Two-Stage: H1 -> M5, Version v4
REM   - Neutraler H1-Bias darf M5-Entries passieren
REM   - Regime-Quelle: market_regime_hmm (hmmlearn-basiert, falls verfügbar)
REM
REM Wichtiger Unterschied:
REM   - Backtest-Horizon = 18 ist im Live-Trader aktuell NICHT als CLI-Parameter exposed.
REM   - Diese Startdatei bildet Test 128 daher bestmöglich ab, aber nicht 100%% identisch.
REM
REM Voraussetzung:
REM   - MT5 Terminal läuft und ist eingeloggt
REM   - Modelle auf dem Windows-Laptop vorhanden
REM   - MT5_SERVER / MT5_LOGIN / MT5_PASSWORD gesetzt oder beim Start eingegeben
REM =============================================================================

setlocal

set "BASE_DIR=C:\Users\Sebastian Setnescu\mt5_trading"
set "PYTHON_EXE=%BASE_DIR%\venv\Scripts\python.exe"
set "TRADER_SCRIPT=%BASE_DIR%\live\live_trader.py"
set "TEST128_LOG_DIR=%BASE_DIR%\logs\paper_test128"

REM ========================================================
REM Fest eingebettete MT5-Zugangsdaten fuer Test 128
REM Hinweis: Investor-Passwort wird aktuell von live_trader.py
REM nicht verwendet, bleibt hier aber zur Referenz hinterlegt.
REM ========================================================
set "MT5_ACCOUNT_NAME=Sebastian Setnescu"
set "MT5_ACCOUNT_TYPE=Demo Allocations LTD"
set "MT5_SERVER=SwissquoteLtd-Server"
set "MT5_LOGIN=6202835"
set "MT5_PASSWORD=*0YsQqAk"
set "MT5_INVESTOR=ZcEpDl@8"

if not exist "%BASE_DIR%" (
    echo [FEHLER] Projektordner nicht gefunden: %BASE_DIR%
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo [WARNUNG] venv-Python nicht gefunden: %PYTHON_EXE%
    echo [INFO] Fallback auf System-Python aus PATH ...
    set "PYTHON_EXE=python"
)

if not exist "%TRADER_SCRIPT%" (
    echo [FEHLER] Script nicht gefunden: %TRADER_SCRIPT%
    pause
    exit /b 1
)

if "%MT5_SERVER%"=="" goto :env_error
if "%MT5_LOGIN%"=="" goto :env_error
if "%MT5_PASSWORD%"=="" goto :env_error

cd /d "%BASE_DIR%"

if not exist "%TEST128_LOG_DIR%" (
    mkdir "%TEST128_LOG_DIR%"
)

set "MT5_TRADING_LOG_DIR=%TEST128_LOG_DIR%"

echo ========================================================
echo   MT5 Paper-Trade - Test 128 vorbereitet
echo   USDCAD + USDJPY ^| Two-Stage v4 ^| M5
echo   Schwelle=45%% ^| TP=0.6%% ^| SL=0.3%%
echo   Cooldown=3 Bars ^| ATR-SL 1.5x ^| Regime=0,1,2,3 ^| Quelle=HMM
echo   Konto: %MT5_ACCOUNT_NAME% ^| %MT5_ACCOUNT_TYPE%
echo   Server: %MT5_SERVER% ^| Login: %MT5_LOGIN%
echo   Log-Ordner: %TEST128_LOG_DIR%
echo   Python: %PYTHON_EXE%
echo   Hinweis: Horizon 18 ist im Live-Trader derzeit nicht direkt setzbar
echo ========================================================

echo [INFO] Starte USDCAD im Paper-Modus...
start "MT5-Paper-USDCAD-Test128" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDCAD --version v4 --paper_trading 1 --schwelle 0.45 --decision_mapping class --regime_source market_regime_hmm --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --kill_switch_dd 0.15 --kapital_start 10000 --log_subdir paper_test128 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

timeout /t 5 /nobreak >nul

echo [INFO] Starte USDJPY im Paper-Modus...
start "MT5-Paper-USDJPY-Test128" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDJPY --version v4 --paper_trading 1 --schwelle 0.45 --decision_mapping class --regime_source market_regime_hmm --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --kill_switch_dd 0.15 --kapital_start 10000 --log_subdir paper_test128 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

echo.
echo [OK] Test 128 Paper-Trade wurde gestartet.
echo.
echo   Fenster 1: USDCAD Test 128
echo   Fenster 2: USDJPY Test 128
echo.
echo   Logs werden in %TEST128_LOG_DIR% gespeichert.
echo   Dateien: USDCAD_signals.csv, USDJPY_signals.csv, optional *_closes.csv
echo   Zum Stoppen: Ctrl+C in jedem Fenster druecken.
echo.
pause
exit /b 0

:env_error
echo [FEHLER] MT5_SERVER/MT5_LOGIN/MT5_PASSWORD muessen gesetzt sein.
pause
exit /b 1
