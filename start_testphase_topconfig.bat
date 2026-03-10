@echo off
REM start_testphase_topconfig.bat - stabile Startdatei fuer Windows
setlocal

set "BASE_DIR=C:\Users\Sebastian Setnescu\mt5_trading"
set "PYTHON_EXE=%BASE_DIR%\venv\Scripts\python.exe"
set "TRADER_SCRIPT=%BASE_DIR%\live\live_trader.py"

REM ========================================================
REM Fest eingebettete MT5-Zugangsdaten fuer die Top-Konfiguration
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
    echo [FEHLER] Python nicht gefunden: %PYTHON_EXE%
    pause
    exit /b 1
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

echo ================================================
echo   MT5 Testphase - TOP-Konfiguration (Paper)
echo   USDCAD + USDJPY ^| Two-Stage v4
echo   Schwelle=45%% ^| TP=0.6%% ^| SL=0.3%% ^| Horizon=24
echo   Cooldown=3 Bars ^| ATR-SL 1.5x ^| Regime=0,1,2,3
echo   Konto: %MT5_ACCOUNT_NAME% ^| %MT5_ACCOUNT_TYPE%
echo   Server: %MT5_SERVER% ^| Login: %MT5_LOGIN%
echo ================================================

echo [INFO] Starte USDCAD...
start "MT5-Testphase-USDCAD-v4" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDCAD --version v4 --paper_trading 1 --schwelle 0.45 --decision_mapping class --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

timeout /t 5 /nobreak >nul

echo [INFO] Starte USDJPY...
start "MT5-Testphase-USDJPY-v4" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDJPY --version v4 --paper_trading 1 --schwelle 0.45 --decision_mapping class --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --two_stage_kongruenz 0 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

echo.
echo [OK] Beide Trader gestartet.
pause
exit /b 0

:env_error
echo [FEHLER] MT5_SERVER/MT5_LOGIN/MT5_PASSWORD muessen gesetzt sein.
pause
exit /b 1
