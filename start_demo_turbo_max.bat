@echo off
REM =============================================================================
REM start_demo_turbo_max.bat – MAX Trade-Aktivität im Demo/Paper-Modus
REM
REM Ziel:
REM   - Sehr hohe Signal-/Trade-Frequenz fuer sichtbare Bewegung im Chart
REM   - Nur fuer Demo/Paper (nicht fuer Echtgeld)
REM
REM Einstellungen:
REM   - Mapping: long_prob
REM   - Long >= 0.50, Short <= 0.50 (maximal aktiv)
REM   - Regime-Quelle: market_regime_hmm
REM   - Regime-Filter: 0,1,2,3 (alle erlaubt)
REM   - Two-Stage-Kongruenzfilter: AUS
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================
echo   MT5 ML-Trading - DEMO TURBO MAX
echo   Maximaler Signal-/Trade-Durchsatz (Paper)
echo ================================================
echo.

call :ensure_var MT5_SERVER "MT5_SERVER"
if errorlevel 1 goto :env_error

call :ensure_var MT5_LOGIN "MT5_LOGIN"
if errorlevel 1 goto :env_error

call :ensure_var MT5_PASSWORD "MT5_PASSWORD"
if errorlevel 1 goto :env_error

echo [INFO] Verwendete Session-Variablen:
echo   MT5_SERVER=%MT5_SERVER%
echo   MT5_LOGIN=%MT5_LOGIN%
echo   MT5_PASSWORD=***
echo.

cd /d "C:\Users\Sebastian Setnescu\mt5_trading"

REM Fenster 1: USDCAD (v4) Turbo Max
start "MT5-DEMO-USDCAD-TURBO" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --paper_trading 1 --schwelle 0.50 --short_schwelle 0.50 --decision_mapping long_prob --regime_source market_regime_hmm --regime_filter 0,1,2,3 --two_stage_enable 1 --two_stage_kongruenz 0 --two_stage_ltf_timeframe M5 --two_stage_version v4 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY (v5) Turbo Max
start "MT5-DEMO-USDJPY-TURBO" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v5 --paper_trading 1 --schwelle 0.50 --short_schwelle 0.50 --decision_mapping long_prob --regime_source market_regime_hmm --regime_filter 0,1,2,3 --two_stage_enable 1 --two_stage_kongruenz 0 --two_stage_ltf_timeframe M5 --two_stage_version v5 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo ✅ Demo-Turbo-Max gestartet.
echo.
echo Hinweis:
echo - Sehr aggressiv: viele Signale/Trades moeglich.
echo - Nur Paper-Modus.
echo - Zum Stoppen: Ctrl+C in jedem Fenster.
echo.
pause
endlocal
exit /b 0

:ensure_var
set "var_name=%~1"
set "var_label=%~2"
call set "var_value=%%%var_name%%%"

if not "%var_value%"=="" goto :eof

echo [WARNUNG] %var_label% ist nicht gesetzt.
set /p "user_input=Bitte %var_label% jetzt eingeben: "

if "%user_input%"=="" (
	echo [FEHLER] %var_label% darf nicht leer sein.
	exit /b 1
)

set "%var_name%=%user_input%"
goto :eof

:env_error
echo.
echo [ABBRUCH] Umgebungsvariablen sind unvollstaendig.
echo Script wird beendet.
pause
endlocal
exit /b 1
