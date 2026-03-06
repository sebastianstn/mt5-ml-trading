@echo off
REM =============================================================================
REM start_both_traders.bat – Startet beide Paper-Trader parallel
REM
REM Ausführen auf Windows 11 Laptop:
REM     Doppelklick auf diese Datei ODER in PowerShell: .\start_both_traders.bat
REM
REM Voraussetzung: 
REM     - MT5 Terminal läuft und ist angemeldet
REM     - Virtuelle Umgebung existiert (venv/)
REM
REM WICHTIG:
REM     - Dieses Skript ist der "Baseline-Betrieb" (beide Symbole auf v4).
REM     - NICHT parallel zu start_shadow_compare.bat ausführen.
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================
echo   MT5 ML-Trading - Baseline Start
echo   USDCAD v4 + USDJPY v4 (Paper-Trading)
echo ================================================
echo.

echo [HINWEIS] Nicht parallel mit start_shadow_compare.bat starten!
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

REM Fenster 1: USDCAD (Two-Stage v4: H1+M5, M5-Takt, Long>=55%% / Short<=45%%)
start "MT5-Trader-USDCAD-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY (Two-Stage v4: H1+M5, M5-Takt, Long>=55%% / Short<=45%%)
start "MT5-Trader-USDJPY-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v4 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo ✅ Baseline-Betrieb gestartet!
echo.
echo Fenster 1: USDCAD (Two-Stage v4)
echo Fenster 2: USDJPY (Two-Stage v4)
echo.
echo Zum Stoppen: Ctrl+C in jedem Fenster druecken
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
