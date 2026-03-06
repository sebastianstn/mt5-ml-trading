@echo off
REM =============================================================================
REM start_shadow_compare.bat – Startet kontrollierten Shadow-Compare Betrieb
REM
REM Ausführen auf Windows 11 Laptop:
REM     Doppelklick auf diese Datei ODER in PowerShell: .\start_shadow_compare.bat
REM
REM Shadow-Setup:
REM     - USDCAD bleibt stabil auf Two-Stage v4
REM     - USDJPY läuft als Kandidat auf Two-Stage v5
REM
REM Voraussetzung:
REM     - MT5 Terminal läuft und ist angemeldet
REM     - Virtuelle Umgebung existiert (venv/)
REM     - Modelle v4/v5 wurden nach models/ deployed
REM     - Umgebungsvariablen gesetzt: MT5_SERVER, MT5_LOGIN, MT5_PASSWORD
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================
echo   MT5 ML-Trading - Shadow Compare Start
echo   USDCAD v4 (stabil) + USDJPY v5 (kandidat)
echo ================================================
echo.

call :ensure_var MT5_SERVER "MT5_SERVER"
if errorlevel 1 goto :env_error

call :ensure_var MT5_LOGIN "MT5_LOGIN"
if errorlevel 1 goto :env_error

call :ensure_var MT5_PASSWORD "MT5_PASSWORD"
if errorlevel 1 goto :env_error

echo.
echo [INFO] Verwendete Session-Variablen:
echo   MT5_SERVER=%MT5_SERVER%
echo   MT5_LOGIN=%MT5_LOGIN%
echo   MT5_PASSWORD=***
echo.
echo [HINWEIS] Falls du diese Werte dauerhaft speichern willst, nutze einmalig:
echo   setx MT5_SERVER "..."
echo   setx MT5_LOGIN "..."
echo   setx MT5_PASSWORD "..."
echo.

cd /d "C:\Users\Sebastian Setnescu\mt5_trading"

REM Fenster 1: USDCAD stabil auf v4 (strengeres Mapping: Long>=55%, Short<=45%)
start "MT5-Shadow-USDCAD-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY als v5-Kandidat (strengeres Mapping: Long>=55%, Short<=45%)
start "MT5-Shadow-USDJPY-v5" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v5 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v5 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo ✅ Shadow-Compare gestartet!
echo.
echo Fenster 1: USDCAD - Two-Stage v4 (Kontrollgruppe)
echo Fenster 2: USDJPY - Two-Stage v5 (Testgruppe)
echo.
echo Hinweis: Beide laufen im Paper-Modus.
echo Zum Stoppen: Ctrl+C in jedem Fenster druecken.
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
