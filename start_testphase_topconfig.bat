@echo off
REM =============================================================================
REM start_testphase_topconfig.bat – MT5 Paper-Testphase mit neuer Top-Konfiguration
REM
REM Ziel:
REM     Startet USDCAD + USDJPY im Paper-Modus mit der live-kompatiblen
REM     Top-Konfiguration aus dem aktuellen Sweep:
REM       - Schwelle: 0.54
REM       - Regime-Filter: 0,1,2
REM       - Two-Stage: H1 (HTF) + M5 (LTF), Version v4
REM       - ATR-SL: 1.5x
REM
REM Ausführen auf Windows 11 Laptop (MT5-Host):
REM     Doppelklick auf diese Datei
REM     ODER in PowerShell: .\start_testphase_topconfig.bat
REM
REM Voraussetzungen:
REM     - MT5 Terminal läuft und ist eingeloggt
REM     - C:\Users\Sebastian Setnescu\mt5_trading ist vorhanden
REM     - venv + Modelle v4 sind deployed
REM     - MT5_SERVER / MT5_LOGIN / MT5_PASSWORD sind gesetzt
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================
echo   MT5 Testphase - TOP-Konfiguration (Paper)
echo   USDCAD + USDJPY | Two-Stage v4
echo ================================================
echo.

call :ensure_var MT5_SERVER "MT5_SERVER"
if errorlevel 1 goto :env_error

call :ensure_var MT5_LOGIN "MT5_LOGIN"
if errorlevel 1 goto :env_error

call :ensure_var MT5_PASSWORD "MT5_PASSWORD"
if errorlevel 1 goto :env_error

cd /d "C:\Users\Sebastian Setnescu\mt5_trading"

echo [INFO] Starte USDCAD (Paper, Top-Config)...
start "MT5-Testphase-USDCAD-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --paper_trading 1 --schwelle 0.54 --decision_mapping class --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --heartbeat_log 1 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

timeout /t 5 /nobreak >nul

echo [INFO] Starte USDJPY (Paper, Top-Config)...
start "MT5-Testphase-USDJPY-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v4 --paper_trading 1 --schwelle 0.54 --decision_mapping class --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --heartbeat_log 1 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo [OK] Beide Trader laufen jetzt in der Testphase (Paper).
echo [INFO] Naechste Schritte:
echo   1) Dashboard in MT5 pruefen (kein dauerhaftes MISSING/STALE)
echo   2) Log-Sync nach Linux pruefen
echo   3) Nach 24-48h auf Linux auswerten: scripts\evaluate_mt5_testphase.py
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
echo [FEHLER] Pflicht-Umgebungsvariable fehlt. Abbruch.
echo Setze sie dauerhaft mit: setx VARIABLENNAME "WERT"
pause
endlocal
exit /b 1
