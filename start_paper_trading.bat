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
REM     - Schwelle: Long >= 55%%, Short <= 45%%
REM     - Regime-Filter: nur Trends (1,2) – Choppy ausgeschlossen
REM     - ATR-SL 1.5x mit Mindest-SL (4x Spread)
REM     - RRR 3:1 (TP=0.9%%, SL=0.3%%)
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

echo ================================================
echo   MT5 ML-Trading - Demo-Live (PnL-Tracking aktiv)
echo   USDCAD v4 + USDJPY v4
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

REM Fenster 1: USDCAD v4 (Two-Stage H1+M5, optimiert)
start "MT5-Demo-USDCAD-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --paper_trading 0 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY v4 (Two-Stage H1+M5, optimiert)
start "MT5-Demo-USDJPY-v4" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v4 --paper_trading 0 --schwelle 0.55 --short_schwelle 0.45 --decision_mapping long_prob --regime_filter 1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server %MT5_SERVER% --mt5_login %MT5_LOGIN% --mt5_password %MT5_PASSWORD%"

echo.
echo [OK] Demo-Live-Trading gestartet (PnL-Tracking aktiv)!
echo.
echo   Fenster 1: USDCAD v4 (Two-Stage, Regime 1+2)
echo   Fenster 2: USDJPY v4 (Two-Stage, Regime 1+2)
echo.
echo   Zum Stoppen: Ctrl+C in jedem Fenster druecken.
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
