@echo off
REM =============================================================================
REM run_phase7_autocheck.bat – Phase-7 Auto-Check Runner (Windows 11)
REM
REM Windows-Aequivalent zu run_phase7_autocheck.sh
REM
REM Fuehrt zyklisch Sync-Verifikation + Daily-Dashboard aus.
REM
REM Verwendung:
REM     run_phase7_autocheck.bat              (Standard: alle 30 Min)
REM     run_phase7_autocheck.bat 15           (alle 15 Min)
REM
REM Stoppen mit: Ctrl+C
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

REM Standard-Intervall: 30 Minuten
set "INTERVAL_MINUTES=30"
if not "%~1"=="" set "INTERVAL_MINUTES=%~1"

set "LOG_DIR_ARG=logs"
if not "%~2"=="" set "LOG_DIR_ARG=%~2"

set "SYMBOLS_ARG=USDCAD,USDJPY"
if not "%~3"=="" set "SYMBOLS_ARG=%~3"

set "TIMEFRAME_ARG=M5_TWO_STAGE"
if not "%~4"=="" set "TIMEFRAME_ARG=%~4"

set "MAX_AGE_MINUTES=10"
if not "%~5"=="" set "MAX_AGE_MINUTES=%~5"

set "REPORTS_DIR=%ROOT_DIR%\reports"
set "CHECK_LOG=%REPORTS_DIR%\phase7_autocheck.log"
set "LATEST_STATUS=%REPORTS_DIR%\phase7_autocheck_latest.txt"

if not exist "%REPORTS_DIR%" mkdir "%REPORTS_DIR%"

REM Python-Interpreter pruefen
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python-Interpreter nicht gefunden: %PYTHON_BIN%
    pause
    exit /b 1
)

REM Intervall in Sekunden umrechnen
set /a "INTERVAL_SECONDS=%INTERVAL_MINUTES% * 60"

echo [INFO] Starte Phase-7 Auto-Check: alle %INTERVAL_MINUTES% Minute(n).
echo [INFO] Stoppen mit: Ctrl+C

:loop

echo.
echo ================================================================
echo [%date% %time%] PHASE7 AUTO-CHECK
echo LogDir=%LOG_DIR_ARG% ^| Symbols=%SYMBOLS_ARG% ^| MaxAge=%MAX_AGE_MINUTES%min ^| TF=%TIMEFRAME_ARG%
echo ================================================================

set "VERIFY_RC=0"
set "DASHBOARD_RC=0"

"%PYTHON_BIN%" "%ROOT_DIR%\scripts\verify_live_log_sync.py" --log_dir %LOG_DIR_ARG% --symbols %SYMBOLS_ARG% --max_age_minutes %MAX_AGE_MINUTES% --check_watchdog
if errorlevel 1 set "VERIFY_RC=1"

"%PYTHON_BIN%" "%ROOT_DIR%\reports\daily_phase7_dashboard.py" --log_dir %LOG_DIR_ARG% --hours 24 --timeframe %TIMEFRAME_ARG%
if errorlevel 1 set "DASHBOARD_RC=1"

REM Status-Datei schreiben
(
    echo timestamp=%date% %time%
    echo verify_exit_code=%VERIFY_RC%
    echo dashboard_exit_code=%DASHBOARD_RC%
) > "%LATEST_STATUS%"

if "%VERIFY_RC%"=="0" if "%DASHBOARD_RC%"=="0" (
    echo [%date% %time%] Ergebnis: OK
) else (
    echo [%date% %time%] Ergebnis: NICHT_OK (verify=%VERIFY_RC%, dashboard=%DASHBOARD_RC%)
)

echo [INFO] Naechster Check in %INTERVAL_MINUTES% Minute(n)...
timeout /t %INTERVAL_SECONDS% /nobreak >nul
goto loop

endlocal
exit /b 0
