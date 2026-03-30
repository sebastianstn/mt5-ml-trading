@echo off
REM =============================================================================
REM sunday_check.bat – Woechentliche Sonntagsroutine (Windows 11)
REM
REM Windows-Aequivalent zu scripts\sunday_check.sh
REM
REM Fuehrt alle woechentlichen KPI-Checks automatisch aus und zeigt
REM eine fertige Zusammenfassung fuer das Wochen-Protokoll in der Roadmap.
REM
REM Verwendung: Doppelklick oder in PowerShell: .\scripts\sunday_check.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0\.."
pushd "%ROOT_DIR%"
set "ROOT_DIR=%CD%"
popd

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

echo.
echo ============================================================
echo   MT5 Trading – Sonntagsroutine ^| %date%
echo ============================================================
echo.

REM -- Schritt 1: Letzte Sync-Zeit pruefen --
echo [1/4] Server-Sync pruefen
echo ------------------------------------------

set "USDCAD_SIGNALS=%ROOT_DIR%\logs\USDCAD_signals.csv"
set "USDJPY_SIGNALS=%ROOT_DIR%\logs\USDJPY_signals.csv"

if exist "%USDCAD_SIGNALS%" (
    for /f "usebackq" %%L in (`powershell -NoProfile -Command "(Get-Content '%USDCAD_SIGNALS%' -Tail 1).Split(',')[0]"`) do (
        echo   USDCAD Signals: OK - Letzter Eintrag: %%L
    )
) else (
    echo   USDCAD Signals: FEHLT - Datei nicht gefunden!
)

if exist "%USDJPY_SIGNALS%" (
    for /f "usebackq" %%L in (`powershell -NoProfile -Command "(Get-Content '%USDJPY_SIGNALS%' -Tail 1).Split(',')[0]"`) do (
        echo   USDJPY Signals: OK - Letzter Eintrag: %%L
    )
) else (
    echo   USDJPY Signals: FEHLT - Datei nicht gefunden!
)

REM Abgeschlossene Trades zaehlen
set "USDCAD_N=0"
set "USDJPY_N=0"

if exist "%ROOT_DIR%\logs\USDCAD_closes.csv" (
    for /f %%C in ('powershell -NoProfile -Command "((Get-Content '%ROOT_DIR%\logs\USDCAD_closes.csv').Count - 1)"') do set "USDCAD_N=%%C"
)
if exist "%ROOT_DIR%\logs\USDJPY_closes.csv" (
    for /f %%C in ('powershell -NoProfile -Command "((Get-Content '%ROOT_DIR%\logs\USDJPY_closes.csv').Count - 1)"') do set "USDJPY_N=%%C"
)

set /a "TOTAL_TRADES=USDCAD_N + USDJPY_N"
echo.
echo   Abgeschlossene Trades: USDCAD=%USDCAD_N% ^| USDJPY=%USDJPY_N% ^| Gesamt=%TOTAL_TRADES%

if %TOTAL_TRADES% LSS 30 (
    set /a "MISSING=30 - TOTAL_TRADES"
    echo   Noch !MISSING! Trades bis zur statistischen Signifikanz (min. 30^)
)
echo.

REM -- Schritt 2: KPI-Report generieren --
echo [2/4] KPI-Report generieren
echo ------------------------------------------
echo.

cd /d "%ROOT_DIR%"
"%PYTHON_BIN%" reports\weekly_kpi_report.py --tage 7 --log_dir logs
echo.

REM -- Schritt 3: Trades dieser Woche --
echo [3/4] Trades dieser Woche
echo ------------------------------------------
echo.

echo   Letzte 5 Closes pro Symbol:
echo.
if exist "%ROOT_DIR%\logs\USDCAD_closes.csv" (
    echo   USDCAD:
    powershell -NoProfile -Command "Get-Content '%ROOT_DIR%\logs\USDCAD_closes.csv' -Tail 5 | ForEach-Object { Write-Output ('    ' + $_) }"
    echo.
)
if exist "%ROOT_DIR%\logs\USDJPY_closes.csv" (
    echo   USDJPY:
    powershell -NoProfile -Command "Get-Content '%ROOT_DIR%\logs\USDJPY_closes.csv' -Tail 5 | ForEach-Object { Write-Output ('    ' + $_) }"
    echo.
)

REM -- Schritt 4: Erinnerungen --
echo [4/4] Checkliste
echo ------------------------------------------
echo.
echo   Laptop-Checks:
echo     [ ]  Beide Trader-Fenster noch offen? (USDCAD + USDJPY)
echo     [ ]  MT5 Terminal verbunden? (gruenes Symbol unten rechts)
echo     [ ]  Laptop NICHT im Schlafmodus?
echo     [ ]  Dashboard zeigt CONNECTED?
echo.
echo   Ist heute der 1. Sonntag im Monat?
echo     [ ]  Falls ja: Frische Daten laden + Retraining pruefen:
echo          python data_loader.py
echo          python retraining.py --symbol USDCAD --sharpe_limit 0.5
echo          python retraining.py --symbol USDJPY --sharpe_limit 0.5
echo.
echo   Roadmap.md aktualisieren:
echo     Wochen-Protokoll eintragen (Trades, Rendite, GO/NO-GO)
echo.
echo ============================================================
echo.
pause
endlocal
exit /b 0
