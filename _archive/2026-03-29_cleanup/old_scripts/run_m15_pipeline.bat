@echo off
REM =============================================================================
REM run_m15_pipeline.bat – M15-Pipeline (Windows 11)
REM
REM Windows-Aequivalent zu run_m15_pipeline.sh
REM
REM Voraussetzung: M15-Rohdaten wurden mit data_loader.py erzeugt
REM                und liegen in data\SYMBOL_M15.csv
REM
REM Ausfuehren auf Windows 11:
REM     Doppelklick oder in PowerShell: .\run_m15_pipeline.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

set "SYMBOLS=USDCAD USDJPY"
set "VERSION=v1"
set "TRIALS=50"
set "SCHWELLE=0.60"
set "REGIME_FILTER=1,2"

REM Python-Interpreter pruefen
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python-Interpreter nicht gefunden: %PYTHON_BIN%
    echo Bitte zuerst virtuelle Umgebung einrichten.
    pause
    exit /b 1
)

REM Rohdaten pruefen
for %%S in (%SYMBOLS%) do (
    if not exist "%ROOT_DIR%\data\%%S_M15.csv" (
        echo [FEHLER] Fehlende M15-Rohdaten: data\%%S_M15.csv
        echo Bitte ausfuehren: python data_loader.py --timeframe M15
        pause
        exit /b 1
    )
)

echo [1/6] Feature Engineering (M15)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\features\feature_engineering.py" --symbol %%S --timeframe M15
    if errorlevel 1 (
        echo [FEHLER] Feature Engineering fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [2/6] Labeling (M15)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\features\labeling.py" --symbol %%S --version %VERSION% --timeframe M15
    if errorlevel 1 (
        echo [FEHLER] Labeling fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [3/6] Training (M15, Optuna %TRIALS% Trials)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\train_model.py" --symbol %%S --version %VERSION% --timeframe M15 --trials %TRIALS%
    if errorlevel 1 (
        echo [FEHLER] Training fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [4/6] Walk-Forward (M15)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\walk_forward.py" --symbol %%S --version %VERSION% --timeframe M15
    if errorlevel 1 (
        echo [FEHLER] Walk-Forward fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [5/6] Backtest (M15)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\backtest\backtest.py" --symbol %%S --version %VERSION% --timeframe M15 --schwelle %SCHWELLE% --regime_filter %REGIME_FILTER%
    if errorlevel 1 (
        echo [FEHLER] Backtest fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [6/6] KPI-Report (M15)
"%PYTHON_BIN%" "%ROOT_DIR%\reports\weekly_kpi_report.py" --timeframe M15 --tage 7
if errorlevel 1 (
    echo [WARNUNG] KPI-Report fehlgeschlagen - nicht kritisch.
)

echo.
echo Fertig. Wichtige Artefakte:
echo   - backtest\USDCAD_M15_trades.csv
echo   - backtest\USDJPY_M15_trades.csv
echo   - reports\weekly_kpi_report_M15.md
echo.
pause
endlocal
exit /b 0
