@echo off
REM =============================================================================
REM run_intraday_pipelines.bat – Intraday-Pipelines M15, M30, M60 (Windows 11)
REM
REM Windows-Aequivalent zu run_intraday_pipelines.sh
REM
REM Voraussetzung: Rohdaten data\SYMBOL_<TF>.csv vorhanden
REM
REM Ausfuehren auf Windows 11:
REM     Doppelklick oder in PowerShell: .\run_intraday_pipelines.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

set "SYMBOLS=USDCAD USDJPY"
set "TIMEFRAMES=M15 M30 M60"
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
for %%T in (%TIMEFRAMES%) do (
    for %%S in (%SYMBOLS%) do (
        if not exist "%ROOT_DIR%\data\%%S_%%T.csv" (
            echo [FEHLER] Fehlende Rohdaten: data\%%S_%%T.csv
            echo Bitte zuerst auf Windows exportieren: python data_loader.py --symbol %%S --timeframe %%T
            pause
            exit /b 1
        )
    )
)

for %%T in (%TIMEFRAMES%) do (
    echo.
    echo ==============================================================
    echo Starte Pipeline fuer Timeframe: %%T
    echo ==============================================================

    echo [1/6] Feature Engineering (%%T)
    for %%S in (%SYMBOLS%) do (
        echo   -^> %%S
        "%PYTHON_BIN%" "%ROOT_DIR%\features\feature_engineering.py" --symbol %%S --timeframe %%T
        if errorlevel 1 (
            echo [FEHLER] Feature Engineering fuer %%S %%T fehlgeschlagen!
            pause
            exit /b 1
        )
    )

    echo [2/6] Labeling (%%T)
    for %%S in (%SYMBOLS%) do (
        echo   -^> %%S
        "%PYTHON_BIN%" "%ROOT_DIR%\features\labeling.py" --symbol %%S --version %VERSION% --timeframe %%T
        if errorlevel 1 (
            echo [FEHLER] Labeling fuer %%S %%T fehlgeschlagen!
            pause
            exit /b 1
        )
    )

    echo [3/6] Training (%%T, Optuna %TRIALS% Trials)
    for %%S in (%SYMBOLS%) do (
        echo   -^> %%S
        "%PYTHON_BIN%" "%ROOT_DIR%\train_model.py" --symbol %%S --version %VERSION% --timeframe %%T --trials %TRIALS%
        if errorlevel 1 (
            echo [FEHLER] Training fuer %%S %%T fehlgeschlagen!
            pause
            exit /b 1
        )
    )

    echo [4/6] Walk-Forward (%%T)
    for %%S in (%SYMBOLS%) do (
        echo   -^> %%S
        "%PYTHON_BIN%" "%ROOT_DIR%\walk_forward.py" --symbol %%S --version %VERSION% --timeframe %%T
        if errorlevel 1 (
            echo [FEHLER] Walk-Forward fuer %%S %%T fehlgeschlagen!
            pause
            exit /b 1
        )
    )

    echo [5/6] Backtest (%%T)
    for %%S in (%SYMBOLS%) do (
        echo   -^> %%S
        "%PYTHON_BIN%" "%ROOT_DIR%\backtest\backtest.py" --symbol %%S --version %VERSION% --timeframe %%T --schwelle %SCHWELLE% --regime_filter %REGIME_FILTER%
        if errorlevel 1 (
            echo [FEHLER] Backtest fuer %%S %%T fehlgeschlagen!
            pause
            exit /b 1
        )
    )

    echo [6/6] KPI-Report (%%T)
    "%PYTHON_BIN%" "%ROOT_DIR%\reports\weekly_kpi_report.py" --timeframe %%T --tage 7
    if errorlevel 1 (
        echo [WARNUNG] KPI-Report fuer %%T fehlgeschlagen - nicht kritisch.
    )
)

echo.
echo Fertig. Wichtige Artefakte pro Timeframe:
echo   - backtest\USDCAD_^<TF^>_trades.csv
echo   - backtest\USDJPY_^<TF^>_trades.csv
echo   - reports\weekly_kpi_report_^<TF^>.md
echo.
pause
endlocal
exit /b 0
