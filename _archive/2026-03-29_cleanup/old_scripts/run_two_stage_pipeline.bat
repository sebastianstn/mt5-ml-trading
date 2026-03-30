@echo off
REM =============================================================================
REM run_two_stage_pipeline.bat – Zwei-Stufen-Pipeline (Windows 11)
REM
REM Windows-Aequivalent zu run_two_stage_pipeline.sh
REM
REM Stufe 1: HTF-Bias auf H1
REM Stufe 2: LTF-Entry auf M5
REM
REM Ausfuehren auf Windows 11:
REM     Doppelklick oder in PowerShell: .\run_two_stage_pipeline.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

set "SYMBOLS=USDCAD USDJPY"
set "VERSION=v4"
set "LTF_TIMEFRAME=M5"

REM Suffix fuer Dateinamen
if "%VERSION%"=="v1" (
    set "H1_SUFFIX="
    set "LTF_SUFFIX="
) else (
    set "H1_SUFFIX=_%VERSION%"
    set "LTF_SUFFIX=_%VERSION%"
)

REM Python-Interpreter pruefen
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python-Interpreter nicht gefunden: %PYTHON_BIN%
    echo Bitte zuerst virtuelle Umgebung einrichten.
    pause
    exit /b 1
)

echo [1/2] Pruefe Eingabedaten
for %%S in (%SYMBOLS%) do (
    if not exist "%ROOT_DIR%\data\%%S_H1_labeled!H1_SUFFIX!.csv" (
        echo [FEHLER] Fehlende Datei: data\%%S_H1_labeled!H1_SUFFIX!.csv
        pause
        exit /b 1
    )
    if not exist "%ROOT_DIR%\data\%%S_%LTF_TIMEFRAME%_labeled!LTF_SUFFIX!.csv" (
        echo [FEHLER] Fehlende Datei: data\%%S_%LTF_TIMEFRAME%_labeled!LTF_SUFFIX!.csv
        echo Bitte zuerst Feature-Engineering + Labeling fuer %LTF_TIMEFRAME% ausfuehren.
        echo Tipp: run_m5_pipeline.bat
        pause
        exit /b 1
    )
)

echo [2/2] Trainiere Zwei-Stufen-Modelle
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S ^| HTF=H1 ^| LTF=%LTF_TIMEFRAME% ^| Version=%VERSION%
    "%PYTHON_BIN%" "%ROOT_DIR%\train_two_stage.py" --symbol %%S --ltf_timeframe %LTF_TIMEFRAME% --version %VERSION%
    if errorlevel 1 (
        echo [FEHLER] Two-Stage Training fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo.
echo Fertig. Gespeicherte Artefakte in models\:
echo   - lgbm_htf_bias_SYMBOL_H1_%VERSION%.pkl
echo   - lgbm_ltf_entry_SYMBOL_%LTF_TIMEFRAME%_%VERSION%.pkl
echo   - two_stage_SYMBOL_%LTF_TIMEFRAME%_%VERSION%.json
echo.
pause
endlocal
exit /b 0
