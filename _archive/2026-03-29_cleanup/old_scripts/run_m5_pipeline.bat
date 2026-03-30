@echo off
REM =============================================================================
REM run_m5_pipeline.bat – M5-Datenkette fuer Option 1 (HTF=H1 / LTF=M5)
REM
REM Windows-Aequivalent zu run_m5_pipeline.sh
REM
REM Schritte:
REM   1) Prueft Rohdaten data\SYMBOL_M5.csv
REM   2) Berechnet M5-Features
REM   3) Labelt M5-Daten im ATR-Modus (v4)
REM
REM Ausfuehren auf Windows 11:
REM     Doppelklick oder in PowerShell: .\run_m5_pipeline.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
REM Trailing Backslash entfernen
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
    set "PYTHON_BIN=python"
)

set "SYMBOLS=USDCAD USDJPY"
set "VERSION=v4"
set "TIMEFRAME=M5"
set "HORIZON=5"
set "ATR_FAKTOR=1.5"

REM Python-Interpreter pruefen
"%PYTHON_BIN%" --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python-Interpreter nicht gefunden: %PYTHON_BIN%
    echo Bitte zuerst virtuelle Umgebung einrichten.
    pause
    exit /b 1
)

echo [1/4] Pruefe M5-Rohdaten
for %%S in (%SYMBOLS%) do (
    if not exist "%ROOT_DIR%\data\%%S_%TIMEFRAME%.csv" (
        echo [FEHLER] Fehlende Rohdaten: data\%%S_%TIMEFRAME%.csv
        echo Bitte ausfuehren: python data_loader.py --symbol %%S --timeframe %TIMEFRAME% --bars 30000
        pause
        exit /b 1
    )
)

echo [2/4] Feature Engineering (%TIMEFRAME%)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\features\feature_engineering.py" --symbol %%S --timeframe %TIMEFRAME%
    if errorlevel 1 (
        echo [FEHLER] Feature Engineering fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [3/4] Labeling (%TIMEFRAME%, %VERSION%, Modus=ATR)
for %%S in (%SYMBOLS%) do (
    echo   -^> %%S
    "%PYTHON_BIN%" "%ROOT_DIR%\features\labeling.py" --symbol %%S --timeframe %TIMEFRAME% --version %VERSION% --modus atr --atr_faktor %ATR_FAKTOR% --horizon %HORIZON%
    if errorlevel 1 (
        echo [FEHLER] Labeling fuer %%S fehlgeschlagen!
        pause
        exit /b 1
    )
)

echo [4/4] Pruefe erzeugte labeled-Dateien
for %%S in (%SYMBOLS%) do (
    if not exist "%ROOT_DIR%\data\%%S_%TIMEFRAME%_labeled_%VERSION%.csv" (
        echo [FEHLER] Erwartete Datei fehlt: %%S_%TIMEFRAME%_labeled_%VERSION%.csv
        pause
        exit /b 1
    )
    echo   OK: %%S_%TIMEFRAME%_labeled_%VERSION%.csv
)

echo.
echo Fertig. Naechster Schritt:
echo   run_two_stage_pipeline.bat
echo.
echo Erwartete Two-Stage-Artefakte in models\:
echo   - lgbm_htf_bias_SYMBOL_H1_%VERSION%.pkl
echo   - lgbm_ltf_entry_SYMBOL_%TIMEFRAME%_%VERSION%.pkl
echo   - two_stage_SYMBOL_%TIMEFRAME%_%VERSION%.json
echo.
pause
endlocal
exit /b 0
