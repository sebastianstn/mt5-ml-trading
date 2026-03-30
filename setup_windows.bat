@echo off
REM =============================================================================
REM setup_windows.bat – Projekt-Setup auf Windows 11
REM
REM Erstellt virtuelle Umgebung und installiert alle Abhaengigkeiten.
REM
REM Verwendung:
REM     1. Python 3.9+ installieren (python.org → "Add to PATH" aktivieren!)
REM     2. Dieses Skript per Doppelklick oder PowerShell ausfuehren
REM
REM Ergebnis:
REM     - .venv\ mit allen Paketen
REM     - pandas_ta separat installiert (ohne numba)
REM     - Ordner data\, models\, plots\, logs\, reports\ erstellt
REM =============================================================================

setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

echo ==========================================================
echo   MT5 ML-Trading-System – Windows 11 Setup
echo ==========================================================
echo.

REM --- Python pruefen ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python nicht gefunden!
    echo Bitte Python 3.9+ von python.org installieren.
    echo WICHTIG: "Add Python to PATH" aktivieren!
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PY_VERSION=%%V"
echo [OK] Python %PY_VERSION% gefunden.
echo.

REM --- Virtuelle Umgebung erstellen ---
if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    echo [INFO] Virtuelle Umgebung existiert bereits: .venv\
    echo        Zum Neuerstellen: rmdir /s /q .venv und Skript erneut ausfuehren.
) else (
    echo [INFO] Erstelle virtuelle Umgebung ^(.venv^)...
    python -m venv "%ROOT_DIR%\.venv"
    if errorlevel 1 (
        echo [FEHLER] Konnte virtuelle Umgebung nicht erstellen!
        pause
        exit /b 1
    )
    echo [OK] Virtuelle Umgebung erstellt.
)
echo.

set "PIP=%ROOT_DIR%\.venv\Scripts\pip.exe"
set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"

REM --- pip aktualisieren ---
echo [INFO] Aktualisiere pip...
"%PYTHON_BIN%" -m pip install --upgrade pip
echo.

REM --- Abhaengigkeiten installieren ---
echo [INFO] Installiere Abhaengigkeiten aus requirements-laptop.txt...
"%PIP%" install -r "%ROOT_DIR%\requirements-laptop.txt"
if errorlevel 1 (
    echo [FEHLER] Installation fehlgeschlagen!
    echo Tipp: Einzelne Pakete manuell installieren falls ein Paket Probleme macht.
    pause
    exit /b 1
)
echo [OK] Hauptpakete installiert.
echo.

REM --- pandas_ta separat (ohne numba) ---
echo [INFO] Installiere pandas_ta ^(--no-deps, ohne numba^)...
"%PIP%" install pandas_ta --no-deps
if errorlevel 1 (
    echo [WARNUNG] pandas_ta konnte nicht installiert werden.
    echo           Live-Trading funktioniert trotzdem (Fallback-Indikatoren).
)
echo [OK] pandas_ta installiert.
echo.

REM --- Ordner erstellen ---
echo [INFO] Erstelle Projektordner...
if not exist "%ROOT_DIR%\data" mkdir "%ROOT_DIR%\data"
if not exist "%ROOT_DIR%\models" mkdir "%ROOT_DIR%\models"
if not exist "%ROOT_DIR%\plots" mkdir "%ROOT_DIR%\plots"
if not exist "%ROOT_DIR%\logs" mkdir "%ROOT_DIR%\logs"
if not exist "%ROOT_DIR%\reports" mkdir "%ROOT_DIR%\reports"
echo [OK] Ordner erstellt.
echo.

REM --- Installierte Versionen anzeigen ---
echo ==========================================================
echo   Installierte Versionen:
echo ==========================================================
"%PYTHON_BIN%" -c "import pandas; print(f'  pandas:       {pandas.__version__}')"
"%PYTHON_BIN%" -c "import numpy; print(f'  numpy:        {numpy.__version__}')"
"%PYTHON_BIN%" -c "import lightgbm; print(f'  lightgbm:     {lightgbm.__version__}')"
"%PYTHON_BIN%" -c "import xgboost; print(f'  xgboost:      {xgboost.__version__}')"
"%PYTHON_BIN%" -c "import sklearn; print(f'  scikit-learn:  {sklearn.__version__}')"
"%PYTHON_BIN%" -c "import optuna; print(f'  optuna:       {optuna.__version__}')"
"%PYTHON_BIN%" -c "import joblib; print(f'  joblib:       {joblib.__version__}')"
"%PYTHON_BIN%" -c "import matplotlib; print(f'  matplotlib:   {matplotlib.__version__}')"
"%PYTHON_BIN%" -c "try: import MetaTrader5 as mt5; print(f'  MetaTrader5:  {mt5.__version__}')" 2>nul
"%PYTHON_BIN%" -c "except: print('  MetaTrader5:  nicht installiert')" 2>nul
echo.

echo ==========================================================
echo   Setup abgeschlossen!
echo ==========================================================
echo.
echo   Aktivieren:
echo     .venv\Scripts\activate
echo.
echo   Naechste Schritte:
echo     1. .env-Datei erstellen (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
echo     2. Daten laden: python data_loader.py
echo     3. Features berechnen: python features\feature_engineering.py --symbol USDCAD
echo     4. Training: python train_model.py --symbol USDCAD
echo.
pause
endlocal
exit /b 0
