@echo off
REM =============================================================================
REM start_testphase_topconfig_H1_M15.bat – Aktive Paper-Trading Top-Konfiguration
REM
REM Laeuft auf: Windows 11 Laptop (NICHT auf dem Linux-Server)
REM
REM Zweck:
REM   Startet den Live-Trader im Paper-Modus mit der aktuell besten
REM   Konfiguration (Two-Stage v4, H1-Bias + M15-Entry).
REM   Dies ist die HAUPT-Startdatei fuer den laufenden Betrieb (Phase 7).
REM
REM Konfiguration (Stand: 14.03.2026):
REM   - Beide Symbole: USDCAD + USDJPY auf v4
REM   - Two-Stage: H1-Bias + M15-Entry (Kongruenzfilter AKTIV)
REM   - Schwelle: 50%% (Long/Short via long_prob-Mapping)
REM   - Regime-Quelle: market_regime_hmm (hmmlearn-basiert)
REM   - Regime-Filter: alle Regime erlaubt (0=Seitwaerts,1=Auf,2=Ab,3=HighVola)
REM   - ATR-SL: 2.0x ATR_14 (dynamischer Stop-Loss)
REM   - TP/SL: 0.6%% / 0.3%% (RRR 2:1)
REM   - Cooldown: 3 Bars zwischen Trades
REM   - Nach Start: 5 Kerzen nur beobachten (Aufwaermphase)
REM   - Neutraler H1-Bias: ERLAUBT M15-Entries (K1-Test)
REM   - Spread-Gate: max 2.0 Pips (Trade wird blockiert wenn hoeher)
REM   - Kill-Switch: Drawdown > 15%% → automatischer Stopp
REM   - Startkapital: 10.000 (fuer Kill-Switch-Berechnung)
REM   - Heartbeat: aktiv (CSV-Update pro Kerze fuer Monitoring)
REM   - Log-Ordner: logs/ (Standard, kein Unterordner)
REM
REM Voraussetzung:
REM   - MT5 Terminal laeuft und ist angemeldet (Demo-Konto)
REM   - Virtuelle Umgebung existiert (venv/)
REM   - Modelle v4 nach models/ deployed (deploy_to_laptop.sh)
REM   - Sync-Task registriert (register_test128_log_sync_to_server.bat)
REM
REM Zum Stoppen:
REM   Ctrl+C in jedem Fenster druecken.
REM =============================================================================

setlocal

REM --- Pfade ---
set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
set "PYTHON_EXE=%BASE_DIR%\.venv\Scripts\python.exe"
set "TRADER_SCRIPT=%BASE_DIR%\live\live_trader.py"

REM ========================================================
REM Fest eingebettete MT5-Zugangsdaten fuer die Top-Konfiguration
REM Hinweis: Investor-Passwort wird von live_trader.py nicht
REM verwendet, bleibt hier aber zur Referenz hinterlegt.
REM ========================================================
set "MT5_ACCOUNT_NAME=Sebastian Setnescu"
set "MT5_ACCOUNT_TYPE=Demo Allocations LTD"
set "MT5_SERVER=SwissquoteLtd-Server"
set "MT5_LOGIN=6202835"
set "MT5_PASSWORD=*0YsQqAk"
set "MT5_INVESTOR=ZcEpDl@8"

REM --- Voraussetzungen pruefen ---
if not exist "%BASE_DIR%" (
    echo [FEHLER] Projektordner nicht gefunden: %BASE_DIR%
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    if exist "%BASE_DIR%\venv\Scripts\python.exe" (
        echo [WARNUNG] .venv-Python nicht gefunden, nutze Legacy-Pfad venv\Scripts\python.exe
        set "PYTHON_EXE=%BASE_DIR%\venv\Scripts\python.exe"
    ) else (
        echo [FEHLER] Kein Projekt-Python gefunden.
        echo Erwartet: %BASE_DIR%\.venv\Scripts\python.exe
        echo Optionaler Altpfad: %BASE_DIR%\venv\Scripts\python.exe
        echo.
        echo Loesung auf Windows:
        echo   cd /d %BASE_DIR%
        echo   .\setup_windows.bat
        echo.
        echo Alternativ manuell:
        echo   python -m venv .venv
        echo   .\.venv\Scripts\pip install -r requirements-laptop.txt
        pause
        exit /b 1
    )
)

if not exist "%TRADER_SCRIPT%" (
    echo [FEHLER] Script nicht gefunden: %TRADER_SCRIPT%
    pause
    exit /b 1
)

if "%MT5_SERVER%"=="" goto :env_error
if "%MT5_LOGIN%"=="" goto :env_error
if "%MT5_PASSWORD%"=="" goto :env_error

cd /d "%BASE_DIR%"

REM --- Startinfo anzeigen ---
echo ========================================================
echo   MT5 Testphase - TOP-Konfiguration (Paper)
echo   USDCAD + USDJPY ^| Two-Stage v4 (H1-Bias + M15-Entry)
echo   Schwelle=50%% ^| Mapping=long_prob ^| TP=0.6%% ^| SL=0.3%% ^| ATR-SL=2.0x
echo   Cooldown=3 Bars ^| Regime=0,1,2,3 ^| Quelle=HMM
echo   Kill-Switch=15%% DD ^| Startkapital=10.000
echo   Konto: %MT5_ACCOUNT_NAME% ^| %MT5_ACCOUNT_TYPE%
echo   Server: %MT5_SERVER% ^| Login: %MT5_LOGIN%
echo   Python: %PYTHON_EXE%
echo   Log-Ordner: %BASE_DIR%\logs
echo ========================================================

REM --- Fenster 1: USDCAD (Paper, Two-Stage H1+M15, v4) ---
echo [INFO] Starte USDCAD...
start "MT5-Testphase-USDCAD-v4" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDCAD --version v4 --paper_trading 1 --schwelle 0.50 --short_schwelle 0.50 --decision_mapping long_prob --regime_source market_regime_hmm --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 2.0 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_mode primary --two_stage_ltf_timeframe M15 --two_stage_version v4 --two_stage_htf_schwelle 0.35 --two_stage_ltf_schwelle 0.50 --two_stage_kongruenz 1 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --log_dir "%BASE_DIR%\logs" --kill_switch_dd 0.15 --kapital_start 10000 --max_spread_pips 2.0 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM --- Fenster 2: USDJPY (Paper, Two-Stage H1+M15, v4) ---
echo [INFO] Starte USDJPY...
start "MT5-Testphase-USDJPY-v4" "%ComSpec%" /k ""%PYTHON_EXE%" "%TRADER_SCRIPT%" --symbol USDJPY --version v4 --paper_trading 1 --schwelle 0.50 --short_schwelle 0.50 --decision_mapping long_prob --regime_source market_regime_hmm --regime_filter 0,1,2,3 --atr_sl 1 --atr_faktor 2.0 --lot 0.01 --tp_pct 0.006 --sl_pct 0.003 --two_stage_enable 1 --two_stage_mode primary --two_stage_ltf_timeframe M15 --two_stage_version v4 --two_stage_htf_schwelle 0.35 --two_stage_ltf_schwelle 0.50 --two_stage_kongruenz 1 --two_stage_allow_neutral_htf 1 --two_stage_cooldown_bars 3 --startup_observation_bars 5 --heartbeat_log 1 --log_dir "%BASE_DIR%\logs" --kill_switch_dd 0.15 --kapital_start 10000 --max_spread_pips 2.0 --mt5_server "%MT5_SERVER%" --mt5_login "%MT5_LOGIN%" --mt5_password "%MT5_PASSWORD%""

REM --- Abschluss-Info ---
echo.
echo [OK] Beide Trader gestartet.
echo.
echo   Fenster 1: USDCAD v4 (Two-Stage H1-M15, Paper)
echo   Fenster 2: USDJPY v4 (Two-Stage H1-M15, Paper)
echo.
echo   Logs werden in %BASE_DIR%\logs gespeichert.
echo   Dateien: USDCAD_signals.csv, USDJPY_signals.csv, live_trader.log
echo   Zum Stoppen: Ctrl+C in jedem Fenster druecken.
echo.
pause
exit /b 0

:env_error
echo [FEHLER] MT5_SERVER/MT5_LOGIN/MT5_PASSWORD muessen gesetzt sein.
pause
exit /b 1
