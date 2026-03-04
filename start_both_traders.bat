@echo off
REM =============================================================================
REM start_both_traders.bat – Startet beide Paper-Trader parallel
REM
REM Ausführen auf Windows 11 Laptop:
REM     Doppelklick auf diese Datei ODER in PowerShell: .\start_both_traders.bat
REM
REM Voraussetzung: 
REM     - MT5 Terminal läuft und ist angemeldet
REM     - Virtuelle Umgebung existiert (venv/)
REM =============================================================================

echo ================================================
echo   MT5 ML-Trading - Starte beide Trader
echo   USDCAD + USDJPY (Paper-Trading)
echo ================================================
echo.

cd /d "C:\Users\Sebastian Setnescu\mt5_trading"

REM Fenster 1: USDCAD (Single-Stage v4, Alle Regime, Schwelle 52%)
start "MT5-Trader-USDCAD" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDCAD --version v4 --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --mt5_server SwissquoteLtd-Server --mt5_login 6202835 --mt5_password *0YsQqAk"

REM Warte 5 Sekunden vor dem zweiten Start (MT5-Verbindung stabilisieren)
timeout /t 5 /nobreak >nul

REM Fenster 2: USDJPY (Two-Stage Shadow v4: H1+M5, Alle Regime, Schwelle 52%)
start "MT5-Trader-USDJPY" cmd /k "cd /d "C:\Users\Sebastian Setnescu\mt5_trading" && venv\Scripts\activate.bat && python live\live_trader.py --symbol USDJPY --version v4 --schwelle 0.52 --regime_filter 0,1,2 --atr_sl 1 --atr_faktor 1.5 --lot 0.01 --two_stage_enable 1 --two_stage_ltf_timeframe M5 --two_stage_version v4 --mt5_server SwissquoteLtd-Server --mt5_login 6202835 --mt5_password *0YsQqAk"

echo.
echo ✅ Beide Trader gestartet!
echo.
echo Fenster 1: USDCAD (Single-Stage v4, Schwelle 52%%, Alle Regime)
echo Fenster 2: USDJPY (Two-Stage Shadow v4: H1+M5, Schwelle 52%%, Alle Regime)
echo.
echo Zum Stoppen: Ctrl+C in jedem Fenster druecken
echo.
pause
